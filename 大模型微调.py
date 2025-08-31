
# LoRA 微调脚手架（为单卡 4070Ti-S / 16GB 显存 适配）
# 要点：
# - 使用 bitsandbytes 低比特表示 (4-bit) 来加载大模型，降低显存占用
# - 使用 PEFT 的 LoRA 只训练少量参数（适合单卡微调）
# - 使用 transformers.Trainer 简化训练循环（也可改为自定义循环）
# - 对 16GB 显存的推荐超参数在脚本底部给出

from pathlib import Path
import os
import math
import torch
from dataclasses import dataclass, field

# HF / bitsandbytes / peft / datasets / transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -----------------------------
# 配置（用户可编辑）
# -----------------------------
@dataclass
class Config:
    model_name: str = "baichuan-inc/Baichuan-7B"  # 替换为你想微调的模型
    output_dir: str = "./lora_out"
    dataset_name: str = None  # e.g. 'wikitext', 或 None 表示使用本地文件 data.txt
    dataset_path: str = "./data.txt"  # 当 dataset_name 为 None 时使用

    # LoRA 超参（针对 16GB 卡的推荐）
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])  # 常见的 target modules

    # 训练超参
    per_device_train_batch_size: int = 1  # micro-batch
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 3e-4
    fp16: bool = True
    max_seq_length: int = 1024
    save_total_limit: int = 3

cfg = Config()

# -----------------------------
# 环境 & 依赖提示
# -----------------------------
# pip install -U "transformers>=4.35.0" accelerate datasets bitsandbytes peft
# 注意：bitsandbytes 需和 CUDA 版本匹配（你是 CUDA 12.8，安装时请参考 bitsandbytes 官方说明）

# -----------------------------
# 准备模型（4-bit + LoRA-ready）
# -----------------------------

print("准备 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
# 某些模型可能没有 pad token，强制设置
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("构建 BitsAndBytes 配置并加载模型（4-bit）...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# 从 HF hub 加载模型（4-bit），device_map 使用 auto 以便在单卡上放置参数
print(f"加载模型 {cfg.model_name} （4-bit）...")
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 在 k-bit 低比特模式下，准备模型以便能安全地应用 LoRA（会启用一些保护措施）
print("prepare_model_for_kbit_training...")
model = prepare_model_for_kbit_training(model)

# 配置 LoRA
print("配置 LoRA...")
lora_config = LoraConfig(
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    target_modules=cfg.target_modules,
    lora_dropout=cfg.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# 数据加载与处理
# -----------------------------
print("加载数据集...")
if cfg.dataset_name:
    dataset = load_dataset(cfg.dataset_name, split="train")
else:
    # 从本地文本文件读取（每行一个样本）
    path = Path(cfg.dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}. 请创建一个包含训练样本的 data.txt")
    dataset = load_dataset("text", data_files={"train": str(path)})["train"]

# 简单的 tokenizer map 函数
def tokenize_function(examples):
    # 'text' 字段名可能因数据集而异
    texts = examples["text"] if "text" in examples else examples
    return tokenizer(texts, truncation=True, max_length=cfg.max_seq_length)

# 使用 map 进行分批 tokenization
tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
)

# 数据整理（拼接/切分）——一种简单做法: 每条已 tokenized 的样本即为一个训练样本
# 如果是大语料，建议用连续流式拼接再切分以提高利用率

# DataCollator 用于动态 padding 及创建 labels
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -----------------------------
# Trainer 设置
# -----------------------------
print("设置 TrainingArguments 与 Trainer...")
training_args = TrainingArguments(
    output_dir=cfg.output_dir,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    num_train_epochs=cfg.num_train_epochs,
    learning_rate=cfg.learning_rate,
    fp16=cfg.fp16,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=cfg.save_total_limit,
    remove_unused_columns=False,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# -----------------------------
# 训练
# -----------------------------
print("开始训练 (LoRA 微调)...")
trainer.train()

# 保存 LoRA adapter（仅保存 PEFT 的 adapter 而非全模型）
print("保存 LoRA adapter...")
model.save_pretrained(cfg.output_dir)
print("完成。LoRA adapter 已保存到:", cfg.output_dir)

# 说明：
# - 训练后，你可以用 AutoModelForCausalLM.from_pretrained + PeftModel.from_pretrained 在推理时合并 LoRA 或直接加载 adapter。
# - 若要在推理时将 LoRA 权重合并回基础模型（merge），可参考 peft 文档。
