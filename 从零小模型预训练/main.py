import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import math
print("当前工作目录:", os.getcwd())
# -----------------------------
# 数据集准备
# -----------------------------
# 读取训练文本
with open("凡人修仙传.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 构建字符级词表
vocab = sorted(set(text))  # 所有唯一字符
stoi = {ch:i for i,ch in enumerate(vocab)}  # char -> index
itos = {i:ch for i,ch in enumerate(vocab)}  # index -> char
vocab_size = len(vocab)

# 将文本编码成索引序列
ids = [stoi[c] for c in text]
# 划分训练集和验证集
train_ids = torch.tensor(ids[:int(0.9*len(ids))], dtype=torch.long)
val_ids = torch.tensor(ids[int(0.9*len(ids)):], dtype=torch.long)

# 自定义 Dataset 类，负责生成 (x, y) 对
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size  # 每个样本长度为 block_size
    def __getitem__(self, idx):
        # 输入 x 是当前片段，目标 y 是向右偏移一位的下一个字符
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+1+self.block_size]
        return x, y

block_size = 128  # 序列长度
train_dataset = CharDataset(train_ids, block_size)
val_dataset = CharDataset(val_ids, block_size)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# -----------------------------
# Transformer 模型组件
# -----------------------------
d_model = 128  # embedding 维度
n_head = 2     # 注意力头数
d_ff = 512     # 前馈网络隐藏维度
n_layer = 2    # Transformer 层数

# 生成因果 mask，保证模型只能看到当前位置及之前的 token
def causal_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0)  # [1, T, T]

# 多头自注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head  # 每个头的维度
        self.qkv = nn.Linear(d_model, d_model*3)  # 一次生成 Q, K, V
        self.proj = nn.Linear(d_model, d_model)   # 输出投影

    def forward(self, x):
        B, T, C = x.size()
        # 线性投影后 reshape 成多头格式
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.d_k).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, head, T, d_k]
        # 计算注意力得分
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_k)
        mask = causal_mask(T).to(att.device)
        att = att.masked_fill(mask==0, float('-inf'))  # 上三角置 -inf
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1,2).reshape(B,T,C)
        return self.proj(out)  # 投回原维度

# 前馈网络模块
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)

# 单个 Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  # 前 LayerNorm
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)  # 后 LayerNorm
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x):
        # 残差连接 + 注意力
        x = x + self.attn(self.ln1(x))
        # 残差连接 + 前馈
        x = x + self.ff(self.ln2(x))
        return x

# Toy GPT 模型
class ToyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, d_ff, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_emb = nn.Embedding(block_size, d_model)    # 位置 embedding
        self.blocks = nn.Sequential(*[TransformerBlock(d_model,n_head,d_ff) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)  # 输出 LayerNorm
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # LM head
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)  # token + position
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # 交叉熵计算损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -----------------------------
# 初始化模型与优化器
# -----------------------------
model = ToyGPT(vocab_size, d_model, n_layer, n_head, d_ff, block_size).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
start_epoch = 0
# 如果toy_gpt.pth权重文件存在 加载权重文件
if os.path.exists("checkpoint.pth"):
    checkpoint = torch.load("checkpoint.pth")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1

# 使用 AMP 混合精度训练
scaler = torch.amp.GradScaler('cuda')

# -----------------------------
# 训练循环
# -----------------------------
total_epochs = 5
for epoch in range(start_epoch, total_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (xb, yb) in enumerate(train_loader, 1):
        xb, yb = xb.cuda(), yb.cuda()
        optimizer.zero_grad()
        # 混合精度 forward
        with torch.amp.autocast(device_type='cuda'):
            logits, loss = model(xb, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        # 每 50 个 batch 打印一次进度
        if batch_idx % 500 == 0:
            avg_loss = running_loss / 500
            print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, Loss: {avg_loss:.4f}")
            running_loss = 0.0
    print(f"Epoch {epoch+1} finished")
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, "checkpoint.pth")


# -----------------------------
# 测试生成文本
# -----------------------------
model.eval()
context = torch.tensor([stoi[c] for c in "韩立是谁"], dtype=torch.long).unsqueeze(0).cuda()
with torch.no_grad():
    for _ in range(50):
        logits, _ = model(context)
        next_id = torch.argmax(logits[0,-1,:])  # 选取概率最高的 token
        context = torch.cat([context, next_id.unsqueeze(0).unsqueeze(0)], dim=1)

print("生成结果:", ''.join([itos[i] for i in context[0].tolist()]))
