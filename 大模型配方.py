from dataclasses import dataclass
from typing import Optional, Tuple
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# Utilities & Config
# ------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layers: int = 12
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 4 * 512
    rope_theta: float = 10000.0  # RoPE base
    dropout: float = 0.0
    max_seq_len: int = 1024

@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 200
    max_steps: int = 2000
    grad_clip: float = 1.0
    grad_accum_steps: int = 4
    batch_size: int = 4  # per-device micro-batch
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path: str = "ckpt.pt"

# ------------------------------
# RoPE (Rotary Positional Embedding)
# ------------------------------

class Rotary(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        # dim must be even; we'll apply to half-dims per head subspace
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0):
        # x: [B, T, H, D] or [T, B, H, D]; apply along last dim pairs
        # create phase angles
        t = torch.arange(offset, offset + seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,d->td", t, self.inv_freq)  # [T, D/2]
        cos = freqs.cos().unsqueeze(-1)
        sin = freqs.sin().unsqueeze(-1)
        # interleave to match last dim pairs
        # helper to rotate (a,b) -> (a*cos - b*sin, a*sin + b*cos)
        x_ = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1, x2 = x_[..., 0], x_[..., 1]
        x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot.flatten(-2)

# ------------------------------
# Attention + MLP Blocks (Pre-LN)
# ------------------------------

class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, rope_theta: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = Rotary(self.d_head, theta=rope_theta)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to heads
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head)
        q, k, v = map(split_heads, (q, k, v))
        # apply RoPE to q, k
        q = self.rope(q, seq_len=T)
        k = self.rope(k, seq_len=T)
        # scaled dot-product attention
        att = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.d_head)
        if attn_mask is not None:
            att = att + attn_mask  # mask should be additive with -inf on disallowed positions
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = torch.einsum("bhts,bshd->bthd", att, v).contiguous()
        y = y.view(B, T, -1)
        y = self.resid_drop(self.proj(y))
        return y

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MHA(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.rope_theta)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

# ------------------------------
# Causal LM Model
# ------------------------------

class CausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight

    def _causal_mask(self, T: int, device):
        # Float mask with -inf above diagonal, for additive masking
        mask = torch.full((T, T), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        # idx: [B, T]
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = self.pos_drop(x)
        attn_mask = self._causal_mask(T, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]
        loss = None
        if targets is not None:
            # standard next-token CE loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ------------------------------
# AdamW with weight-decay exclusion (bias, LayerNorm)
# ------------------------------

def build_optimizer(model: nn.Module, tcfg: TrainConfig):
    decay, no_decay = set(), set()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith("bias") or 'ln' in n.lower() or 'layernorm' in n.lower():
            no_decay.add(n)
        else:
            decay.add(n)
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    optim_groups = [
        {"params": [param_dict[n] for n in sorted(list(decay))], "weight_decay": tcfg.weight_decay},
        {"params": [param_dict[n] for n in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=tcfg.lr, betas=tcfg.betas, eps=tcfg.eps)
    return optimizer

# ------------------------------
# Cosine LR with warmup (by steps)
# ------------------------------

def build_scheduler(optimizer: torch.optim.Optimizer, warmup: int, max_steps: int):
    def lr_lambda(step):
        if step < warmup:
            return max(step / float(max(1, warmup)), 1e-6)
        # cosine from 1 -> 0 over remaining steps
        progress = (step - warmup) / float(max(1, max_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ------------------------------
# Toy Dataset (replace with your own)
# ------------------------------

class ToyTextDataset(Dataset):
    """A tiny character-level dataset to make the script runnable.
    Replace with a real tokenized dataset for actual training.
    """
    def __init__(self, text: str, seq_len: int, vocab: Optional[str] = None):
        super().__init__()
        self.seq_len = seq_len
        if vocab is None:
            vocab = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.ids = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def __len__(self):
        return max(1, len(self.ids) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.ids[idx: idx + self.seq_len]
        y = self.ids[idx + 1: idx + 1 + self.seq_len]
        return x, y

# ------------------------------
# Training Loop (AMP + accumulation + clipping)
# ------------------------------

def train_one_run():
    raw_text = (
        "To be, or not to be, that is the question:\n"
        "Whether 'tis nobler in the mind to suffer\n"
        "The slings and arrows of outrageous fortune,\n"
    ) * 64

    mcfg = ModelConfig(max_seq_len=128)
    dataset = ToyTextDataset(raw_text, seq_len=mcfg.max_seq_len)
    mcfg.vocab_size = dataset.vocab_size

    model = CausalLM(mcfg)
    tcfg = TrainConfig()

    device = torch.device(tcfg.device)
    model.to(device)

    loader = DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=True, drop_last=True)
    optimizer = build_optimizer(model, tcfg)
    scheduler = build_scheduler(optimizer, tcfg.warmup_steps, tcfg.max_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=tcfg.amp and device.type == 'cuda')

    model.train()
    step = 0
    accum = 0
    running = 0.0
    t0 = time.time()

    while step < tcfg.max_steps:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.cuda.amp.autocast(enabled=tcfg.amp and device.type == 'cuda'):
                logits, loss = model(xb, yb)
                loss = loss / tcfg.grad_accum_steps
            scaler.scale(loss).backward()
            running += loss.item()
            accum += 1

            if accum % tcfg.grad_accum_steps == 0:
                if tcfg.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                step += 1
                if step % 50 == 0:
                    dt = time.time() - t0
                    lr = scheduler.get_last_lr()[0]
                    print(f"step {step:5d} | loss {running * tcfg.grad_accum_steps / 50:.4f} | lr {lr:.2e} | {dt:.1f}s")
                    running = 0.0
                    t0 = time.time()

                if step >= tcfg.max_steps:
                    break

    # save checkpoint
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': mcfg,
        'train_config': tcfg,
    }
    torch.save(ckpt, tcfg.ckpt_path)
    print(f"Saved checkpoint to {tcfg.ckpt_path}")


if __name__ == "__main__":
    train_one_run()
