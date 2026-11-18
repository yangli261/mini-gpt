import math
import torch
from torch import nn
from torch.nn import functional as F

class AttentionHead(nn.Module):
    def __init__(self, channels, head_dim, max_length):
        super().__init__()

        self.query = nn.Linear(channels, head_dim, bias=False)
        self.key = nn.Linear(channels, head_dim, bias=False)
        self.value = nn.Linear(channels, head_dim, bias=False)

        self.register_buffer(
            "allowed",
            torch.tril(
                torch.ones(max_length, max_length, dtype=torch.bool)
            ),
        )

    def forward(self, h):
        T = h.shape[1]
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(
            ~self.allowed[:T, :T], float("-inf")
        )

        weights = scores.softmax(dim=-1)
        return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, channels=8, num_heads=2, max_length=4):
        super().__init__()
        assert channels % num_heads == 0
        head_dim = channels // num_heads

        self.heads = nn.ModuleList([
            AttentionHead(channels, head_dim, max_length)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            num_heads * head_dim, channels
        )

    def forward(self, h):
        results = [head(h) for head in self.heads]

        joined = torch.cat(
            [value for value, _ in results], dim=-1
        )
        weights = torch.stack(
            [weight for _, weight in results], dim=1
        )

        return self.projection(joined), weights

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, max_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(channels)
        self.attn = MultiHeadAttention(channels, num_heads, max_length)
        self.ln2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, h):
        update, _ = self.attn(self.ln1(h))
        h = h + update
        h = h + self.ffn(self.ln2(h))
        return h


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, max_length, channels, num_heads, num_layers):
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, channels)
        self.position_embedding = nn.Embedding(max_length, channels)
        self.blocks = nn.Sequential(*(
            TransformerBlock(channels, num_heads, max_length)
            for _ in range(num_layers)
        ))
        self.final_norm = nn.LayerNorm(channels)
        self.lm_head = nn.Linear(channels, vocab_size)

    def forward(self, idx):
        T = idx.shape[1]
        if not 1 <= T <= self.max_length:
            raise ValueError("Input length must be in [1, max_length]")
        positions = torch.arange(T, device=idx.device)
        h = self.token_embedding(idx) + self.position_embedding(positions)
        h = self.blocks(h)
        return self.lm_head(self.final_norm(h))