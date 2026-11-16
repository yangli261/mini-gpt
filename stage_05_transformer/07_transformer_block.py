import math
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(42)
device = torch.device("cuda")


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


B, T, C = 2, 4, 8
blocks = nn.Sequential(
    TransformerBlock(C, num_heads=2, max_length=T),
    TransformerBlock(C, num_heads=2, max_length=T),
).to(device)
head = nn.Linear(C, 4).to(device)

# 用假数据代替字符与位置嵌入，只检验 Block 的计算路径。
h = torch.randn(B, T, C, device=device)
targets = torch.tensor([[1, 2, 3, 0], [2, 1, 3, 0]], device=device)

features = blocks(h)
logits = head(features)
loss = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
loss.backward()

print("输入和堆叠后特征：", h.shape, features.shape)
print("logits：", logits.shape)
print("损失：", loss.item())
print("第一层 Query 梯度范数：", blocks[0].attn.heads[0].query.weight.grad.norm().item())
print("第一层 MLP 梯度范数：", blocks[0].ffn[0].weight.grad.norm().item())

with torch.no_grad():
    changed = h.clone()
    changed[:, -1, 0] += 5.0
    changed_logits = head(blocks(changed))
    print("改末位置后，前三个位置不变：",
          torch.allclose(logits[:, :-1], changed_logits[:, :-1], atol=1e-6))
    print("末位置也不变：",
          torch.allclose(logits[:, -1], changed_logits[:, -1], atol=1e-6))