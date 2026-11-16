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


class DemoLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(4, 8)
        self.position = nn.Embedding(4, 8)
        self.attention = MultiHeadAttention()
        self.output = nn.Linear(8, 4)

    def forward(self, idx):
        positions = torch.arange(
            idx.shape[1], device=idx.device
        )
        h = self.token(idx) + self.position(positions)
        context, weights = self.attention(h)
        return self.output(context), weights


model = DemoLM().to(device)

x = torch.tensor(
    [[0, 1, 2, 3], [0, 2, 1, 3]], device=device
)
y = torch.tensor(
    [[1, 2, 3, 0], [2, 1, 3, 0]], device=device
)

logits, weights = model(x)

print("logits形状：", logits.shape)
print("权重形状：", weights.shape)

for r in range(2):
    print(
        f"头{r}，第一条序列权重：\n",
        weights[0, r].detach().cpu(),
    )

print("每行权重和：", weights.sum(dim=-1).detach().cpu())

loss = F.cross_entropy(
    logits.reshape(-1, 4), y.reshape(-1)
)
loss.backward()

print("损失：", loss.item())

for r, head in enumerate(model.attention.heads):
    print(
        f"头{r}的query梯度范数：",
        head.query.weight.grad.norm().item(),
    )

print(
    "输出投影梯度范数：",
    model.attention.projection.weight.grad.norm().item(),
)