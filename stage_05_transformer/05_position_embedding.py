import math
import torch
from torch import nn

torch.manual_seed(42)
device = torch.device("cuda")


class PositionDemo(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(4, 8)
        self.position_embedding = nn.Embedding(4, 8)

        self.query = nn.Linear(8, 4, bias=False)
        self.key = nn.Linear(8, 4, bias=False)
        self.value = nn.Linear(8, 8, bias=False)
        self.head = nn.Linear(8, 4)

        self.register_buffer(
            "allowed",
            torch.tril(torch.ones(4, 4, dtype=torch.bool)),
        )

    def forward(self, idx, use_positions):
        _, T = idx.shape
        h = self.token_embedding(idx)

        if use_positions:
            positions = torch.arange(T, device=idx.device)
            h = h + self.position_embedding(positions)

        q = self.query(h)
        k = self.key(h)
        v = self.value(h)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(
            ~self.allowed[:T, :T], float("-inf")
        )

        weights = scores.softmax(dim=-1)
        return self.head(weights @ v), weights


model = PositionDemo().to(device)

# abcd 与 acbd：末字符相同，前文顺序不同。
idx = torch.tensor(
    [[0, 1, 2, 3], [0, 2, 1, 3]],
    device=device,
)

with torch.no_grad():
    without, _ = model(idx, use_positions=False)
    with_pos, weights = model(idx, use_positions=True)

    p = model.position_embedding(
        torch.arange(4, device=device)
    )

print("位置向量形状：", p.shape)
print("输入词向量形状：", model.token_embedding(idx).shape)

print(
    "无位置时，最后位置logits相同：",
    torch.allclose(
        without[0, -1], without[1, -1], atol=1e-6
    ),
)
print(
    "无位置时，最大差：",
    (without[0, -1] - without[1, -1]).abs().max().item(),
)

print(
    "有位置时，最后位置logits相同：",
    torch.allclose(
        with_pos[0, -1], with_pos[1, -1], atol=1e-6
    ),
)
print(
    "有位置时，最大差：",
    (with_pos[0, -1] - with_pos[1, -1]).abs().max().item(),
)

print("abcd最后位置权重：", weights[0, -1].cpu())
print("acbd最后位置权重：", weights[1, -1].cpu())