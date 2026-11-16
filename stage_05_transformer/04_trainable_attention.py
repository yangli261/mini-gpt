import math
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(42)
device = torch.device("cuda")


class TinyAttentionLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(4, 8)
        self.query = nn.Linear(8, 4, bias=False)
        self.key = nn.Linear(8, 4, bias=False)
        self.value = nn.Linear(8, 8, bias=False)
        self.head = nn.Linear(8, 4)

        self.register_buffer(
            "allowed",
            torch.tril(torch.ones(4, 4, dtype=torch.bool)),
        )

    def forward(self, idx):
        _, T = idx.shape

        x = self.embedding(idx)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(
            ~self.allowed[:T, :T], float("-inf")
        )

        weights = scores.softmax(dim=-1)
        context = weights @ v

        return self.head(context), weights


model = TinyAttentionLM().to(device)

# abcd -> bcda；acbd -> cbda。
x = torch.tensor(
    [[0, 1, 2, 3], [0, 2, 1, 3]],
    device=device,
)
y = torch.tensor(
    [[1, 2, 3, 0], [2, 1, 3, 0]],
    device=device,
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

optimizer.zero_grad(set_to_none=True)

logits, weights = model(x)
loss = F.cross_entropy(
    logits.reshape(-1, 4),
    y.reshape(-1),
)

print("logits形状：", logits.shape)
print("第一条序列的注意力：\n", weights[0].detach().cpu())
print("更新前损失：", loss.item())

loss.backward()

print("\n各参数的梯度范数：")
for name, parameter in model.named_parameters():
    norm = (
        None
        if parameter.grad is None
        else parameter.grad.norm().item()
    )
    print(name, norm)

optimizer.step()

with torch.no_grad():
    new_logits, new_weights = model(x)
    new_loss = F.cross_entropy(
        new_logits.reshape(-1, 4),
        y.reshape(-1),
    )

print("\n更新后损失：", new_loss.item())
print(
    "第一条序列最后位置的权重，更新前：",
    weights[0, -1].detach().cpu(),
)
print(
    "第一条序列最后位置的权重，更新后：",
    new_weights[0, -1].cpu(),
)