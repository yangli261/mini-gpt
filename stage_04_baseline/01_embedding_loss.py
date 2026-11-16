import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(42)
device = torch.device("cuda")

chars = list("abcd")
stoi = {c: i for i, c in enumerate(chars)}
vocab_size = len(chars)


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        return self.table(idx)


model = BigramModel(vocab_size).to(device)

# 第一行：abc -> bcd
# 第二行：bcd -> cda
x = torch.tensor(
    [[0, 1, 2], [1, 2, 3]],
    dtype=torch.long,
    device=device,
)
y = torch.tensor(
    [[1, 2, 3], [2, 3, 0]],
    dtype=torch.long,
    device=device,
)

logits = model(x)
B, T, V = logits.shape

print("参数表形状：", model.table.weight.shape)
print("输入形状：", x.shape)
print("目标形状：", y.shape)
print("logits形状：", logits.shape)

print(
    "查表一致：",
    torch.allclose(logits[0, 0], model.table.weight[0]),
)
print(
    "相同字符分数一致：",
    torch.allclose(logits[0, 1], logits[1, 0]),
)

# 将每个位置看作一个分类样本。
loss = F.cross_entropy(
    logits.reshape(B * T, V),
    y.reshape(B * T),
)
print("交叉熵：", loss.item())

# 只为观察而转换成概率。
with torch.no_grad():
    probs = logits.softmax(dim=-1)
    target_id = y[0, 0].item()
    first_loss = -probs[0, 0, target_id].log()

    print("首位置各字符概率：", probs[0, 0].cpu().tolist())
    print("首位置概率之和：", probs[0, 0].sum().item())
    print("首位置目标字符：", chars[target_id])
    print("首位置损失：", first_loss.item())

loss.backward()
print("参数梯度形状：", model.table.weight.grad.shape)