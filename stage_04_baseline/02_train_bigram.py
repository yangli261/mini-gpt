from pathlib import Path
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(42)

# ---------- 1. 数据 ----------
root = Path(__file__).resolve().parent.parent
text = (
    root / "data" / "tinyshakespeare.txt"
).read_text(encoding="utf-8")

if not text.isascii():
    raise ValueError("本课要求 ASCII 文本")

# 与阶段三相同：ASCII 编号为 0～127。
V = 128
encode = lambda s: [ord(c) for c in s]
decode = lambda ids: "".join(chr(i) for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = len(data)

splits = {
    "train": data[:int(0.8 * n)],
    "val": data[int(0.8 * n):int(0.9 * n)],
    "test": data[int(0.9 * n):],
}

B, T = 32, 32
if any(len(part) <= T for part in splits.values()):
    raise ValueError("数据分区长度必须大于 T")

device = torch.device("cuda")
train_rng = torch.Generator().manual_seed(123)


def get_batch(split, rng):
    part = splits[split]
    starts = torch.randint(
        len(part) - T, (B,), generator=rng
    ).tolist()

    x = torch.stack([part[i:i + T] for i in starts])
    y = torch.stack([part[i + 1:i + T + 1] for i in starts])

    return x.to(device), y.to(device)


# ---------- 2. 模型与损失 ----------
class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(V, V)

        # 所有分数从 0 开始，对应均匀概率。
        nn.init.zeros_(self.table.weight)

    def forward(self, x):
        return self.table(x)


def batch_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(
        logits.reshape(-1, V),
        y.reshape(-1),
    )


@torch.no_grad()
def evaluate(model):
    model.eval()
    result = {}

    for split in ("train", "val"):
        # 每次评估相同批次，不干扰训练的随机采样。
        rng = torch.Generator().manual_seed(999)
        losses = []

        for _ in range(10):
            x, y = get_batch(split, rng)
            losses.append(batch_loss(model, x, y).item())

        result[split] = sum(losses) / len(losses)

    model.train()
    return result


# ---------- 3. 训练 ----------
model = BigramModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

max_steps = 1200
history = []

print("均匀预测的理论损失：", math.log(V))

for step in range(max_steps + 1):
    if step % 300 == 0:
        scores = evaluate(model)
        history.append({"step": step, **scores})

        print(
            f"step={step:4d} "
            f"train={scores['train']:.4f} "
            f"val={scores['val']:.4f}"
        )

    if step == max_steps:
        break

    x, y = get_batch("train", train_rng)

    optimizer.zero_grad(set_to_none=True)
    loss = batch_loss(model, x, y)
    loss.backward()
    optimizer.step()


# ---------- 4. 生成与保存 ----------
model.eval()

with torch.no_grad():
    # 从一个换行字符开始生成。
    tokens = torch.tensor(
        [encode("\n")],
        dtype=torch.long,
        device=device,
    )

    for _ in range(300):
        # 本模型只需要最后一个字符。
        logits = model(tokens[:, -1:])[:, -1, :]
        probabilities = logits.softmax(dim=-1)

        next_id = torch.multinomial(
            probabilities, num_samples=1
        )
        tokens = torch.cat((tokens, next_id), dim=1)

sample = decode(tokens[0].cpu().tolist())
print("生成样例：", repr(sample))

out = root / "outputs" / "bigram"
out.mkdir(parents=True, exist_ok=True)

torch.save({
    "model_state": model.state_dict(),
    "vocab_size": V,
    "encoding": "ASCII",
    "steps": max_steps,
}, out / "model.pt")

(out / "history.json").write_text(
    json.dumps(history, indent=2),
    encoding="utf-8",
)
(out / "sample.txt").write_text(sample, encoding="utf-8")

print("结果保存到：", out)