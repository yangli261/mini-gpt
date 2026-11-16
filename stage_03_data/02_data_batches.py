from pathlib import Path
import torch

# 根据脚本位置定位数据，不依赖终端当前所在目录。
project_root = Path(__file__).resolve().parent.parent
text = (
    project_root / "data" / "tinyshakespeare.txt"
).read_text(encoding="utf-8")

# 本课使用预先定义的 ASCII 字符词表。
if not text.isascii():
    raise ValueError("本课使用 ASCII 语料，请检查下载文件。")

chars = [chr(i) for i in range(128)]
stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}
vocab_size = len(chars)


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# 先按连续区间划分，再分别采样。
n = len(data)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

splits = {
    "train": data[:train_end],
    "val": data[train_end:val_end],
    "test": data[val_end:],
}

batch_size = 4
block_size = 8
device = torch.device("cuda")

# 固定随机种子，便于复现本课采样。
generator = torch.Generator().manual_seed(42)

for name, part in splits.items():
    if len(part) <= block_size:
        raise ValueError(f"{name} 数据太短")
    print(name, "字符数：", len(part))

print("固定词表大小：", vocab_size)
print("语料实际字符种类：", len(set(text)))


def get_batch(split):
    source = splits[split]

    starts = torch.randint(
        low=0,
        high=len(source) - block_size,
        size=(batch_size,),
        generator=generator,
    )

    x = torch.stack([
        source[i:i + block_size]
        for i in starts.tolist()
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in starts.tolist()
    ])

    # 检查输入与目标确实错开一位。
    assert torch.equal(x[:, 1:], y[:, :-1])

    return x.to(device), y.to(device)


# 测试集留到最终评测，本课不采样它。
for split in ("train", "val"):
    x, y = get_batch(split)

    print(f"\n{split} 输入形状：", x.shape)
    print("目标形状：", y.shape)
    print("设备：", x.device)
    print("首条输入：", repr(decode(x[0].cpu().tolist())))
    print("首条目标：", repr(decode(y[0].cpu().tolist())))