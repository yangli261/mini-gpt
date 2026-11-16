import torch

text = "hello world\nhello pytorch\n"

# 去重并排序，得到字符词表。
chars = sorted(set(text))
vocab_size = len(chars)

# 两个方向的映射。
stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for index, char in enumerate(chars)}


def encode(s):
    return [stoi[char] for char in s]


def decode(ids):
    return "".join(itos[index] for index in ids)


print("字符表：", [repr(char) for char in chars])
print("词表大小：", vocab_size)
print("字符到编号：", stoi)

# 文本转换为整数张量，再解码验证。
ids = encode(text)
data = torch.tensor(ids, dtype=torch.long)
restored = decode(data.tolist())

print("数据形状：", data.shape)
print("数据类型：", data.dtype)
print("前 8 个编号：", data[:8].tolist())
print("解码后的文本：", repr(restored))

assert restored == text

# 从连续文本中构造下一字符预测任务。
block_size = 8
x = data[:block_size]
y = data[1:block_size + 1]

print("\n输入：", repr(decode(x.tolist())))
print("目标：", repr(decode(y.tolist())))
print("输入形状：", x.shape)
print("目标形状：", y.shape)

# 展示每个位置对应的监督目标。
for position in range(block_size):
    context = decode(x[:position + 1].tolist())
    next_char = decode([y[position].item()])
    print(
        f"位置 {position}: "
        f"上下文={context!r} -> 下一字符={next_char!r}"
    )

# 添加批次轴，表示一个样本。
x_batch = x.unsqueeze(0)
y_batch = y.unsqueeze(0)

print("\n单样本批次输入形状：", x_batch.shape)
print("单样本批次目标形状：", y_batch.shape)