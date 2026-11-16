import torch

# 一维张量：一条包含 4 个字符编号的序列。
sequence = torch.tensor([10, 20, 30, 40])

print("=== 一条序列 ===")
print(sequence)
print("形状：", sequence.shape)
print("维度数量：", sequence.ndim)
print("数据类型：", sequence.dtype)
print("所在设备：", sequence.device)
print("元素总数：", sequence.numel())

# 二维张量：两条序列组成一个批次。
batch = torch.tensor([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
])

print("\n=== 一个批次 ===")
print(batch)
print("形状：", batch.shape)
print("维度数量：", batch.ndim)
print("元素总数：", batch.numel())

# 索引从 0 开始。
print("\n=== 获取数据 ===")
print("第一条序列：", batch[0])
print("第一条序列的形状：", batch[0].shape)
print("第二条序列的第三个编号：", batch[1, 2].item())

# 添加一个长度为 1 的维度。
print("\n=== 添加维度 ===")
print("序列添加批次维度：", sequence.unsqueeze(0).shape)
print("批次添加末尾维度：", batch.unsqueeze(-1).shape)

# 把数据复制到 GPU，形状不变。
batch_gpu = batch.to("cuda")

print("\n=== 移动到 GPU ===")
print("所在设备：", batch_gpu.device)
print("形状：", batch_gpu.shape)

print("\n=== 整数索引与切片 ===")

print("整数索引：", batch[1])
print("形状：", batch[1].shape)

print("切片：", batch[1:2])
print("形状：", batch[1:2].shape)

print("取出后添加维度：", batch[1].unsqueeze(0))
print("形状：", batch[1].unsqueeze(0).shape)