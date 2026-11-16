import torch

a = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])

b = torch.tensor([
    [10.0, 20.0],
    [30.0, 40.0],
])

print("=== 逐元素乘法 ===")
print(a * b)
print("形状：", (a * b).shape)

print("\n=== 矩阵乘法 ===")
print(a @ b)
print("形状：", (a @ b).shape)

# 2 个样本，每个样本有 3 个特征。
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

# 将 3 个输入特征转换为 2 个输出特征。
w = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

y = x @ w

print("\n=== 特征变换 ===")
print("输入形状：", x.shape)
print("权重形状：", w.shape)
print("输出形状：", y.shape)
print("输出：")
print(y)