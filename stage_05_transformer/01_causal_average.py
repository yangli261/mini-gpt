import torch

torch.set_printoptions(precision=3, sci_mode=False)

# 手工指定特征，让计算结果容易检查。
# 本课先不训练 Embedding。
h = torch.tensor([[
    [1.0, 0.0],
    [0.0, 2.0],
    [3.0, 0.0],
    [0.0, 4.0],
]])  # [B=1, T=4, C=2]

B, T, C = h.shape

# 下三角：只允许当前位置及更早位置。
mask = torch.tril(torch.ones(T, T))

# 每行除以允许的位置数量。
weights = mask / mask.sum(dim=-1, keepdim=True)
context = weights @ h

print("因果平均权重：\n", weights)
print("输入形状：", h.shape)
print("上下文形状：", context.shape)
print("各位置上下文：\n", context[0])

# 用累积和交叉检查同一个公式。
lengths = torch.arange(
    1, T + 1, dtype=h.dtype
).view(1, T, 1)

by_cumsum = h.cumsum(dim=1) / lengths
assert torch.allclose(context, by_cumsum)

# 改变最后位置，检查前面是否受到影响。
changed = h.clone()
changed[:, -1, :] = torch.tensor([100.0, 100.0])
changed_context = weights @ changed

print(
    "改未来后，前3个位置不变：",
    torch.allclose(
        context[:, :3, :],
        changed_context[:, :3, :],
    ),
)
print("改未来后的最后位置：", changed_context[0, -1])

# 交换前两个位置，检查第三位置的平均值。
reordered = h[:, [1, 0, 2, 3], :]
reordered_context = weights @ reordered

print(
    "前文换序后，第三位置均值相同：",
    torch.allclose(
        context[:, 2, :],
        reordered_context[:, 2, :],
    ),
)