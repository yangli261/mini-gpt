import math
import torch

torch.set_printoptions(precision=3, sci_mode=False)

# 三个位置，每个向量有两个分量。
q = torch.tensor([[
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
]])

k = torch.tensor([[
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
]])

v = torch.tensor([[
    [10.0, 0.0],
    [0.0, 20.0],
    [30.0, 0.0],
]])

B, T, d_k = q.shape
allowed = torch.tril(torch.ones(T, T, dtype=torch.bool))


def attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    masked = scores.masked_fill(~allowed, float("-inf"))
    weights = masked.softmax(dim=-1)
    context = weights @ v
    return scores, weights, context


scores, weights, context = attention(q, k, v)

print("匹配分数：\n", scores[0])
print("注意力权重：\n", weights[0])
print("每行权重和：", weights.sum(dim=-1))
print("上下文结果：\n", context[0])
print("形状：", weights.shape, context.shape)

# 只改变最后位置的query。
changed_q = q.clone()
changed_q[:, -1, :] = torch.tensor([0.0, 1.0])

_, new_weights, new_context = attention(changed_q, k, v)

print("改query后的最后一行权重：", new_weights[0, -1])
print("改query后的最后位置结果：", new_context[0, -1])

# 改变未来位置的key/value。
future_k = k.clone()
future_v = v.clone()
future_k[:, -1, :] = torch.tensor([100.0, 100.0])
future_v[:, -1, :] = torch.tensor([100.0, 100.0])

_, _, future_context = attention(q, future_k, future_v)

print(
    "改未来后前两个位置不变：",
    torch.allclose(
        context[:, :2, :],
        future_context[:, :2, :],
    ),
)