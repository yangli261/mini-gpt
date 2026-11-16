import torch
from torch import nn

# 待训练参数，nn.Parameter 默认 requires_grad=True。
w = nn.Parameter(torch.tensor(0.0, device="cuda"))
b = nn.Parameter(torch.tensor(0.0, device="cuda"))

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="cuda")
target = 3 * x + 2

# 显式告诉优化器：更新 w 和 b。
optimizer = torch.optim.SGD([w, b], lr=0.1)

print("初始参数：", w.item(), b.item())

for step in range(101):
    # 1. 清空上一轮梯度。
    optimizer.zero_grad(set_to_none=True)

    # 2. 前向计算，建立新计算图。
    prediction = w * x + b
    loss = (prediction - target).square().mean()

    # 3. 反向传播，将梯度写入参数的 .grad。
    loss.backward()

    # 打印的是本次参数更新前的状态。
    if step % 20 == 0:
        print(
            f"step={step:3d} | loss={loss.item():.8f} | "
            f"w={w.item():.6f} | b={b.item():.6f}"
        )

    # 4. 使用梯度更新参数。
    optimizer.step()

# 训练结束，检查效果。
with torch.no_grad():
    prediction = w * x + b
    final_loss = (prediction - target).square().mean()

    new_x = torch.tensor([3.0, 4.0], device="cuda")
    new_prediction = w * new_x + b

print("\n训练后的参数：", w.item(), b.item())
print("训练数据预测：", prediction.cpu().tolist())
print("最终损失：", final_loss.item())
print("新输入的预测：", new_prediction.cpu().tolist())