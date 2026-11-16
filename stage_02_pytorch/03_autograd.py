import torch

# 直接在 GPU 上创建需要求导的参数。
w = torch.tensor(2.0, device="cuda", requires_grad=True)
x = torch.tensor(3.0, device="cuda")
target = torch.tensor(10.0, device="cuda")

# 前向计算创建计算图。
y = w * x
loss = (y - target).square()

print("=== 计算图 ===")
print("w 是叶子张量：", w.is_leaf)
print("y 是叶子张量：", y.is_leaf)
print("y 的梯度函数：", y.grad_fn)
print("首次反向传播前 w.grad：", w.grad)
print("预测值：", y.item())
print("损失：", loss.item())

loss.backward()
print("第一次梯度：", w.grad.item())

# 重新前向计算，创建一张新图。
loss2 = (w * x - target).square()
loss2.backward()
print("累积后的梯度：", w.grad.item())

# 清空已有梯度，再重新计算。
w.grad = None
loss3 = (w * x - target).square()
loss3.backward()
print("清零并重新求导后的梯度：", w.grad.item())

# 参数更新不记录计算图。
learning_rate = 0.01

with torch.no_grad():
    w -= learning_rate * w.grad

w.grad = None

print("\n=== 更新参数 ===")
print("更新后的 w：", w.item())
print("w.requires_grad：", w.requires_grad)
print("w.is_leaf：", w.is_leaf)
print("梯度清零后：", w.grad)

# 只检查更新效果，不准备反向传播。
with torch.no_grad():
    new_y = w * x
    new_loss = (new_y - target).square()

print("更新后的预测值：", new_y.item())
print("更新后的损失：", new_loss.item())
print("检查损失是否记录梯度：", new_loss.requires_grad)