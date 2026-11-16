import sys
import torch

print("Python 路径：", sys.executable)
print("PyTorch 版本：", torch.__version__)
print("PyTorch CUDA 版本：", torch.version.cuda)
print("GPU 是否可用：", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch 未能使用 GPU，请把输出发给我排查。")

print("显卡名称：", torch.cuda.get_device_name(0))

# 在 GPU 上创建张量，并启用梯度计算。
x = torch.tensor([1.0, 2.0, 3.0], device="cuda", requires_grad=True)

# y = x₁² + x₂² + x₃²
y = (x ** 2).sum()

# 自动计算 y 对 x 的导数。
y.backward()

print("张量所在设备：", x.device)
print("计算结果：", y.item())
print("梯度：", x.grad.cpu().tolist())
print("验证通过：GPU 运算和自动求导正常。")