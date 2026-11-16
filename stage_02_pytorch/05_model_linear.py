import torch
from torch import nn

torch.manual_seed(42)
device = torch.device("cuda")


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)


# 先移动模型，再创建优化器。
model = LinearModel().to(device)

# 5 个样本，每个样本 1 个特征。
x = torch.tensor(
    [[-2.0], [-1.0], [0.0], [1.0], [2.0]],
    device=device,
)
target = 3 * x + 2

print("=== 模型与参数 ===")
print(model)

for name, parameter in model.named_parameters():
    print(
        name,
        "shape=", tuple(parameter.shape),
        "device=", parameter.device,
        "requires_grad=", parameter.requires_grad,
    )

print("输入形状：", x.shape)
print("输出形状：", model(x).shape)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

model.train()

for step in range(101):
    optimizer.zero_grad(set_to_none=True)

    prediction = model(x)
    loss = loss_fn(prediction, target)

    loss.backward()

    if step % 20 == 0:
        print(f"step={step:3d} | loss={loss.item():.8f}")

    optimizer.step()

model.eval()

with torch.no_grad():
    new_x = torch.tensor([[3.0], [4.0]], device=device)
    result = model(new_x)
    final_loss = loss_fn(model(x), target)

print("\n训练后的权重：", model.linear.weight.detach().cpu().tolist())
print("训练后的偏置：", model.linear.bias.detach().cpu().tolist())
print("新输入预测：", result.cpu().tolist())
print("最终损失：", final_loss.item())
print("参数保存键：", list(model.state_dict().keys()))