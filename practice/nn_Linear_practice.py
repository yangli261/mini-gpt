import torch
from torch import nn

device = "cuda"
x = torch.tensor([[-2.0],[-1.0],[0.0],[1.0],[2.0]],device = device)
y = 5.0*x - 1.0

model = nn.Linear(1,1).to(device)
optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)

step_num = 60
for step in range(1,step_num + 1):
    optimizer.zero_grad(set_to_none = True)
    loss = (model(x) - y).square().mean()
    loss.backward()
    if step % 5 == 0:
        print(f"第{step}步更新之前的损失为：{loss:3f}")
    optimizer.step()

model.eval()
with torch.no_grad():
    test_x = torch.tensor([[3.0],[4.0]],device = device)
    prediction = model(test_x)

expect_y = torch.tensor([[14.0],[19.0]],device = device)
assert prediction.shape == (2,1),print("预测形状错误")
assert torch.allclose(prediction,expect_y),print(" 预测结果偏差较大")
print(f"预测形状为:{prediction.shape}")
print(f"预测结果为{prediction.cpu().tolist()}")
