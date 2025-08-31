import time

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cpu")

# 1. 定义一个只有一层的模型（输入1维 -> 输出1维）
model = nn.Linear(1, 1).to(device)

# 2. 准备数据 (y = 2x + 1)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], device=device)

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
now = int(time.time() * 1000)
# 4. 训练
for epoch in range(100000):
    # 前向传播
    y_pred = model(x)
    # 计算损失
    loss = criterion(y_pred, y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
print("耗时:", int(time.time() * 1000) - now, "ms")
# 5. 训练完成，查看学到的参数
print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())

# 6. 预测
test_x = torch.tensor([[40.0],[50.0]], device=device)
print("Prediction for x=5:", model(test_x))
