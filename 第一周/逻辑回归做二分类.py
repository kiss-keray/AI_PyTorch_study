import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 生成二维点数据
torch.manual_seed(0)
num_points = 200
# 类别0：均值在(-2,-2)
class0 = torch.randn(num_points, 2) + torch.tensor([-2.0, -2.0])
labels0 = torch.zeros(num_points, 1)
# 类别1：均值在(2,2)
class1 = torch.randn(num_points, 2) + torch.tensor([2.0, 2.0])
labels1 = torch.ones(num_points, 1)

X = torch.cat([class0, class1], dim=0)
y = torch.cat([labels0, labels1], dim=0)

# 2. 定义简单的神经网络
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. 训练
for epoch in range(5000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# 4. 绘制分类边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 200), torch.linspace(y_min, y_max, 200), indexing="ij")
grid = torch.cat([xx.reshape(-1,1), yy.reshape(-1,1)], dim=1)
with torch.no_grad():
    Z = model(grid).reshape(xx.shape)

plt.contourf(xx, yy, Z.numpy(), levels=[0,0.5,1], alpha=0.3, cmap="RdBu")
plt.scatter(class0[:,0], class0[:,1], c='blue', label='Class 0')
plt.scatter(class1[:,0], class1[:,1], c='red', label='Class 1')
plt.legend()
plt.title("2D Point Classification with Neural Network")
plt.show()
