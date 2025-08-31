import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ========= 1. 设备选择 =========
device = torch.device("mps")
print("Using device:", device)

# ========= 2. 数据集 =========
transform = transforms.Compose([
    transforms.ToTensor(),                # 转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# ========= 3. 定义模型 (CNN) =========
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 14x14 -> 14x14
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)   # 10 分类

    def forward(self, x):
        x = self.relu(self.conv1(x))
        # x = self.pool(x)
        x = self.relu(self.conv2(x))
        # x = self.pool(x)    # 14x14 -> 7x7
        x = x.view(x.size(0), -1)  # 展平 (batch, 64*7*7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# ========= 4. 损失函数 & 优化器 =========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
now = int(time.time() * 1000)
# ========= 5. 训练循环 =========
for epoch in range(3):  # 训练 3 个 epoch
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/3], Loss: {total_loss/len(train_loader):.4f}")

print("耗时:", int(time.time() * 1000) - now, "ms")

# ========= 6. 测试 =========
model.eval()
correct = 0
total = 0
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
now = int(time.time() * 1000)
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("耗时:", int(time.time() * 1000) - now, "ms")
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Using device: mps
# Epoch [1/3], Loss: 0.1331
# Epoch [2/3], Loss: 0.0392
# Epoch [3/3], Loss: 0.0195
# 耗时: 37908 ms
# 耗时: 734 ms
# Test Accuracy: 98.44%

# Using device: mps
# Epoch [1/3], Loss: 0.1429
# Epoch [2/3], Loss: 0.0403
# Epoch [3/3], Loss: 0.0244
# 耗时: 31074 ms
# 耗时: 683 ms
# Test Accuracy: 98.69%

# Using device: mps
# Epoch [1/3], Loss: 0.1540
# Epoch [2/3], Loss: 0.0457
# Epoch [3/3], Loss: 0.0320
# 耗时: 26481 ms
# 耗时: 709 ms
# Test Accuracy: 99.00%