import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # PIL.Image 或 ndarray -> Tensor
    transforms.Normalize((0.5,), (0.5,))
])

# 加载训练集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# DataLoader: 批量 + 打乱
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 使用示例
for images, labels in train_loader:
    print(images.shape, labels.shape)
    break
