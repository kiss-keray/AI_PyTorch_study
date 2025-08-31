import torch
import torch.nn as nn

# 定义一个简单的全连接网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()   # 必须调用父类构造函数
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleNN(10, 100, 5)

# 打印模型结构
print(model)
t = torch.randn(5, 10)
print(t)
o = model(t)
print(o)
print(o.shape)
for name, param in model.named_parameters():
    print(name, param.shape)