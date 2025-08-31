import torch
# 直接从列表创建
a = torch.tensor([[1, 2, 3]]) #指定一维数组
print(a)
# 全零、全一、随机
zeros = torch.zeros(3, 4)    # 3x4全零
print(zeros)
ones = torch.ones(2, 5)      # 2x5全一
print(ones)
rand = torch.rand(2, 3)      # [0,1)均匀分布随机数
print(rand)
randn = torch.randn(2, 3)    # 标准正态分布随机数
print(randn)

# 其他
eye = torch.eye(3)            # 单位矩阵
print(eye)
arange = torch.arange(0, 10, 2)  # 类似 range
print(arange)
linspace = torch.linspace(0, 1, steps=10)  # 等间隔分布
print(linspace)
