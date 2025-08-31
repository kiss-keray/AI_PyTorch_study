import torch

# 创建张量
x = torch.randn(3, 3, device=torch.device("mps"))
y = torch.randn(3, 3, device=torch.device("mps"))
print(x)
print(y)
# 张量求和
z = x + y
print(z)
# 张量乘法
z = x * y
print(z)
# 张量减法
z = x - y
print(z)
# 张量除法
z = x / y
print(z)


