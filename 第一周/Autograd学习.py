# 基础张量梯度
import torch

# 创建张量并要求梯度
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(x)
y = x[0] ** 2 + 3 * x[1]  # 2^2 + 3*3 = 13
print(y)
# 反向传播   x^2的导数为2*x , 3*x的导数为3
y.backward(retain_graph=True)

print(x.grad)  # 输出梯度 [dy/dx0, dy/dx1] => [4.0, 3.0]  dx0是对x[0作为变量的导数, dx1是对x[1]作为变量的导数
# 注意：如果不指定retain_graph=True，计算图在反向传播后会被释放
# 如果需要多次调用backward()，需要保留计算图
y = x[0] ** 3 + 3 * x[1]  # 2^3 + 3*3 = 17
x.grad.zero_()
y.backward()
print(x.grad)