import torch
print(torch.__version__)
print(torch.mps.is_available())
if torch.cuda.is_available():
    print(torch.version.cuda)  # PyTorch 编译的 CUDA 版本
    print(torch.cuda.get_device_name(0))  # GPU 名称