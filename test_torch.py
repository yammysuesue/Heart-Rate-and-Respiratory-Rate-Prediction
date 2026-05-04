# test torch2 
import torch
print(torch.__version__)
print(torch.version.cuda)
# test torch2.0 is available
print(torch.cuda.is_available())

# test torch2.0 is available
print(torch.cuda.get_device_name(0))

# test torch2.0 is available

A = torch.randn(10, 10).cuda()
B = torch.randn(10, 10).cuda()
C = A @ B
print(C)