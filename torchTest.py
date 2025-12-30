import torch

print(torch.cuda.is_available())     # True
print(torch.version.hip)             # e.g. '6.1.0'
print(torch.version.cuda)            # None
print(torch.cuda.get_device_name(0)) # AMD Radeon RX 7900 XTX