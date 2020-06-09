import torch
print(torch.cuda.is_available())

dev_gpu0 = torch.device('cuda:0')

array = torch.zeros(4)
array_0 = array.to(dev_gpu0)

print(torch.cuda.get_device_name(0))