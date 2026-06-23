import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

x = torch.rand(5000, 5000).cuda()
y = torch.mm(x, x)

print("Tensor device:", y.device)
print("VRAM allocated (GB):", torch.cuda.memory_allocated()/1e9)