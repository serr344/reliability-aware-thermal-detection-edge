"""Check the setup: can PyTorch see the GPU, and is Ultralytics installed?"""
import torch
import ultralytics

print("PyTorch     :", torch.__version__)
print("CUDA (torch):", torch.version.cuda)
print("GPU found   :", torch.cuda.is_available())

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print("GPU         :", gpu.name)
    print("VRAM        :", round(gpu.total_memory / 1024**3, 1), "GB")

print("Ultralytics :", ultralytics.__version__)
