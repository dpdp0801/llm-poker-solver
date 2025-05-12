import torch, platform
print("Torch:", torch.__version__, "MPS available:", torch.backends.mps.is_available())
print("Python:", platform.python_version()) 