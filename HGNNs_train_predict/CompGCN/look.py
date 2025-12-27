# import torch
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
#     print(f"CUDA compute capability: {torch.cuda.get_device_capability(device)}")
# else:
#     print("CUDA is not available.")
    
# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA version: {torch.version.cuda}")

# import torch_scatter
# print(torch_scatter.__version__)

# import torch
# if torch.cuda.is_available():
#     print(f"CUDA device count: {torch.cuda.device_count()}")
#     for i in range(torch.cuda.device_count()):
#         print(f"Device {i}: {torch.cuda.get_device_name(i)}")
#         print(f"Compute capability: {torch.cuda.get_device_capability(i)}")
# else:
#     print("CUDA is not available.")

import torch
print(torch.__version__)
print(torch.cuda.is_available())