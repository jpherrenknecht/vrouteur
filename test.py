print ('test')
import torch

print("PyTorch version :", torch.__version__)
print("CUDA available ? :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU utilisé :", torch.cuda.get_device_name(0))
    print("CUDA version :", torch.version.cuda)
    print("Arch list :", torch.cuda.get_arch_list())

    # petit test GPU
    a = torch.rand(2,2, device="cuda")
    b = torch.rand(2,2, device="cuda")
    print("Test addition GPU :\n", a + b)