import torch
import torchvision.models as models

# Load ResNet-18 model (no pretrained weights for simplicity)
model = models.resnet18(weights=None)

print("Layer-wise parameter count for ResNet-18:\n")
total_params = 0

for name, param in model.named_parameters():
    if param.requires_grad:  # only count trainable params
        num_params = param.numel()
        total_params += num_params
        print(f"{name:<40} {num_params}")

print("\n--------------------------------------------")
print(f"Total Trainable Parameters: {total_params:,}")
