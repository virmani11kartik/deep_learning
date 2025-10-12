import torch
import torch.nn as nn
import torchvision.models as models

# Build the model (no pretrained weights for simplicity)
model = models.resnet18(weights=None)

# --- Buckets ---
bn_affine_params = []      # (i) BatchNorm gamma/beta
bias_params = []           # (ii) biases from Conv/Linear
rest_params = []           # (iii) everything else

# We'll also track names to avoid double-counting
picked = set()
named_params = dict(model.named_parameters())

# (i) BatchNorm affine params (gamma=weight, beta=bias)
for mod_name, mod in model.named_modules():
    if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if mod.weight is not None:
            name = f"{mod_name}.weight" if mod_name else "weight"
            if name in named_params:
                bn_affine_params.append(named_params[name])
                picked.add(name)
        if mod.bias is not None:
            name = f"{mod_name}.bias" if mod_name else "bias"
            if name in named_params:
                bn_affine_params.append(named_params[name])
                picked.add(name)

# (ii) Biases from Conv/Linear
for mod_name, mod in model.named_modules():
    if isinstance(mod, (nn.Conv2d, nn.Linear)) and (mod.bias is not None):
        name = f"{mod_name}.bias" if mod_name else "bias"
        if name in named_params and name not in picked:
            bias_params.append(named_params[name])
            picked.add(name)

# (iii) Everything else (e.g., Conv/Linear weights, etc.)
for name, p in model.named_parameters():
    if name not in picked:
        rest_params.append(p)

# --- Sanity prints ---
def count_params(param_list):
    return sum(p.numel() for p in param_list)

print("Parameter split for ResNet-18")
print(f"(i)  BN affine (gamma/beta): {count_params(bn_affine_params):>10,} params, {len(bn_affine_params)} tensors")
print(f"(ii) Conv/FC biases:         {count_params(bias_params):>10,} params, {len(bias_params)} tensors")
print(f"(iii) Rest (weights, etc.):  {count_params(rest_params):>10,} params, {len(rest_params)} tensors")
print(f"TOTAL:                       {count_params(bn_affine_params)+count_params(bias_params)+count_params(rest_params):>10,}")

# --- Example: build optimizer with different weight_decay per group ---
optimizer = torch.optim.AdamW([
    {"params": rest_params,       "weight_decay": 1e-4},  # decay on weights
    {"params": bn_affine_params,  "weight_decay": 0.0},   # no decay on BN gamma/beta
    {"params": bias_params,       "weight_decay": 0.0},   # no decay on biases
], lr=3e-4)
