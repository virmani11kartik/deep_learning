#!/usr/bin/env python3
"""
Corrected grad + 5-step signed-gradient attack analysis for CIFAR-10.
Saves:
 - ./adv_analysis/grad_visuals_correct.png
 - ./adv_analysis/grad_visuals_incorrect.png (if any)
 - ./adv_analysis/attack_loss_curve.png
 - ./adv_analysis/attack_image_progression_example0.png
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "./adv_analysis"
os.makedirs(save_dir, exist_ok=True)

# Paths / model checkpoint
checkpoint_path = "checkpoints/allcnn_t_best.pth"  # adjust if needed

# CIFAR normalization (must match training)
mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
std  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

# Attack settings
eps_pixels = 8.0              # total l_inf budget in pixel units (0..255)
num_steps = 5
alpha_pixels = eps_pixels / num_steps

# Convert pixel-space eps/alpha to normalized-space (per-channel)
eps_norm_np = (eps_pixels / 255.0) / std    # numpy array shape (3,)
alpha_norm_np = (alpha_pixels / 255.0) / std

# Convert to torch tensors and move to device when needed
eps_norm_tensor_cpu = torch.tensor(eps_norm_np, dtype=torch.float32).view(1,3,1,1)   # CPU tensor
alpha_norm_tensor_cpu = torch.tensor(alpha_norm_np, dtype=torch.float32).view(1,3,1,1)

# ---------------------
# Load model: replace this with your model class import if needed
# Example expects you have `allcnn_t` in the python path (like previous script).
try:
    from allcnn import allcnn_t
except Exception as e:
    print("Could not import allcnn_t from train_allcnn_t_cifar10.py. Make sure that file is present.")
    raise

net = allcnn_t(c1=96, c2=192).to(device)
if os.path.isfile(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(ckpt["model"])
    print("Loaded checkpoint:", checkpoint_path)
else:
    print("Warning: checkpoint not found at", checkpoint_path, "- running with random weights.")

net.eval()

# ---------------------
# Data loader: get a single minibatch of b=100
batch_size = 100
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())
])
test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
indices = list(range(len(test_ds)))[:batch_size]
subset = Subset(test_ds, indices)
loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# helper: unnormalize (tensor in normalized space -> uint8 HxWxC)
def unnormalize_to_uint8(tensor_norm):
    # tensor_norm: torch tensor CxHxW (normalized), values roughly in normalized range
    arr = tensor_norm.detach().cpu().numpy().transpose(1,2,0)
    arr = (arr * std.reshape(1,1,3)) + mean.reshape(1,1,3)   # now in [0,1] approx
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return arr

# gradient visualization helper
def visualize_grad_image(grad_tensor):
    g = grad_tensor.cpu().numpy()  # (C,H,W)
    # simple per-channel absolute normalized maps
    out = np.zeros((g.shape[1], g.shape[2], 3), dtype=np.float32)
    for c in range(3):
        arr = np.abs(g[c])
        mx = arr.max() if arr.max() > 0 else 1.0
        out[..., c] = arr / mx
    out = (out * 255).astype(np.uint8)
    return out

# ---------------------
# Get a batch
(xs, ys) = next(iter(loader))
xs = xs.to(device)
ys = ys.to(device)
orig_xs = xs.detach().clone()   # normalized-space originals

# Categorize predictions (to separate correct / incorrect)
with torch.no_grad():
    logits = net(xs)
preds = logits.argmax(dim=1)
correct_mask = (preds == ys)
incorrect_mask = ~correct_mask
print(f"Batch: {xs.size(0)} images -> correct {correct_mask.sum().item()}, incorrect {incorrect_mask.sum().item()}")

# ---------------------
# Compute input gradients dx for each sample (store CPU tensors)
dx_list = []
for i in range(xs.size(0)):
    x_i = xs[i:i+1].detach().clone().to(device)
    x_i.requires_grad_(True)
    y_i = ys[i:i+1]
    logits_i = net(x_i)
    loss_i = F.cross_entropy(logits_i, y_i)
    # backward
    loss_i.backward()
    dx = x_i.grad.detach().clone().squeeze(0).cpu()  # shape (C,H,W) on CPU
    dx_list.append(dx)
    # free gradient
    x_i.grad.zero_()

# Visualize some correct examples and their grads
n_show = 6
correct_idxs = [i for i in range(len(dx_list)) if correct_mask[i].item()][:n_show]
incorrect_idxs = [i for i in range(len(dx_list)) if incorrect_mask[i].item()][:n_show]

# ensure lists have something
if len(correct_idxs) == 0:
    correct_idxs = list(range(min(n_show, len(dx_list))))
if len(incorrect_idxs) == 0:
    incorrect_idxs = []

# Plot correct examples
plt.figure(figsize=(3*len(correct_idxs), 6))
for j, idx in enumerate(correct_idxs):
    ax = plt.subplot(2, len(correct_idxs), j+1)
    ax.imshow(unnormalize_to_uint8(orig_xs[idx]))
    ax.set_title(f"idx {idx}\nlabel {int(ys[idx])} pred {int(preds[idx])}\n(correct)")
    ax.axis("off")
for j, idx in enumerate(correct_idxs):
    ax = plt.subplot(2, len(correct_idxs), len(correct_idxs)+j+1)
    ax.imshow(visualize_grad_image(dx_list[idx]))
    ax.set_title("abs-per-channel grad")
    ax.axis("off")
plt.suptitle("Correctly classified examples (top) and gradients (bottom)")
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(os.path.join(save_dir, "grad_visuals_correct.png"), dpi=150)
plt.close()

# Plot incorrect examples if present
if len(incorrect_idxs) > 0:
    plt.figure(figsize=(3*len(incorrect_idxs), 6))
    for j, idx in enumerate(incorrect_idxs):
        ax = plt.subplot(2, len(incorrect_idxs), j+1)
        ax.imshow(unnormalize_to_uint8(orig_xs[idx]))
        ax.set_title(f"idx {idx}\nlabel {int(ys[idx])} pred {int(preds[idx])}\n(incorrect)")
        ax.axis("off")
    for j, idx in enumerate(incorrect_idxs):
        ax = plt.subplot(2, len(incorrect_idxs), len(incorrect_idxs)+j+1)
        ax.imshow(visualize_grad_image(dx_list[idx]))
        ax.set_title("abs-per-channel grad")
        ax.axis("off")
    plt.suptitle("Misclassified examples (top) and gradients (bottom)")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(save_dir, "grad_visuals_incorrect.png"), dpi=150)
    plt.close()
else:
    print("No misclassified examples in this batch to save gradient visuals for.")

print("Saved gradient visualizations to:", save_dir)

# ---------------------
# 5-step signed-gradient attack (iterative FGSM)
# Work in normalized-space. Use proper torch tensors for alpha/eps
eps_norm = eps_norm_tensor_cpu.to(device)      # shape (1,3,1,1) on device
alpha_norm = alpha_norm_tensor_cpu.to(device)

x_adv = orig_xs.clone().detach().to(device)    # normalized-space adv examples
x_adv.requires_grad_(True)

loss_history = []
for step in range(num_steps):
    net.zero_grad()
    x_adv.requires_grad_(True)
    logits = net(x_adv)
    losses_per_sample = F.cross_entropy(logits, ys, reduction="none")
    loss_mean = losses_per_sample.mean().item()
    loss_history.append(loss_mean)

    # gradient w.r.t x_adv
    grads = torch.autograd.grad(losses_per_sample.sum(), x_adv, retain_graph=False)[0]  # (b,3,H,W)

    # signed-gradient step (broadcast alpha_norm)
    x_adv = x_adv.detach() + alpha_norm * grads.sign().detach()

    # project back to orig +/- eps_norm (elementwise clamp)
    upper = orig_xs.to(device) + eps_norm
    lower = orig_xs.to(device) - eps_norm
    x_adv = torch.max(torch.min(x_adv, upper), lower).detach()
    x_adv.requires_grad_(True)

print("Loss history (per-step mean):", loss_history)

# Plot loss vs step
plt.figure()
plt.plot(np.arange(num_steps), loss_history, marker='o')
plt.xlabel("Attack step (k)")
plt.ylabel("Mean cross-entropy loss (batch avg)")
plt.title(f"5-step signed gradient attack (eps={eps_pixels} px total, alpha={alpha_pixels:.3f} px/step)")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "attack_loss_curve.png"), dpi=150)
plt.close()
print("Saved attack loss curve to", os.path.join(save_dir, "attack_loss_curve.png"))

# Save example progression for the first sample
x0 = orig_xs[0:1].to(device)
x_curr = x0.clone().detach()
images_seq = [unnormalize_to_uint8(x_curr[0].cpu())]
x_tmp = x0.clone().detach()
x_tmp.requires_grad_(True)
for step in range(num_steps):
    net.zero_grad()
    logits = net(x_tmp)
    loss_tmp = F.cross_entropy(logits, ys[0:1])
    loss_tmp.backward()
    grad_tmp = x_tmp.grad.detach()
    x_tmp = x_tmp.detach() + alpha_norm * grad_tmp.sign().detach()
    upper = x0 + eps_norm
    lower = x0 - eps_norm
    x_tmp = torch.max(torch.min(x_tmp, upper), lower).detach()
    x_tmp.requires_grad_(True)
    images_seq.append(unnormalize_to_uint8(x_tmp[0].cpu()))

plt.figure(figsize=(3*len(images_seq), 3))
for i, img in enumerate(images_seq):
    ax = plt.subplot(1, len(images_seq), i+1)
    ax.imshow(img)
    ax.set_title(f"step {i}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "attack_image_progression_example0.png"), dpi=150)
plt.close()
print("Saved perturbation progression:", os.path.join(save_dir, "attack_image_progression_example0.png"))

print("Done.")
