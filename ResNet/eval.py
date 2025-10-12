#!/usr/bin/env python3
"""
Evaluate clean accuracy and 1-step FGSM (l_inf) accuracy on CIFAR-10 validation set.

Usage:
    python eval_fgsm.py --checkpoint checkpoints/allcnn_t_best.pth --batch-size 256
"""
import os, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------
# Parse args
# -------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--data", type=str, default="./data")
ap.add_argument("--checkpoint", type=str, required=True)
ap.add_argument("--batch-size", type=int, default=256)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--eps-pixels", type=float, default=8.0, help="epsilon in pixel units (0..255)")
ap.add_argument("--device", type=str, default=None)
args = ap.parse_args()

device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
print("Using device:", device)

# -------------------------
# CIFAR-10 normalization (must match training)
# -------------------------
mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
std  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

# convert eps from pixel units to normalized-space per-channel
eps_pixels = float(args.eps_pixels)
eps_norm = (eps_pixels / 255.0) / std   # numpy (3,)
eps_norm_t = torch.tensor(eps_norm, dtype=torch.float32).view(1,3,1,1).to(device)

# Bounds in normalized-space corresponding to valid pixel range [0,1]
min_norm = ((0.0 - mean) / std).astype(np.float32)
max_norm = ((1.0 - mean) / std).astype(np.float32)
min_norm_t = torch.tensor(min_norm, dtype=torch.float32).view(1,3,1,1).to(device)
max_norm_t = torch.tensor(max_norm, dtype=torch.float32).view(1,3,1,1).to(device)

# -------------------------
# Load model architecture & checkpoint
# -------------------------
# Replace import with your model class if different
try:
    from allcnn import allcnn_t
except Exception:
    raise RuntimeError("Could not import allcnn_t from train_allcnn_t_cifar10.py. "
                       "Make sure that file is in the same folder or adjust the import.")

net = allcnn_t(c1=96, c2=192).to(device)
ckpt = torch.load(args.checkpoint, map_location=device)
if "model" in ckpt:
    net.load_state_dict(ckpt["model"])
else:
    net.load_state_dict(ckpt)
net.eval()
print("Loaded checkpoint:", args.checkpoint)

# -------------------------
# Data loader: validation set (no augmentation)
# -------------------------
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist()),
])
val_ds = datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_tf)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

# -------------------------
# Helpers
# -------------------------
@torch.no_grad()
def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).sum().item()

# -------------------------
# Main evaluation loop
# -------------------------
total = 0
clean_correct = 0
adv_correct = 0
t0 = time.time()

for batch_idx, (x, y) in enumerate(val_loader):
    x = x.to(device)
    y = y.to(device)
    b = x.size(0)
    total += b

    # --- clean predictions ---
    with torch.no_grad():
        logits = net(x)
        clean_correct += accuracy_from_logits(logits, y)

    # --- FGSM (1-step) attack: compute gradient w.r.t input and take ±eps step ---
    # Need grad on inputs
    x_adv = x.detach().clone()
    x_adv.requires_grad_(True)

    logits_adv = net(x_adv)
    loss = F.cross_entropy(logits_adv, y)
    # compute gradients of loss wrt input
    loss.backward()
    grad = x_adv.grad.detach()  # shape (b,3,H,W)

    # signed gradient step in normalized space
    # broadcast per-channel eps_norm_t
    step = eps_norm_t * torch.sign(grad)
    x_adv = x_adv.detach() + step.detach()

    # clip to be valid image (in normalized space)
    x_adv = torch.max(torch.min(x_adv, max_norm_t), min_norm_t)

    # (optional) also ensure we didn't move beyond orig +/- eps_norm per channel:
    # lower = x - eps_norm_t; upper = x + eps_norm_t
    # x_adv = torch.max(torch.min(x_adv, upper), lower)

    # Evaluate model on adversarial examples
    with torch.no_grad():
        logits_adv2 = net(x_adv)
        adv_correct += accuracy_from_logits(logits_adv2, y)

    # free gradients
    net.zero_grad()
    x_adv.grad = None

# Final numbers
clean_acc = 100.0 * clean_correct / total
adv_acc   = 100.0 * adv_correct   / total
dt = time.time() - t0

print(f"Validation size: {total}")
print(f"Clean accuracy: {clean_acc:.3f}%")
print(f"1-step FGSM (eps={eps_pixels}px) accuracy: {adv_acc:.3f}%")
print(f"Elapsed: {dt:.1f}s")

