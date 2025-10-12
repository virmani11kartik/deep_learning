#!/usr/bin/env python3
import os, time, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import trange, tqdm

# -------------------------
# Model (your architecture)
# -------------------------
class View(nn.Module):
    def __init__(self, o):
        super().__init__()
        self.o = o
    def forward(self, x):
        return x.view(-1, self.o)

class allcnn_t(nn.Module):
    def __init__(self, c1=96, c2=192):
        super().__init__()
        d = 0.5
        def convbn(ci, co, ksz, s=1, pz=0):
            return nn.Sequential(
                nn.Conv2d(ci, co, ksz, stride=s, padding=pz, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(co),
            )
        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3,   c1, 3, 1, 1),
            convbn(c1,  c1, 3, 1, 1),
            convbn(c1,  c1, 3, 2, 1),     # downsample (16x16)
            nn.Dropout(d),
            convbn(c1,  c2, 3, 1, 1),
            convbn(c2,  c2, 3, 1, 1),
            convbn(c2,  c2, 3, 2, 1),     # downsample (8x8)
            nn.Dropout(d),
            convbn(c2,  c2, 3, 1, 1),
            convbn(c2,  c2, 3, 1, 1),
            convbn(c2, 10, 1, 1, 0),      # 1x1 conv to 10 classes
            nn.AvgPool2d(8),              # global avg pool (to 1x1)
            View(10),
        )
        print('Num parameters:', sum(p.numel() for p in self.m.parameters()))
    def forward(self, x):
        return self.m(x)

# -------------------------
# Utilities
# -------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True

@torch.no_grad()
def accuracy(output, target):
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item() * 100.0

def get_param_groups(model):
    """
    Apply weight decay only to true weights. No decay on biases or BN affine params.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if name.endswith(".bias") or "bn" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": 1e-3},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def get_lr(epoch):
    # 100 epochs: 0.1 for first 40, 0.01 next 40, 0.001 last 20
    if epoch < 40: return 0.1
    if epoch < 80: return 0.01
    return 0.001

def make_loaders(data_root, batch_size, workers, val_split=5000):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    full_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    if val_split > 0:
        train_size = len(full_train) - val_split
        train_set, val_idx = random_split(full_train, [train_size, val_split],
                                          generator=torch.Generator().manual_seed(42))
        # Re-wrap the validation subset with clean (test) transform
        val_set = datasets.CIFAR10(root=data_root, train=True, download=False, transform=test_tf)
        val_set = torch.utils.data.Subset(val_set, val_idx.indices)
    else:
        train_set = full_train
        val_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                            num_workers=workers, pin_memory=True)
    return train_loader, val_loader

# -------------------------
# Train / Eval loops
# -------------------------
def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss, total_top1, n = 0.0, 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward(); optimizer.step()
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_top1 += accuracy(logits, targets) * bs
        n += bs
    return total_loss / n, 100.0 - (total_top1 / n)  # loss, error%

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_top1, n = 0.0, 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_top1 += accuracy(logits, targets) * bs
        n += bs
    return total_loss / n, 100.0 - (total_top1 / n)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--c1", type=int, default=96)
    ap.add_argument("--c2", type=int, default=192)
    ap.add_argument("--val-split", type=int, default=5000)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = make_loaders(args.data, args.batch_size, args.workers, args.val_split)

    # Model
    model = allcnn_t(c1=args.c1, c2=args.c2).to(device)

    # Optimizer (SGD + Nesterov), proper WD grouping
    param_groups = get_param_groups(model)
    optimizer = torch.optim.SGD(param_groups, lr=0.1, momentum=0.9, nesterov=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Logs
    tr_losses, tr_errors, va_losses, va_errors = [], [], [], []
    best_val_err = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    # Training
    for epoch in trange(args.epochs, desc="Epochs", leave=True):
        # Piecewise LR
        lr = get_lr(epoch)
        for pg in optimizer.param_groups: pg["lr"] = lr

        t0 = time.time()
        tr_loss, tr_err = train_one_epoch(model, train_loader, optimizer, device, scaler)
        va_loss, va_err = evaluate(model, val_loader, device)
        dt = time.time() - t0

        tr_losses.append(tr_loss); tr_errors.append(tr_err)
        va_losses.append(va_loss); va_errors.append(va_err)

        tqdm.write(f"Epoch {epoch+1:03d}/{args.epochs} | lr={lr:.3g} | "
                   f"train loss={tr_loss:.4f}, err={tr_err:.2f}% | "
                   f"val loss={va_loss:.4f}, err={va_err:.2f}% | {dt:.1f}s")

        if va_err < best_val_err:
            best_val_err = va_err
            torch.save({"model": model.state_dict(),
                        "epoch": epoch+1,
                        "val_err": va_err},
                       "checkpoints/allcnn_t_best.pth")

    # Save last
    torch.save({"model": model.state_dict(),
                "epoch": args.epochs,
                "val_err": va_errors[-1]},
               "checkpoints/allcnn_t_last.pth")

    # Save curves + plot
    with open("training_curves.json", "w") as f:
        json.dump({"train_loss": tr_losses, "train_err": tr_errors,
                   "val_loss": va_losses, "val_err": va_errors}, f)

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.plot(tr_losses, label="train"); plt.plot(va_losses, label="val")
        plt.title("Loss"); plt.xlabel("epoch"); plt.legend()
        plt.subplot(1,2,2); plt.plot(tr_errors, label="train"); plt.plot(va_errors, label="val")
        plt.title("Top-1 Error (%)"); plt.xlabel("epoch"); plt.legend()
        plt.tight_layout(); plt.savefig("training_curves.png", dpi=150)
        tqdm.write("Saved: training_curves.png and checkpoints/*.pth")
    except Exception as e:
        tqdm.write(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
