import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as thv
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
import matplotlib.pyplot as plt

# --------------------------
# Repro & device
# --------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda")  if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

# --------------------------
# Data: half-per-class splits (same policy as your Part (a))
# --------------------------
def select_half_per_class_torch(X_u8: torch.Tensor, y: torch.Tensor):
    """
    X_u8: (N, 28, 28) uint8 or float; y: (N,) long
    Returns half-per-class selection without replacement, shuffled.
    """
    X = X_u8.float() / 255.0  # normalize here (if not already)
    idx_all = []
    for c in range(10):
        idx_c = (y == c).nonzero(as_tuple=True)[0]
        perm = idx_c[torch.randperm(idx_c.numel())]
        k = perm.numel() // 2
        idx_all.append(perm[:k])
    idx_all = torch.cat(idx_all)
    idx_all = idx_all[torch.randperm(idx_all.numel())]
    return X[idx_all], y[idx_all]

def make_loaders(batch=32):
    train = thv.datasets.MNIST("./", download=True, train=True)
    test  = thv.datasets.MNIST("./", download=True, train=False)

    Xtr, Ytr = select_half_per_class_torch(train.data, train.targets)
    Xva, Yva = select_half_per_class_torch(test.data,  test.targets)

    # Add channel dim (N,1,28,28)
    Xtr = Xtr.unsqueeze(1)
    Xva = Xva.unsqueeze(1)

    ds_tr = TensorDataset(Xtr, Ytr)
    ds_va = TensorDataset(Xva, Yva)

    # Training loader: we want *updates-based* loop (10k updates), so set shuffle=True and drop_last=True
    # Validation loader: iterate once sequentially, no shuffling
    tr_loader = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=True)
    va_loader = DataLoader(ds_va, batch_size=batch, shuffle=False, drop_last=False)
    return tr_loader, va_loader

# --------------------------
# Model: Conv2d embedding -> Flatten -> Linear -> ReLU
# --------------------------
class TorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding: 28x28 -> 7x7 with 8 channels (k=4, s=4)
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=4, bias=True)
        self.fc   = nn.Linear(7*7*8, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (b,1,28,28)
        h = self.conv(x)                 # (b,8,7,7)
        h = torch.flatten(h, 1)          # (b,392)
        logits = self.fc(h)              # (b,10)
        out = self.relu(logits)          # to mirror your (g) design (ReLU before CE)
        return out, logits               # return both (post-ReLU and raw logits)

# --------------------------
# Metrics helpers
# --------------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, tot_err, tot_n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        out, logits = model(xb)
        # Using logits with CE is standard; since we applied ReLU after fc,
        # compare both ways for completeness. We'll use logits here (recommended).
        loss = criterion(logits, yb)

        preds = torch.argmax(out, dim=1)   # consistent with what the network "outputs"
        err = (preds != yb).float().sum().item()

        bs = yb.size(0)
        tot_loss += loss.item() * bs
        tot_err  += err
        tot_n    += bs
    return tot_loss / tot_n, tot_err / tot_n

def error_from_out(out, y):
    preds = torch.argmax(out, dim=1)
    return (preds != y).float().mean().item()

# --------------------------
# Training
# --------------------------
def main():
    batch_size = 32
    iters = 20_0000
    lr = 0.01
    log_every = 100
    val_every = 1_000

    tr_loader, va_loader = make_loaders(batch=batch_size)
    model = TorchNet().to(device)
    criterion = nn.CrossEntropyLoss()      # expects raw logits
    opt = optim.SGD(model.parameters(), lr=lr)

    # We'll advance through the training loader repeatedly until we hit `iters` updates
    tr_iter = iter(tr_loader)

    # Logs
    tr_steps, tr_loss_hist, tr_err_hist = [], [], []
    va_steps, va_loss_hist, va_err_hist = [], [], []

    for t in trange(iters, desc="Updates"):
        try:
            xb, yb = next(tr_iter)
        except StopIteration:
            tr_iter = iter(tr_loader)
            xb, yb = next(tr_iter)

        xb = xb.to(device)
        yb = yb.to(device)

        model.train()
        opt.zero_grad()

        out, logits = model(xb)
        # Use logits for CE (recommended)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()

        if (t + 1) % log_every == 0:
            # Training error computed from post-ReLU output (to mirror your numpy net behavior)
            with torch.no_grad():
                tr_err = error_from_out(out, yb)
            tr_steps.append(t + 1)
            tr_loss_hist.append(loss.item())
            tr_err_hist.append(tr_err)

        if (t + 1) % val_every == 0:
            vloss, verr = evaluate(model, va_loader, criterion)
            va_steps.append(t + 1)
            va_loss_hist.append(vloss)
            va_err_hist.append(verr)

    # ----------------------
    # Plot curves
    # ----------------------
    os.makedirs("outputs_p3i", exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.plot(tr_steps, tr_loss_hist, label="train loss")
    plt.xlabel("updates"); plt.ylabel("loss"); plt.title("Training Loss (PyTorch)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("outputs_p3i/train_loss.png", dpi=160)

    plt.figure(figsize=(6,4))
    plt.plot(tr_steps, tr_err_hist, label="train error")
    plt.xlabel("updates"); plt.ylabel("error"); plt.title("Training Error (PyTorch)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("outputs_p3i/train_error.png", dpi=160)

    plt.figure(figsize=(6,4))
    plt.plot(va_steps, va_loss_hist, marker='o', label="val loss")
    plt.xlabel("updates"); plt.ylabel("loss"); plt.title("Validation Loss (PyTorch)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("outputs_p3i/val_loss.png", dpi=160)

    plt.figure(figsize=(6,4))
    plt.plot(va_steps, va_err_hist, marker='o', label="val error")
    plt.xlabel("updates"); plt.ylabel("error"); plt.title("Validation Error (PyTorch)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("outputs_p3i/val_error.png", dpi=160)

    print("Saved plots to outputs_p3i/")

if __name__ == "__main__":
    main()
