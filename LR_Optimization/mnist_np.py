import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision.datasets.utils import download_url
import matplotlib.pyplot as plt

DATA_DIR = "./data"
DOWNSAMPLE_TO_14 = True 
IMG_SIZE = 14 if DOWNSAMPLE_TO_14 else 28
LR = 0.1 
EPOCHS = 15
BATCH_SIZE = 256
L2 = 1e-4
SEED = 42
MAX_ITERS = 400
INIT_FILE = "init_weights_logreg.npy"
LAMBDA = 1e-3

np.random.seed(SEED)

# -----------------------------
# Load MNIST (download only)
# -----------------------------
train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True)
val_set   = datasets.MNIST(root=DATA_DIR, train=False, download=True)


def to_numpy_mnist_subset(mnist_dataset, classes=(0, 1), img_size=14):
    xs = []
    ys = []
    for img, label in mnist_dataset:
        if label in classes:
            if img_size != 28:
                img = img.resize((img_size, img_size), resample=Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)  # shape (H,W)
            arr = arr / 255.0                        # scale to [0,1]
            xs.append(arr.reshape(-1))               # flatten
            ys.append(1 if label == classes[1] else 0)
    X = np.stack(xs, axis=0)         # (N, D)
    y = np.array(ys, dtype=np.int64) # (N,)
    return X, y

def standardize_train_apply_to_val(Xtr, Xval):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-8
    return (Xtr - mu) / sd, (Xval - mu) / sd

def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])



# Filter to classes {0,1} and convert to numpy
Xtr, ytr = to_numpy_mnist_subset(train_set, classes=(0,1), img_size=IMG_SIZE)
Xva, yva = to_numpy_mnist_subset(val_set, classes=(0,1), img_size=IMG_SIZE)

# Standardize features (using train stats), then add bias
Xtr, Xva = standardize_train_apply_to_val(Xtr, Xva)
Xtr = add_bias(Xtr)
Xva = add_bias(Xva)


N, D = Xtr.shape
print(f"Train: {Xtr.shape}, Val: {Xva.shape}")

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def loss_and_grad(w, w0, X, y, lam):
    z = X.dot(w) + w0                     
    yz = y * z
    sig = 1.0 / (1.0 + np.exp(yz))
    loss = np.log1p(np.exp(-yz)).mean() + 0.5*lam*(np.dot(w,w) + w0*w0)
    g_common = - (y * sig)               
    grad_w  = (X.T @ g_common)/X.shape[0] + lam*w
    grad_w0 = g_common.mean() + lam*w0
    return loss, grad_w, grad_w0

def accuracy(X, y, w, w0):
    p = 1.0 / (1.0 + np.exp(-(X.dot(w)+w0)))          
    yhat = np.where(p>=0.5, +1, -1)
    return (yhat==y).mean()

def main():
    if os.path.exists(INIT_FILE):
        init = np.load(INIT_FILE, allow_pickle=True).item()
        w = init["w"].copy()
        w0 = float(init["w0"])
    else:
        w = np.random.randn(D).astype(np.float32) * 0.01
        w0 = float(np.random.randn() * 0.01)
        np.save(INIT_FILE, {"w": w, "w0": w0})

    losses = []
    for t in range(1, MAX_ITERS+1):
        L, gw, gw0 = loss_and_grad(w, w0, Xtr, ytr, LAMBDA)
        w  -= LR * gw
        w0 -= LR * gw0
        losses.append(L)

    tr_acc = accuracy(Xtr, ytr, w, w0)
    va_acc = accuracy(Xva, yva, w, w0)
    print(f"Train acc: {tr_acc*100:.2f}% | Val acc: {va_acc*100:.2f}% | Final loss: {losses[-1]:.4f}")

    eps = 1e-12
    f_star_hat = np.min(losses)
    y_log = np.log(np.maximum(np.array(losses) - f_star_hat + eps, eps))

    burn, end = int(0.1*MAX_ITERS), MAX_ITERS
    xs = np.arange(burn, end)
    coef = np.polyfit(xs, y_log[burn:end], deg=1) 
    slope = coef[0]
    kappa_hat = -1.0 / slope if slope < 0 else np.inf
    print(f"Slope (semi-log): {slope:.4f}  ==>  κ_hat ≈ {-1.0/slope:.2f}")

    plt.figure(figsize=(6,4))
    plt.semilogy(np.arange(1, MAX_ITERS+1), np.array(losses)-f_star_hat+eps, label="train loss - min(loss)")
    plt.xlabel("parameter updates (t)")
    plt.ylabel("loss gap (log scale)")
    plt.title(f"Full-batch GD, λ={LAMBDA}, slope≈{slope:.4f}, κ̂≈{kappa_hat:.1f}")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig("gd_semilog.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()