import os, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets

# -------------------- config --------------------
DATA_DIR = "./data"
DOWNSAMPLE_TO_14 = True
IMG_SZ = 14 if DOWNSAMPLE_TO_14 else 28
LAM = 1e-3                 # try {1e-4, 1e-3, 1e-2}
SEED = 123
np.random.seed(SEED)

# Training horizon & batch
BATCH_SIZE = 128           # try 64 or 8
UPDATES = 2500             # number of parameter updates for SGD runs
REPORT_EVERY = 10          # compute full loss every k updates (for plotting)
LR_SGD = 0.2               # SGD step size (tune a bit if needed)
BETA_SGD = 0.9             # Nesterov momentum for SGD (0.75–0.95 typical)

# Full-batch Nesterov (AGD) hyperparams
L_smooth = 5.0             # pick/tune; defines kappa=L/m with m=lam
m_strong = LAM
kappa = L_smooth / m_strong
BETA_AGD = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
ETA_AGD  = 1.0 / L_smooth
AGD_ITERS = 400            # to get a clean straight line reference

INIT_FILE = "init_logreg_w_w0.npy"  # shared init for fair comparison


# -------------------- data utils --------------------
def load_mnist_0_1(split="train"):
    ds = datasets.MNIST(root=DATA_DIR, train=(split=="train"), download=True)
    X, y = [], []
    for img, label in ds:
        if label in (0,1):
            if DOWNSAMPLE_TO_14:
                img = img.resize((IMG_SZ, IMG_SZ), resample=Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            X.append(arr.reshape(-1))
            y.append(+1 if label==0 else -1)
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int8)
    return X, y

def standardize_train_apply(Xtr_raw, Xte_raw):
    mu = Xtr_raw.mean(axis=0, keepdims=True)
    sd = Xtr_raw.std(axis=0, keepdims=True) + 1e-8
    return (Xtr_raw - mu)/sd, (Xte_raw - mu)/sd

# -------------------- objective & helpers --------------------
def loss_grad_full(w, w0, X, y, lam):
    z  = X @ w + w0
    yz = y * z
    # stable: log(1 + exp(-yz))
    loss = np.log1p(np.exp(-yz)).mean() + 0.5*lam*(w @ w + w0*w0)
    # sigma(-yz) = 1/(1+exp(yz))
    sig = 1.0 / (1.0 + np.exp(yz))
    g_common = -(y * sig)
    gw  = (X.T @ g_common)/X.shape[0] + lam*w
    gw0 = g_common.mean() + lam*w0
    return loss, gw, gw0

def accuracy_sign(w, w0, X, y):
    z = X @ w + w0
    pred = np.where(z >= 0.0, +1, -1)
    return (pred == y).mean()

# -------------------- shared init --------------------
def get_init(D, force_new=False):
    if (not force_new) and os.path.exists(INIT_FILE):
        obj = np.load(INIT_FILE, allow_pickle=True).item()
        return obj["w"].astype(np.float64).copy(), float(obj["w0"])
    w  = np.random.randn(D).astype(np.float64)*0.01
    w0 = float(np.random.randn()*0.01)
    np.save(INIT_FILE, {"w": w, "w0": w0})
    return w, w0

# -------------------- main --------------------
def main():
    Xtr_raw, ytr = load_mnist_0_1("train")
    Xva_raw, yva = load_mnist_0_1("test")
    Xtr, Xva = standardize_train_apply(Xtr_raw, Xva_raw)
    n, D = Xtr.shape
    print(f"Train: {Xtr.shape}, Val: {Xva.shape}")

    # (i) Full-batch Nesterov (AGD) curve
    w, w0 = get_init(D, force_new=True)  # new init saved; SGD runs reuse same
    losses_agd = []
    w_prev, w0_prev = w.copy(), float(w0)
    for t in range(AGD_ITERS):
        y_w  = w  + BETA_AGD*(w - w_prev)
        y_w0 = w0 + BETA_AGD*(w0 - w0_prev)
        Lval, gw, gw0 = loss_grad_full(y_w, y_w0, Xtr, ytr, LAM)
        w_next  = y_w  - ETA_AGD * gw
        w0_next = y_w0 - ETA_AGD * gw0
        losses_agd.append(Lval)
        w_prev, w0_prev = w,  w0
        w,      w0      = w_next, w0_next

    # Shared init for SGD runs
    w_init, w0_init = np.load(INIT_FILE, allow_pickle=True).item().values()

    # (ii) SGD (no momentum)
    w = w_init.astype(np.float64).copy()
    w0 = float(w0_init)
    losses_sgd = []
    idx = np.arange(n)
    updates = 0
    while updates < UPDATES:
        np.random.shuffle(idx)
        for s in range(0, n, BATCH_SIZE):
            e = min(s + BATCH_SIZE, n)
            Xb, yb = Xtr[idx[s:e]], ytr[idx[s:e]]
            # full gradient formulas but on mini-batch
            z  = Xb @ w + w0
            yz = yb * z
            sig = 1.0 / (1.0 + np.exp(yz))
            g_common = -(yb * sig)
            gw  = (Xb.T @ g_common)/Xb.shape[0] + LAM*w
            gw0 = g_common.mean() + LAM*w0
            w  -= LR_SGD * gw
            w0 -= LR_SGD * gw0
            updates += 1
            if updates % REPORT_EVERY == 0:
                Lfull, _, _ = loss_grad_full(w, w0, Xtr, ytr, LAM)
                losses_sgd.append(Lfull)
            if updates >= UPDATES:
                break

    # (iii) SGD + Nesterov momentum
    w = w_init.astype(np.float64).copy()
    w0 = float(w0_init)
    w_prev, w0_prev = w.copy(), float(w0)
    losses_sgdn = []
    updates = 0
    while updates < UPDATES:
        np.random.shuffle(idx)
        for s in range(0, n, BATCH_SIZE):
            e = min(s + BATCH_SIZE, n)
            Xb, yb = Xtr[idx[s:e]], ytr[idx[s:e]]

            # extrapolation point (Nesterov)
            y_w  = w  + BETA_SGD*(w  - w_prev)
            y_w0 = w0 + BETA_SGD*(w0 - w0_prev)

            # grad at extrapolated point
            z  = Xb @ y_w + y_w0
            yz = yb * z
            sig = 1.0 / (1.0 + np.exp(yz))
            g_common = -(yb * sig)
            gw  = (Xb.T @ g_common)/Xb.shape[0] + LAM*y_w
            gw0 = g_common.mean() + LAM*y_w0

            w_next  = y_w  - LR_SGD * gw
            w0_next = y_w0 - LR_SGD * gw0

            w_prev, w0_prev = w,  w0
            w,      w0      = w_next, w0_next
            updates += 1

            if updates % REPORT_EVERY == 0:
                Lfull, _, _ = loss_grad_full(w, w0, Xtr, ytr, LAM)
                losses_sgdn.append(Lfull)
            if updates >= UPDATES:
                break

    # Final accuracies (just to sanity check)
    print(f"SGD acc:   train {accuracy_sign(w_init,w0_init,Xtr,ytr)*100:.2f}% (init)")
    w_tmp, w0_tmp = get_init(D)  # just load to compute AGD end acc quickly if you want
    print(f"AGD acc:   train {accuracy_sign(w, w0, Xtr, ytr)*100:.2f}% (SGD+NAG end)")

    # ------------- Plot: semi-log Y -------------
    # Align x-axes: AGD uses iterations; SGD series are subsampled every REPORT_EVERY updates
    x_agd  = np.arange(1, AGD_ITERS+1)
    x_sgd  = np.arange(1, len(losses_sgd)+1) * REPORT_EVERY
    x_sgdn = np.arange(1, len(losses_sgdn)+1) * REPORT_EVERY

    # subtract min for semi-log stability (visualizing loss gap)
    eps = 1e-12
    base = min(np.min(losses_agd), np.min(losses_sgd), np.min(losses_sgdn))
    agd_gap  = np.maximum(np.array(losses_agd)  - base + eps, eps)
    sgd_gap  = np.maximum(np.array(losses_sgd)  - base + eps, eps)
    sgdn_gap = np.maximum(np.array(losses_sgdn) - base + eps, eps)

    plt.figure(figsize=(7,5))
    plt.semilogy(x_agd,  agd_gap,  label="(i) Full-batch Nesterov (AGD)")
    plt.semilogy(x_sgd,  sgd_gap,  label=f"(ii) SGD (b={BATCH_SIZE}, no momentum)")
    plt.semilogy(x_sgdn, sgdn_gap, label=f"(iii) SGD + Nesterov (b={BATCH_SIZE})")
    plt.xlabel("parameter updates")
    plt.ylabel("training loss gap (log scale)")
    plt.title("Logistic regression: GD vs SGD (with/without Nesterov)")
    plt.grid(True, which="both", ls="--", lw=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sgd_vs_agd_semilog.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
