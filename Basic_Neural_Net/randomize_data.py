
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision as thv


RNG = np.random.default_rng(42)
SAVE_DIR = "./outputs_hw1_p3a"
os.makedirs(SAVE_DIR, exist_ok=True)

def select_half_per_class(X, y, rng):
    picked = []
    for c in range(10):
        idx = np.nonzero(y == c)[0]
        rng.shuffle(idx)
        k = idx.shape[0] // 2  # 50% of this class within the split
        picked.append(idx[:k])
    picked = np.concatenate(picked)
    rng.shuffle(picked)
    return X[picked], y[picked]

def plot_random_grid(X, y, n=16, outpath="p3a_samples.pdf"):
    assert X.ndim == 3 and X.shape[1:] == (28, 28)
    idx = RNG.choice(X.shape[0], size=n, replace=False)
    imgs, labels = X[idx], y[idx]

    rows = cols = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2))
    axes = axes.ravel()

    for i in range(rows*cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i], cmap="gray")
            ax.set_title(f"label={labels[i]}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved sample grid to {outpath}")    

def main():
    train = thv.datasets.MNIST("./", download=True, train=True)
    val   = thv.datasets.MNIST("./", download=True, train=False)

    Xtr_full = train.data.numpy().astype(np.float32) / 255.0  # (60000, 28, 28)
    Ytr_full = train.targets.numpy().astype(np.int64)
    Xva_full = val.data.numpy().astype(np.float32) / 255.0    # (10000, 28, 28)
    Yva_full = val.targets.numpy().astype(np.int64)

    Xtr, Ytr = select_half_per_class(Xtr_full, Ytr_full, RNG)
    Xva, Yva = select_half_per_class(Xva_full, Yva_full, RNG)

    print("Train subset shape:", Xtr.shape, Ytr.shape)
    print("Val   subset shape:", Xva.shape, Yva.shape)

    def counts(y):
        return np.bincount(y, minlength=10)

    print("Train class counts:", counts(Ytr))
    print("Val   class counts:", counts(Yva))

    plot_random_grid(Xtr, Ytr, n=16, outpath=os.path.join(SAVE_DIR, "p3a_samples.pdf"))

if __name__ == "__main__":
    main()