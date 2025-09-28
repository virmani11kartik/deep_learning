import numpy as np
import time
import torchvision as thv
import matplotlib.pyplot as plt
from tqdm import trange

from embedding_layer import embedding_t
from linear_layer import linear_t
from activation_layer import relu_t
from loss_layer import softmax_cross_entropy_t
from randomize_data import select_half_per_class

RNG = np.random.default_rng(42)


def load_mnist_half_per_class():
    train = thv.datasets.MNIST("./", download=True, train=True)
    val   = thv.datasets.MNIST("./", download=True, train=False)

    Xtr_full = train.data.numpy().astype(np.float32) / 255.0
    Ytr_full = train.targets.numpy().astype(np.int64)
    Xva_full = val.data.numpy().astype(np.float32) / 255.0
    Yva_full = val.targets.numpy().astype(np.int64)
    
    Xtr, Ytr = select_half_per_class(Xtr_full, Ytr_full, RNG)
    Xva, Yva = select_half_per_class(Xva_full, Yva_full, RNG)
    return Xtr, Ytr, Xva, Yva

def sample_minibatch(X, Y, batch_size=32, rng=RNG):
    n = len(Y)
    idx = rng.integers(low=0, high=n, size=batch_size, endpoint=False)  # uniform with replacement
    return X[idx], Y[idx]

def validate(l1, l2, l3, l4, Xva, Yva, batch=32):
    """
    Iterate over the entire validation set once (no randomness),
    compute average loss and error.
    """
    total_loss, total_err, total_n = 0.0, 0.0, 0
    N = len(Yva)
    for i in range(0, N, batch):
        xb = Xva[i:i+batch]
        yb = Yva[i:i+batch]

        # forward only (no grads)
        h1 = l1.forward(xb)        # (b,392)
        h2 = l2.forward(h1)        # (b,10)
        h3 = l3.forward(h2)        # (b,10)
        loss, err = l4.forward(h3, yb)

        bs = len(yb)
        total_loss += loss * bs
        total_err  += err  * bs
        total_n    += bs

    return total_loss / total_n, total_err / total_n

def main():
    # 1) Load dataset (50% per class splits as in part a)
    Xtr, Ytr, Xva, Yva = load_mnist_half_per_class()
    print("Train:", Xtr.shape, Ytr.shape, "| Val:", Xva.shape, Yva.shape)

    # 2) Initialize layers per spec: l1..l4
    l1 = embedding_t()                   # (b,28,28) -> (b,392)
    l2 = linear_t(392, 10)               # (b,392)   -> (b,10)  (one linear to logits)
    l3 = relu_t()                        # (b,10)    -> (b,10)  (per pseudocode)
    l4 = softmax_cross_entropy_t()       # computes loss & error from logits + labels
    net = [l1, l2, l3, l4]

    iters = 20000
    batch_size = 64
    lr = 0.05

    t0 = time.time()
    val_points, val_loss_hist, val_err_hist = [], [], []
    for t in trange(1, iters + 1):
        # 1) minibatch
        xb, yb = sample_minibatch(Xtr, Ytr, batch_size=batch_size)

        # 2) zero grads
        for l in net:
            l.zero_grad()

        # 3) forward: h1,h2,h3, loss
        h1 = l1.forward(xb)              # (b,392)
        h2 = l2.forward(h1)              # (b,10)
        h3 = l3.forward(h2)              # (b,10) (ReLU)
        loss, err = l4.forward(h3, yb)   # scalar loss, training error in [0,1]

        # 4) backward
        dh3 = l4.backward()
        dh2 = l3.backward(dh3)
        dh1 = l2.backward(dh2)
        dx  = l1.backward(dh1)

        # 5) SGD update (gather grads implicitly from the layer objects)
        l1.w -= lr * l1.dw
        l1.b -= lr * l1.db
        l2.W -= lr * l2.dW
        l2.b -= lr * l2.db

        if t % 10 == 0:
            vloss, verr = validate(l1, l2, l3, l4, Xva, Yva, batch=32)
            val_points.append(t)
            val_loss_hist.append(vloss)
            val_err_hist.append(verr)
            # print(f"iter {t:4d} | train loss {loss:.4f} | train err {err:.3f}")

    print(f"iter {t:4d} | train loss {loss:.4f} | train err {err:.3f}")
    print(f"Done {iters} updates in {time.time()-t0:.1f}s.")
    plt.figure(figsize=(6,4))
    plt.plot(val_points, val_loss_hist, marker='o')
    plt.xlabel('updates'); plt.ylabel('val loss'); plt.title('Validation Loss vs Updates')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('val_loss_vs_updates.png', dpi=150)

    plt.figure(figsize=(6,4))
    plt.plot(val_points, val_err_hist, marker='o')
    plt.xlabel('updates'); plt.ylabel('val error'); plt.title('Validation Error vs Updates')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('val_error_vs_updates.png', dpi=150)

if __name__ == "__main__":
    main()