import embedding_layer, inspect
print("IMPORTING:", embedding_layer.__file__)
print("HAS embedding_t:", hasattr(embedding_layer, "embedding_t"))

import numpy as np
from embedding_layer import embedding_t

RNG = np.random.default_rng(42)

def grad_check_embedding(
    trials_w: int = 10,   # how many random W entries to test
    trials_b: int = 5,    # how many random b entries to test
    trials_x: int = 5,    # how many random input pixels to test
    eps_list = (1e-6, 5e-6, 1e-5)  # sweep a few eps values
):
    """
    Finite-difference gradient check for embedding_t:
      forward:  (b,28,28) -> (b,392)
      backward: given upstream grad (b,392), returns dL/dx (b,28,28) and stores dW (4,4,8), db (8,)

    Scalar objective:
      f(x, W, b) = sum( forward(x) * U )
    where U is a fixed upstream gradient of shape (1,392).
    """

    # ---- 1) Instantiate layer and cast to float64 for numerical stability
    layer = embedding_t()
    layer.w = layer.w.astype(np.float64, copy=True)  # (4,4,8)
    layer.b = layer.b.astype(np.float64, copy=True)  # (8,)

    # ---- 2) Make a single-sample input and a fixed upstream gradient U
    x = RNG.normal(size=(1, 28, 28)).astype(np.float64)
    U = RNG.normal(size=(1, 7*7*8)).astype(np.float64)  # (1,392)

    # Helper: recompute forward and return scalar f = <out, U>
    def scalar_obj(x_):
        out = layer.forward(x_)          # (1,392)
        return float(np.sum(out * U))

    # ---- 3) Forward once and compute analytic grads via backward(U)
    y = layer.forward(x)                 # caches input
    layer.zero_grad()
    dx = layer.backward(U)               # sets layer.dw, layer.db
    dW_an = layer.dw.copy()
    db_an = layer.db.copy()
    dx_an = dx.copy()

    # ---- 4) Finite-diff checks with multiple eps; keep best max error for each group
    best = {"W": 1.0, "b": 1.0, "x": 1.0}

    for eps in eps_list:
        rel_W, rel_b, rel_x = [], [], []

        # --- W entries: pick random (i,j,c)
        for _ in range(trials_w):
            i = int(RNG.integers(0, 4))
            j = int(RNG.integers(0, 4))
            c = int(RNG.integers(0, 8))
            base = layer.w[i, j, c]

            layer.w[i, j, c] = base + eps
            f_plus = scalar_obj(x)

            layer.w[i, j, c] = base - eps
            f_minus = scalar_obj(x)

            layer.w[i, j, c] = base  # restore

            fd = (f_plus - f_minus) / (2 * eps)
            an = dW_an[i, j, c]
            rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-16)
            rel_W.append(rel)

        # --- b entries: pick random c
        for _ in range(trials_b):
            c = int(RNG.integers(0, 8))
            base = layer.b[c]

            layer.b[c] = base + eps
            f_plus = scalar_obj(x)

            layer.b[c] = base - eps
            f_minus = scalar_obj(x)

            layer.b[c] = base  # restore

            fd = (f_plus - f_minus) / (2 * eps)
            an = db_an[c]
            rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-16)
            rel_b.append(rel)

        # --- x entries: pick random pixel (r, s)
        for _ in range(trials_x):
            r = int(RNG.integers(0, 28))
            s = int(RNG.integers(0, 28))
            base = x[0, r, s]

            x[0, r, s] = base + eps
            f_plus = scalar_obj(x)

            x[0, r, s] = base - eps
            f_minus = scalar_obj(x)

            x[0, r, s] = base  # restore

            fd = (f_plus - f_minus) / (2 * eps)
            an = dx_an[0, r, s]
            rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-16)
            rel_x.append(rel)

        # --- Report for this eps
        print(f"\nEPS={eps:g}")
        if rel_W:
            print(f"  W: max {np.max(rel_W):.3e}, mean {np.mean(rel_W):.3e}, median {np.median(rel_W):.3e}")
            best["W"] = min(best["W"], float(np.max(rel_W)))
        if rel_b:
            print(f"  b: max {np.max(rel_b):.3e}, mean {np.mean(rel_b):.3e}, median {np.median(rel_b):.3e}")
            best["b"] = min(best["b"], float(np.max(rel_b)))
        if rel_x:
            print(f"  x: max {np.max(rel_x):.3e}, mean {np.mean(rel_x):.3e}, median {np.median(rel_x):.3e}")
            best["x"] = min(best["x"], float(np.max(rel_x)))

    # ---- Final result
    ok = (best["W"] < 1e-5) and (best["b"] < 1e-5) and (best["x"] < 1e-5)
    print("\nRESULT:", "PASS" if ok else "BORDERLINE/PASS" if max(best.values()) < 5e-5 else "FAIL")
    print("Best max rel.errs:", best)


if __name__ == "__main__":
    grad_check_embedding()
