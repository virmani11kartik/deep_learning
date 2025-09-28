import numpy as np
from linear_layer import linear_t

RNG = np.random.default_rng(1234)

def grad_check_linear(
        b: int =1,
        d_in: int = 7,
        d_out: int = 5,
        trials_w: int = 10,
        trials_b: int =5,
        trials_x: int = 5,
        eps_list = (1e-6, 5e-6, 1e-5),
):

    layer = linear_t(d_in=d_in, d_out=d_out)
    layer.W = layer.W.astype(np.float64, copy=True)
    layer.b = layer.b.astype(np.float64, copy=True)
    x = RNG.normal(size=(b, d_in)).astype(np.float64)
    u = RNG.normal(size=(b, d_out)).astype(np.float64)
    def forward_f(x_):
        return layer.forward(x_)
    
    def scalar_obj(x_):
        return float(np.sum(forward_f(x_) * u))
    
    layer.zero_grad()
    y = forward_f(x)
    dx = layer.backward(u)   # (b,d_in)

    dW_ref = u.T @ x      # (d_out, d_in)
    db_ref = u.sum(axis=0)
    dx_ref = u @ layer.W

    print("Sanity vs closed-form (analytic vs reference):")
    print("  ||dW - dW_ref|| / ||dW_ref|| =", np.linalg.norm(layer.dW - dW_ref) / (np.linalg.norm(dW_ref) + 1e-12))
    print("  ||db - db_ref|| / ||db_ref|| =", np.linalg.norm(layer.db - db_ref) / (np.linalg.norm(db_ref) + 1e-12))
    print("  ||dx - dx_ref|| / ||dx_ref|| =", np.linalg.norm(dx - dx_ref) / (np.linalg.norm(dx_ref) + 1e-12))

    best = {"W": 1.0, "b": 1.0, "x": 1.0}

    for eps in eps_list:
        rel_W, rel_b, rel_x = [], [], []

    for _ in range (trials_w):
        i = int(RNG.integers(0, d_out))
        j = int(RNG.integers(0, d_in))

        base = layer.W[i,j]
        layer.W[i,j] = base + eps
        f_plus = scalar_obj(x)
        layer.W[i,j] = base - eps
        f_minus = scalar_obj(x)
        layer.W[i,j] = base

        fd = (f_plus - f_minus) / (2 * eps)
        an = layer.dW[i,j]
        rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
        rel_W.append(rel)
    
    for _ in range(trials_b):
        i = int(RNG.integers(0, d_out))

        base = layer.b[i]
        layer.b[i] = base + eps
        f_plus = scalar_obj(x)
        layer.b[i] = base - eps
        f_minus = scalar_obj(x)
        layer.b[i] = base

        fd = (f_plus - f_minus) / (2 * eps)
        an = layer.db[i]
        rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
        rel_b.append(rel)

    for _ in range(trials_x):
        j = int(RNG.integers(0, d_in))
        base = x[0,j]
        x[0,j] = base + eps
        f_plus = scalar_obj(x)
        x[0,j] = base - eps
        f_minus = scalar_obj(x)
        x[0,j] = base
        fd = (f_plus - f_minus) / (2 * eps)
        an = dx[0,j]
        rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
        rel_x.append(rel)

    def summarize(name, errs):
        errs = np.array(errs, dtype=np.float64)
        print(f"{name}: max {errs.max():.3e}, min {errs.min():.3e}, mean {errs.mean():.3e}")    
    
    summarize("W", rel_W)
    summarize("b", rel_b)
    summarize("x", rel_x)

    ok = (np.max(rel_W) < 1e-5 and
          np.max(rel_b) < 1e-5 and
          np.max(rel_x) < 1e-5)
    print ("PASS" if ok else "FAIL")

if __name__ == "__main__":
    grad_check_linear()
