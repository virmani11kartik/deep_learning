
import numpy as np
from activation_layer import relu_t 

RNG = np.random.default_rng(42)

def grad_check_relu(d: int = 50, trials_x: int = 20, eps_list=(1e-6, 5e-6, 1e-5)):
    relu = relu_t()
    x = RNG.normal(size=(1, d)).astype(np.float64)

    x[np.isclose(x, 0.0)] = 1e-3

    u = RNG.normal(size=(1, d)).astype(np.float64)

    y = relu.forward(x.astype(np.float32))  # layer may use float32 internally; that's fine
    y64 = np.maximum(x, 0.0)
    f = float(np.sum(y64 * u))

    dy = u.astype(np.float32)
    dx_analytic = relu.backward(dy).astype(np.float64)  # shape (1,d)

    best_x = 1.0
    for eps in eps_list:
        errs = []
        for _ in range(trials_x):
            j = int(RNG.integers(0, d))
            base = x[0, j]

            if abs(base) < 1e-8:
                continue

            x[0, j] = base + eps
            f_plus = float(np.sum(np.maximum(x, 0.0) * u))

            x[0, j] = base - eps
            f_minus = float(np.sum(np.maximum(x, 0.0) * u))

            x[0, j] = base

            fd = (f_plus - f_minus) / (2 * eps)
            an = dx_analytic[0, j]
            rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-16)
            errs.append(rel)

        if errs:
            print(f"EPS={eps:g}  |  x: max {np.max(errs):.3e}, mean {np.mean(errs):.3e}, median {np.median(errs):.3e}")
            best_x = min(best_x, float(np.max(errs)))

    print("\nRESULT:", "PASS" if best_x < 1e-6 else "BORDERLINE/PASS" if best_x < 5e-6 else "FAIL",
          "| best max rel.err:", best_x)

if __name__ == "__main__":
    grad_check_relu()
