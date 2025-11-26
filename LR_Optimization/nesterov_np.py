import numpy as np
from mnist_np import Xtr, ytr, Xva, yva, loss_and_grad, accuracy, INIT_FILE


def main():

    # hyper-parameters
    lam = 1e-3        # same λ
    L   = 5.0         # choose/tune; try a few values and pick the best slope
    m   = lam
    kappa = L / m
    beta  = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)   # momentum
    eta   = 1.0 / L                                       # step size
    T     = 400

    init = np.load(INIT_FILE, allow_pickle=True).item()
    w  = init["w"].astype(np.float64).copy()
    w0 = float(init["w0"])
    w_prev, w0_prev = w.copy(), float(w0)


    losses = []
    for t in range(T):
        # extrapolation point
        y_w  = w  + beta * (w  - w_prev)
        y_w0 = w0 + beta * (w0 - w0_prev)

        # full-batch gradient at y
        Lval, gw, gw0 = loss_and_grad(y_w, y_w0, Xtr, ytr, lam)

        # take step
        w_next  = y_w  - eta * gw
        w0_next = y_w0 - eta * gw0

        losses.append(Lval)
        w_prev, w0_prev = w,  w0
        w,      w0      = w_next, w0_next

    print(f"AGD: train acc {accuracy(Xtr,ytr,w,w0)*100:.2f}%, val acc {accuracy(Xva,yva,w,w0)*100:.2f}%")

    # slope on semi-log
    eps = 1e-12
    f_star_hat = np.min(losses)
    y_log = np.log(np.maximum(np.array(losses) - f_star_hat + eps, eps))
    burn = int(0.1*T)
    coef = np.polyfit(np.arange(burn, T), y_log[burn:], 1)
    slope = coef[0]
    print(f"AGD semi-log slope: {slope:.4f}  (expect ≈ -1/sqrt(kappa) = {-1/np.sqrt(kappa):.4f})")


if __name__ == "__main__":
    main()