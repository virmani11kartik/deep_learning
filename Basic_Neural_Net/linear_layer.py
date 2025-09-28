import numpy as np
RNG  = np.random.default_rng(42)

class linear_t:

    def __init__(self, d_in: int, d_out: int):
        # Gaussian init
        W = RNG.normal(size=(d_out, d_in)).astype(np.float32)
        b = RNG.normal(size=(d_out,)).astype(np.float32)

        # Normalize
        W /= np.linalg.norm(W) + 1e-12
        b /= np.linalg.norm(b) + 1e-12

        self.W = W
        self.b = b

        # Grads
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.hl = None   # cache input

    def forward(self, h_l: np.ndarray) -> np.ndarray:
        """
        h_l: (b, d_in) → returns (b, d_out)
        """
        self.hl = h_l
        return h_l @ self.W.T + self.b   # (b,d_out)

    def backward(self, dh_next: np.ndarray) -> np.ndarray:
        """
        dh_next: dL/d(h_out) of shape (b, d_out)
        Returns: dL/d(h_in) of shape (b, d_in)
        """
        assert self.hl is not None, "Call forward() first"

        # Gradients wrt W, b
        # dW = (d_out,d_in) from (b,d_out)^T @ (b,d_in)
        self.dW = dh_next.T @ self.hl
        self.db = dh_next.sum(axis=0)

        # Gradient wrt input
        dx = dh_next @ self.W   # (b,d_in)
        return dx

    def zero_grad(self):
        self.dW[...] = 0.0
        self.db[...] = 0.0

