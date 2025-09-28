import numpy as np

class relu_t:
    """
    Elementwise ReLU:
      forward:  y = max(0, x)
      backward: dL/dx = dL/dy * 1{x > 0}
    No parameters; just caches a boolean mask from the forward pass.
    """
    def __init__(self):
        self.mask = None  # True where input > 0

    def forward(self, h_l: np.ndarray) -> np.ndarray:
        self.mask = (h_l > 0)
        return np.where(self.mask, h_l, 0.0)

    def backward(self, dh_next: np.ndarray) -> np.ndarray:
        # Only let gradients flow where input was positive
        return dh_next * self.mask.astype(np.float32)

    def zero_grad(self):
        pass
