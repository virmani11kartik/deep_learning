import numpy as np
from typing import Tuple

class softmax_cross_entropy_t:
    """
    Combines softmax activation and cross-entropy loss for numerical stability.
    """

    def __init__(self):
        self.probs = None  # (b, C)
        self.y = None      # (b,)

    def forward(self, logits: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        # logits: (b, C)
        # numerically stable softmax
        z = logits - np.max(logits, axis=1, keepdims=True)
        ez = np.exp(z)
        probs = ez / np.sum(ez, axis=1, keepdims=True)

        self.probs = probs.astype(np.float32)
        self.y = y.astype(np.int64)

        b = logits.shape[0]
        # average cross-entropy
        loss = -np.log(probs[np.arange(b), self.y] + 1e-12).mean()

        # error rate
        preds = np.argmax(probs, axis=1)
        err = np.mean(preds != self.y)

        return float(loss), float(err)
    
    def backward(self) -> np.ndarray:
        # dL/dz = (probs - one_hot(y)) / b
        b, C = self.probs.shape
        grad = self.probs.copy()
        grad[np.arange(b), self.y] -= 1.0
        grad /= b
        return grad.astype(np.float32)

    def zero_grad(self):
        pass