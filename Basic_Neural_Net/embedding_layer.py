import numpy as np
RNG = np.random.default_rng(42)

class embedding_t:
    def __init__(self):
        w = RNG.normal(size=(4,4,8)).astype(np.float32)
        b = RNG.normal(size=(8,)).astype(np.float32)
        w /= (np.linalg.norm(w) + 1e-12)
        b /= (np.linalg.norm(b) + 1e-12)
        self.w, self.b = w, b
        self.hl = None
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    @staticmethod
    def _extract_patches(x28: np.ndarray) -> np.ndarray:
        b = x28.shape[0]
        return x28.reshape(b,7,4,7,4).transpose(0,1,3,2,4)

    def forward(self, h_l: np.ndarray) -> np.ndarray:
        self.hl = h_l
        bsz = h_l.shape[0]
        patches = self._extract_patches(h_l)
        out = np.einsum('bxyij,ijc->bxyc', patches, self.w) + self.b
        return out.reshape(bsz, -1)

    def backward(self, dh_next: np.ndarray) -> np.ndarray:
        bsz = self.hl.shape[0]
        dh = dh_next.reshape(bsz,7,7,8)
        patches = self._extract_patches(self.hl)
        self.dw = np.einsum('bxyij,bxyc->ijc', patches, dh)
        self.db = dh.sum(axis=(0,1,2))
        dx = np.zeros_like(self.hl)
        for x in range(7):
            for y in range(7):
                contrib = np.tensordot(dh[:,x,y,:], self.w, axes=([1],[2]))
                dx[:, x*4:(x+1)*4, y*4:(y+1)*4] += contrib
        return dx

    def zero_grad(self):
        self.dw[...] = 0.0
        self.db[...] = 0.0
