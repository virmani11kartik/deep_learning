import math, time, random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def true_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(10.0 * math.pi * (x**4))

def sample_dataset(n: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(int(seed))
    x = torch.rand(n, 1, generator=g)          
    y = true_function(x)                        
    return x, y

class MLP(nn.Module):
    """1D -> hidden -> hidden -> 1D, with BatchNorm + ReLU."""
    def __init__(self, width: int = 128, depth: int = 3):
        super().__init__()
        assert depth in (2, 3)

        layers = []
        layers.append(nn.Linear(1, width))
        layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU(inplace=True))

        if depth == 3:
            layers.append(nn.Linear(width, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    
@dataclass
class TrainConfig:
    epochs: int = 3000
    batch_size: int = 64
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    width: int = 256
    depth: int = 3

def train_mlp(x: torch.Tensor, y: torch.Tensor, cfg: TrainConfig, device = None) ->Tuple[MLP, List[float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n=x.shape[0]
    x=x.to(device)
    y=y.to(device)

    model = MLP(width=cfg.width, depth=cfg.depth).to(device)
    model.train()

    opt = torch.optim.SGD(model.parameters(),
                          lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    
    schd = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr*0.1)

    loss_hist=[]
    for epochs in range(1, cfg.epochs+1):
        idx = torch.randperm(n,device=device)
        for start in range(0,n,cfg.batch_size):
            end = min(start+cfg.batch_size, n)
            b = idx[start:end]
            xb, yb = x[b], y[b]

            pred = model(xb)
            loss = F.mse_loss(pred,yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        
        schd.step()
        loss_hist.append(loss.item())

        if epochs % 1 == 0 or epochs == 1 or epochs ==cfg.epochs:
            with torch.no_grad():
                model.eval()
                train_pred = model(x)
                train_loss = F.mse_loss(train_pred, y).item()
            print(f"epoch {epochs:4d} | batch_loss {loss.item():.6f} | train_mse {train_loss:.6f} | lr {schd.get_last_lr()[0]:.5e}")

            if train_loss < 1e-6:
                print("Early stop: near-zero training error reached.")
                break
        
        model.eval()
        return model, loss_hist
    
@torch.no_grad()
def max_abs_error(model: MLP, a: float, b: float, num_pts:int = 1000, device = None)->float:
        device = device or next(model.parameters()).device
        xs = torch.linspace(a, b, num_pts, device=device).unsqueeze(1) 
        y_true = true_function(xs)
        y_hat = model(xs)
        err = torch.abs(y_hat - y_true)
        return float(err.max().item())

def evaluate_deltas(model: MLP, num_pts: int = 1000) -> Tuple[float, float]:
    device = next(model.parameters()).device
    din = max_abs_error(model, 0.0, 1.0, num_pts=num_pts, device=device)
    dout = max_abs_error(model, 0.0, 1.5, num_pts=num_pts, device=device)
    return din, dout

## for experiment
# once
def run_once(n: int, seed: int = 0, cfg: TrainConfig = TrainConfig()) -> Tuple[float, float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = sample_dataset(n=n, seed=seed)
    model, _ = train_mlp(x, y, cfg, device=device)
    with torch.no_grad():
        train_mse = F.mse_loss(model(x.to(device)), y.to(device)).item()
    delta_in, delta_out = evaluate_deltas(model, num_pts=1000)
    print(f"[n={n:4d}] train_mse={train_mse:.6e} | δ_in={delta_in:.6e} | δ_out={delta_out:.6e}")
    return train_mse, delta_in, delta_out

# many model
def sweep_over_n(n_list: List[int], base_seed: int = 0, cfg: TrainConfig = TrainConfig()):
    rows = []
    for i, n in enumerate(n_list):
        seed = base_seed + i
        t0 = time.time()
        train_mse, din, dout = run_once(n=n, seed=seed, cfg=cfg)
        dt = time.time() - t0
        rows.append((n, train_mse, din, dout, dt))
    print("\nSummary (n, train_mse, δ_in, δ_out, seconds):")
    for r in rows:
        print(r)
    return rows

if __name__ == "__main__":
    cfg = TrainConfig(
        epochs=3000,     
        batch_size=64,
        lr=1e-2,
        momentum=0.9,
        weight_decay=1e-4,
        width=256,
        depth=3,        
    )
    # run_once(n=64, seed=42, cfg=cfg)
    n_list = [8, 16, 32, 64, 128, 256]
    sweep_over_n(n_list, base_seed=0, cfg=cfg)





