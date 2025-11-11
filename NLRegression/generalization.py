import numpy as np
import torch
import matplotlib.pyplot as plt
from nonlinear_regression import sample_dataset, train_mlp, evaluate_deltas, TrainConfig
from nonlinear_regression import MLP

def run_generalization_experiment():
    ns = np.logspace(1, 3, 20).astype(int)
    seeds_per_n = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TrainConfig(
        epochs=3000,
        batch_size=64,  
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.0,
        width=256,
        depth=3,
    )

    mean_din, std_din = [], []
    mean_dout, std_dout = [], []

    for n in ns:
        dins, douts = [], []
        for rep in range(seeds_per_n):
            seed = 1000 * rep + n
            x,y = sample_dataset(n=n, seed=seed)
            model,_ = train_mlp(x,y,cfg,device=device)
            din, dout = evaluate_deltas(model)
            dins.append(din)
            douts.append(dout)
            print(f"n={n:4d} rep={rep+1}/5 | δ_in={din:.3e} | δ_out={dout:.3e}")
        mean_din.append(np.mean(dins))
        std_din.append(np.std(dins))
        mean_dout.append(np.mean(douts))
        std_dout.append(np.std(douts))
        print(f"--> n={n:4d}: mean δ_in={mean_din[-1]:.3e}, mean δ_out={mean_dout[-1]:.3e}")


    plt.figure(figsize=(7,5))
    plt.errorbar(ns, np.log(mean_din), yerr=std_din, fmt='-o', capsize=3,
                    label=r'$\log \delta_{\mathrm{in}}(n)$')
    plt.errorbar(ns, np.log(mean_dout), yerr=std_dout, fmt='-s', capsize=3,
                 label=r'$\log \delta_{\mathrm{out}}(n)$')
    plt.xscale('log')
    plt.xlabel("Number of training points n (log scale)")
    plt.ylabel("log δ (mean ± std)")
    plt.legend()
    plt.title("Generalization within vs outside convex hull")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig("delta_vs_n.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    run_generalization_experiment()