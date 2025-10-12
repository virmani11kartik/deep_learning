import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt

# 32 characters in a block
class CharSeqDataset(Dataset):
    def __init__(self, text, stoi, block_size=32, stride =1):
        self.block_size = block_size
        self.data = text
        self.stride = stride
        self.vocab_size = len(set(text))
        self.stoi = stoi
        self.encoded_data = [self.stoi[c] for c in text]
        self.N = max(0, (len(text) - block_size - 1) // stride + 1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        start = idx * self.stride
        seq = self.data[start:start + self.block_size]
        nxt = self.data[start + 1:start + self.block_size + 1]
        x = torch.tensor([self.stoi[c] for c in seq], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in nxt], dtype=torch.long)
        return x, y
    
class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(RNN, self).__init__()
        self.vocab = vocab_size
        self.hidden = hidden_size
        self.Wx = nn.Linear(vocab_size, hidden_size, bias=True)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wy = nn.Linear(hidden_size, vocab_size, bias=True)
        for m in [self.Wx, self.Wh, self.Wy]:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, x, h=None):
        B, T = x.shape
        x_oh = F.one_hot(x, num_classes=self.vocab).float()
        if h is None:
            h = torch.zeros(B, self.hidden, device=x.device)
        logits = []
        for t in range(T):
            xt = x_oh[:, t, :]
            h = torch.tanh(self.Wx(xt) + self.Wh(h))
            yt = self.Wy(h)
            logits.append(yt.unsqueeze(1))
        logits = torch.cat(logits, dim=1)
        return logits, h

# util
@torch.no_grad()
def batch_error(logits, targets):
    pred = logits.argmax(dim=-1)
    correct = (pred == targets).float().mean().item()
    return 1.0 - correct

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_count = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        logits, _ = model(X)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
        err  = batch_error(logits, Y) * X.numel()  
        total_loss   += loss.item()
        total_error    += err
        total_count += X.numel()
    avg_loss = total_loss / max(1, total_count)
    avg_err  = total_error  / max(1, total_count)
    return avg_loss, avg_err  

def plot_training(updates, train_loss, train_err, out ="training.png"):
    plt.figure(figsize=(8,4.5))
    plt.plot(updates, train_loss, label="Train mini-batch CE")
    plt.plot(updates, train_err,  label="Train mini-batch error")
    plt.xlabel("Weight updates")
    plt.ylabel("Value")
    plt.title("RNN training (mini-batch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--block_size", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max_updates", type=int, default=10000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    shakespeare_path = os.path.join(data_dir, 'shakespeare.txt')
    baskerville_path = os.path.join(data_dir, 'hounds_of_baskervilles.txt')
    war_and_peace_path = os.path.join(data_dir, 'war_and_peace.txt')

    try:
        with open(shakespeare_path, "r", encoding="utf-8") as f:
            text1 = f.read()
        with open(baskerville_path, "r", encoding="utf-8") as f:
            text2 = f.read()
        with open(war_and_peace_path, "r", encoding="utf-8") as f:
            text3 = f.read()
        text = text1 + text2 + text3
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    all_chars = sorted(list(set(text)))
    vocab_size = len(all_chars)
    stoi = { ch:i for i,ch in enumerate(all_chars) }
    itos = { i:ch for i,ch in enumerate(all_chars) }
    print("all chars:",''.join(all_chars))
    print("vocab size:",vocab_size)

    n = len(text)
    training_data = int(n * 0.9)
    validation_data = int(n * 0.95)
    train_text = text[:training_data]
    val_text = text[training_data:validation_data]
    test_text = text[validation_data:]

    train_ds = CharSeqDataset(train_text, stoi, block_size=args.block_size, stride=args.stride)
    val_ds = CharSeqDataset(val_text, stoi, block_size=args.block_size, stride=args.stride)
    test_ds = CharSeqDataset(test_text, stoi, block_size=args.block_size, stride=args.stride)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = args.device
    model = RNN(vocab_size, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    updates = []
    train_loss = []
    train_err = []
    step = 0
    train_iter = iter(train_dl)

    while step < args.max_updates:
        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            X, Y = next(train_iter)
        X, Y = X.to(device), Y.to(device)

        model.train()
        logits, _ = model(X)
        loss = loss_fn(logits.reshape(-1, vocab_size), Y.reshape(-1))
        err  = batch_error(logits, Y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        updates.append(step)
        train_loss.append(loss.item())
        train_err.append(err)

        if step % 100 == 0:
            print(f"Step {step}: Train CE {loss.item():.4f}, Train error {err:.4f}")

        if step % args.eval_every == 0:
            val_loss, val_err = evaluate(model, val_dl, device)
            print(f"--- Step {step}: Val CE {val_loss:.4f}, Val error {val_err:.4f} ---")
        
    plot_training(updates, train_loss, train_err, out="rnn_training.png")
    test_loss, test_err = evaluate(model, test_dl, device)
    print(f"Final Test CE {test_loss:.4f}, Test error {test_err:.4f}")

    save_path = "rnn_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": all_chars,
        "stoi": stoi,
        "itos": itos,
        "hidden_size": args.hidden,
        "seq_len": args.block_size
    }, save_path)
    print(f"Saved trained model to {save_path}")

if __name__ == "__main__":
    main()

