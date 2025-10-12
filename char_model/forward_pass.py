import torch
import argparse

class OneLayerRNN(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab  = vocab_size
        self.hidden = hidden_size
        self.Wx = torch.nn.Linear(vocab_size, hidden_size, bias=True)
        self.Wh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wy = torch.nn.Linear(hidden_size, vocab_size, bias=True)

    @torch.no_grad()
    def step(self, idx_t, h_t):
        x_oh = torch.nn.functional.one_hot(idx_t, num_classes=self.vocab).float()
        h_next = torch.tanh(self.Wx(x_oh) + self.Wh(h_t))
        logits = self.Wy(h_next)
        return logits, h_next

@torch.no_grad()
def generate_text(model, stoi, itos, length=400, start_char="\n", greedy=True, temperature=1.0, device="cpu"):
    model.eval()
    idx = torch.tensor([stoi.get(start_char, 0)], device=device)  
    h   = torch.zeros(1, model.hidden, device=device)           
    out = [itos[idx.item()]]
    for _ in range(length):
        logits, h = model.step(idx, h)
        if greedy:
            idx = torch.argmax(logits, dim=-1)                   
        else:
            probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
            idx = torch.multinomial(probs, num_samples=1).squeeze(1)
        out.append(itos[idx.item()])
    return "".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="rnn_model.pt")
    ap.add_argument("--length", type=int, default=400)
    ap.add_argument("--start_char", type=str, default="\n")
    ap.add_argument("--greedy", action="store_true", help="Use argmax instead of sampling")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    try:
        ckpt = torch.load(args.ckpt, map_location=args.device)
    except Exception as e:
        print(f"Could not load checkpoint '{args.ckpt}': {e}")
        return

    V = len(ckpt["vocab"])
    hidden = ckpt.get("hidden_size", 256)

    model = OneLayerRNN(vocab_size=V, hidden_size=hidden).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    stoi, itos = ckpt["stoi"], ckpt["itos"]

    print("=== generation ===")
    print(generate_text(model, stoi, itos, length=args.length, start_char=args.start_char,
                        greedy=True, device=args.device))

    print("\n=== Temperature sampling ===")
    for T in [0.8, 1.0, 1.2]:
        print(f"\n-- temperature={T} --")
        print(generate_text(model, stoi, itos, length=args.length, start_char="T",
                            greedy=False, temperature=T, device=args.device))

if __name__ == "__main__":
    main()
