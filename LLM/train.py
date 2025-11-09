import os, math, json, time, random
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
from decoder import DecoderTransformer, TransformerConfig

def read_tokenizer_vocab_size(tokenizer_path: str) -> int:
    tok = Tokenizer.from_file(tokenizer_path)
    return tok.get_vocab_size(), tok

def split_sentences_from_files(paths: List[str]) -> List[str]:
    sents = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sents.append(line)
    return sents

def make_splits(sents: List[str], val_frac=0.1, seed=1337) -> Tuple[List[str], List[str]]:
    random.Random(seed).shuffle(sents)
    n_val = int(len(sents) * val_frac)
    return sents[n_val:], sents[:n_val]

class SentenceDataset(Dataset):
    def __init__(self, sentences: List[str], tokenizer: Tokenizer, max_len: int):
        self.sentences = sentences
        self.tok = tokenizer
        self.max_len = max_len
        self.pad_id = self.tok.token_to_id("[PAD]")
        assert self.pad_id is not None, "Tokenizer must contain [PAD]."

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        s = self.sentences[idx]
        enc = self.tok.encode(s)  
        ids = enc.ids
        ids = ids[: self.max_len]
        return torch.tensor(ids, dtype=torch.long)
    
def collate_pad_shift(
    batch_ids: List[torch.Tensor],
    pad_id: int,
    context_len: int
):
    lengths = [min(len(x), context_len) for x in batch_ids]
    T = min(max(lengths), context_len)
    B = len(batch_ids)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, T), dtype=torch.bool)

    for i, seq in enumerate(batch_ids):
        L = min(len(seq), T)
        input_ids[i, :L] = seq[:L]
        attn_mask[i, :L] = True

    target_ids = input_ids.clone()
    target_ids[:, :-1] = input_ids[:, 1:]
    target_ids[:, -1] = pad_id  

    return input_ids, target_ids, attn_mask

@torch.no_grad()
def evaluate(model, loader, pad_id: int, device: torch.device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for input_ids, target_ids, attn_mask in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        attn_mask = attn_mask.to(device)

        logits = model(input_ids, attn_mask=attn_mask) 
        vocab = logits.size(-1)
        loss = F.cross_entropy(
            logits.view(-1, vocab),
            target_ids.view(-1),
            ignore_index=pad_id,
            reduction="sum"
        )
        n_valid = (target_ids != pad_id).sum().item()
        total_loss += loss.item()
        total_tokens += n_valid
    return total_loss / max(1, total_tokens)

def moving_avg(x, k=100):
    if len(x) == 0: return x
    out = []
    s = 0.0
    q = []
    for v in x:
        q.append(v); s += v
        if len(q) > k: s -= q.pop(0)
        out.append(s / len(q))
    return out

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "data")
    tokenizer_path = os.path.join(base, "tokenizer.json")
    train_files = [
        os.path.join(data_dir, "shakespeare.txt"),
        os.path.join(data_dir, "hounds_of_baskervilles.txt"),
        os.path.join(data_dir, "war_and_peace.txt"),
    ]

    vocab_size, tok = read_tokenizer_vocab_size(tokenizer_path)
    pad_id = tok.token_to_id("[PAD]")
    assert pad_id is not None, "Tokenizer must define [PAD]."

    sents = split_sentences_from_files(train_files)
    train_sents, val_sents = make_splits(sents, val_frac=0.1, seed=1337)

    cfg = TransformerConfig(
        vocab_size=vocab_size,
        context_len=128,
        d_model=256,
        num_heads=2,
        head_dim=32,
        num_layers=3,
        dropout=0.1,
        tie_weights=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderTransformer(cfg).to(device)

    train_ds = SentenceDataset(train_sents, tok, max_len=cfg.context_len)
    val_ds   = SentenceDataset(val_sents, tok, max_len=cfg.context_len)

    def collate_fn(batch):
        return collate_pad_shift(batch, pad_id=pad_id, context_len=cfg.context_len)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    max_updates = 20_000
    log_every = 50
    eval_every = 1000
    train_losses = []
    val_points_x, val_losses = [], []

    model.train()
    u = 0
    t0 = time.time()
    while u < max_updates:
        for input_ids, target_ids, attn_mask in train_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            attn_mask = attn_mask.to(device)

            logits = model(input_ids, attn_mask=attn_mask) 
            vocab = logits.size(-1)

            loss = F.cross_entropy(
                logits.view(-1, vocab),
                target_ids.view(-1),
                ignore_index=pad_id
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            train_losses.append(loss.item())
            u += 1

            if u % log_every == 0:
                ma = moving_avg(train_losses, k=200)[-1]
                dt = time.time() - t0
                print(f"update {u:6d} | loss {loss.item():.4f} | ma(200) {ma:.4f} | {dt:.1f}s")

            if u % eval_every == 0:
                val_loss = evaluate(model, val_loader, pad_id, device)
                val_points_x.append(u)
                val_losses.append(val_loss)
                print(f"[VAL] update {u} | val_loss {val_loss:.4f}")

            if u >= max_updates:
                break

    ckpt_path = os.path.join(base, "decoder_ckpt.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    plt.figure()
    plt.plot(moving_avg(train_losses, k=200), label="train (MA-200)")
    if val_points_x:
        xs = list(range(1, len(train_losses)+1))
        plt.scatter(val_points_x, val_losses, marker="x", label="val (per 1000 updates)")
    plt.xlabel("weight updates")
    plt.ylabel("loss (cross-entropy)")
    plt.legend()
    plt.tight_layout()
    out_plot = os.path.join(base, "loss_plot.png")
    plt.savefig(out_plot, dpi=150)
    print(f"Saved plot to {out_plot}")