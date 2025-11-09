import os, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from decoder import DecoderTransformer, TransformerConfig

@torch.no_grad()
def sample_autoregressive(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 8,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_at_sep: bool = True,
    device: torch.device | None = None,
):
    device = device or next(model.parameters()).device

    enc = tokenizer.encode(prompt)
    ids = enc.ids[:]  # list of ints

    pad_id = tokenizer.token_to_id("[PAD]")
    sep_id = tokenizer.token_to_id("[SEP]")
    ctx_len = model.cfg.context_len

    ids = ids[-ctx_len:]

    for _ in range(max_new_tokens):
        ctx = ids[-ctx_len:]
        x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)   
        attn_mask = torch.ones_like(x, dtype=torch.bool, device=device)       

        # Forward pass
        logits = model(x, attn_mask=attn_mask)  
        next_logits = logits[0, -1, :]          

        # Temperature
        if temperature is not None and temperature > 0:
            next_logits = next_logits / temperature

        # Top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(next_logits, k=min(top_k, next_logits.numel()))
            cutoff = v[-1]
            next_logits = torch.where(next_logits >= cutoff, next_logits, torch.tensor(float("-inf"), device=device))

        # Top-p (nucleus) filtering
        if top_p is not None and 0 < top_p < 1:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cdf = torch.cumsum(probs, dim=-1)
            mask = cdf > top_p
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            unsorted = torch.full_like(next_logits, float("-inf"))
            unsorted[sorted_idx] = sorted_logits
            next_logits = unsorted

        # Sample next token
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        ids.append(next_id)

        if stop_at_sep and sep_id is not None and next_id == sep_id:
            break

    text = tokenizer.decode(ids)
    return text, ids

def load_model_and_tokenizer(base_dir: str):
    tok_path = os.path.join(base_dir, "tokenizer.json")
    ckpt_path = os.path.join(base_dir, "decoder_ckpt.pt")

    tokenizer = Tokenizer.from_file(tok_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TransformerConfig(**ckpt["cfg"])
    model = DecoderTransformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    model, tok, device = load_model_and_tokenizer(base)

    prompts = [
        "The Eiffel Tower is in Paris",
        "Sherlock Holmes looked at the",
        "To be or not to",
    ]

    for p in prompts:
        out, ids = sample_autoregressive(
            model,
            tok,
            prompt=p,
            max_new_tokens=8,       # 7–8 word completions per the assignment
            temperature=0.9,        # try 0.7–1.2
            top_k=40,               # small top-k helps early in training
            top_p=None,             # or use top_p=0.9
            stop_at_sep=True,
            device=device,
        )
        print("\nPROMPT:", p)
        print("GEN   :", out)