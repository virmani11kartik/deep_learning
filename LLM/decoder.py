import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F 

# POSITIONAL ENCODING
class SCPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 2048):
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        self.register_buffer("pe",pe)

    def forward(self, x:torch.Tensor) ->torch.Tensor:
        T = x.size(1)
        return x+self.pe[:T].unsqueeze(0)
    
# MULTI HEAD SELF ATTENTION
class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        assert d_model > 0 and num_heads > 0 and head_dim > 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        self.W_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.inner_dim, bias=False)

        self.W_o = nn.Linear(self.inner_dim, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.W_q(x)  
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        if key_padding_mask is not None:
            pad_k = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            attn_scores = attn_scores.masked_fill(pad_k, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = attn_probs @ v
        context = context.transpose(1, 2).contiguous().view(B, T, self.inner_dim)

        out = self.W_o(context) 
        out = self.proj_dropout(out)
        return out
    
# POSITION WISE MLP
class PositionwiseMLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * d_model, bias=False)
        self.fc2 = nn.Linear(2 * d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
# TRANSFORMER DECODER
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False) 
        self.attn = MHA(d_model, num_heads, head_dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = PositionwiseMLP(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor,  key_padding_mask=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x
    
# DECODER MODEL
@dataclass
class TransformerConfig:
    vocab_size: int
    context_len: int = 128
    d_model: int = 256
    num_heads: int = 2
    head_dim: int = 32
    num_layers: int = 3
    dropout: float = 0.1
    tie_weights: bool = True


class DecoderTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_emb.weight, mean=0.0,std=0.02)
        self.pos_enc = SCPositionalEncoding(cfg.d_model, max_length=cfg.context_len)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(cfg.d_model,cfg.num_heads,cfg.head_dim, dropout=cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor, attn_mask=None):
        B, T =idx.shape
        assert T <= self.cfg.context_len, "Sequence length exceeds context length."

        x =self.token_emb(idx)
        x =self.pos_enc(x)
        x =self.dropout(x)

        for block in self.blocks:
            x = block(x,key_padding_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size = 5000
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        context_len=128,
        d_model=256,
        num_heads=2,
        head_dim=32,
        num_layers=3,
        dropout=0.1,
        tie_weights=True,
    )

    model = DecoderTransformer(cfg)
    B, T = 4, 64
    x = torch.randint(0, vocab_size, (B, T), dtype=torch.long)
    logits = model(x)
    print("Logits shape:", logits.shape)
    targets = torch.randint(0, vocab_size, (B, T), dtype=torch.long)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    print("Loss:", loss.item())



