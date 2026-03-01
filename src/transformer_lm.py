# Decoder-only (GPT-style) Transformer language model

import math
import torch
import torch.nn as nn


class TransformerLM(nn.Module):
    # causal (decoder-only) transformer language model

    def __init__(self, vocab_size, embed_dim=128, n_heads=4, d_ff=512,
                 num_layers=2, max_seq_len=128, dropout=0.1, embedding_layer=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        if embedding_layer is not None:
            self.embedding = embedding_layer
            actual_embed_dim = embedding_layer.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            actual_embed_dim = embed_dim

        self.embed_proj = None
        if actual_embed_dim != embed_dim:
            self.embed_proj = nn.Linear(actual_embed_dim, embed_dim)

        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # using TransformerEncoderLayer since this is decoder-only (self-attention only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def _make_causal_mask(self, seq_len, device):
        # upper triangular float('-inf') mask — version-safe across PyTorch 1.x and 2.x
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, x):
        # x: (B, T) token indices -> logits: (B, T, vocab_size)
        B, T = x.shape
        device = x.device

        # token + positional embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        emb = self.embedding(x)  # (B, T, E)
        if self.embed_proj is not None:
            emb = self.embed_proj(emb)
        emb = emb + self.pos_embedding(positions)
        emb = self.dropout(emb)

        # causal mask
        causal_mask = self._make_causal_mask(T, device)

        # padding mask: True where padded (pad_idx=0)
        padding_mask = (x == 0)  # (B, T)

        output = self.transformer(
            emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )  # (B, T, E)

        output = self.ln_f(output)
        logits = self.fc(output)  # (B, T, V)
        return logits

    def get_hidden_states(self, x):
        # extract hidden states for downstream tasks, returns (B, T, embed_dim)
        B, T = x.shape
        device = x.device

        positions = torch.arange(T, device=device).unsqueeze(0)
        emb = self.embedding(x)
        if self.embed_proj is not None:
            emb = self.embed_proj(emb)
        emb = emb + self.pos_embedding(positions)

        causal_mask = self._make_causal_mask(T, device)
        padding_mask = (x == 0)

        output = self.transformer(
            emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        output = self.ln_f(output)
        return output
