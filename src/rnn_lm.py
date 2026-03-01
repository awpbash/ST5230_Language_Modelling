# Vanilla RNN language model

import torch
import torch.nn as nn


class RNNLM(nn.Module):
    # vanilla RNN language model for next-token prediction

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.1, embedding_layer=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if embedding_layer is not None:
            self.embedding = embedding_layer
            actual_embed_dim = embedding_layer.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            actual_embed_dim = embed_dim

        # project if embedding dim doesn't match expected input dim
        self.embed_proj = None
        if actual_embed_dim != embed_dim:
            self.embed_proj = nn.Linear(actual_embed_dim, embed_dim)

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (B, T) token indices -> logits: (B, T, vocab_size)
        emb = self.embedding(x)  # (B, T, E)
        if self.embed_proj is not None:
            emb = self.embed_proj(emb)
        emb = self.dropout(emb)

        output, _ = self.rnn(emb, hidden)  # (B, T, H)
        output = self.dropout(output)
        logits = self.fc(output)  # (B, T, V)
        return logits

    def get_hidden_states(self, x):
        # extract hidden states for downstream tasks, returns (B, T, H)
        emb = self.embedding(x)
        if self.embed_proj is not None:
            emb = self.embed_proj(emb)
        output, _ = self.rnn(emb)
        return output
