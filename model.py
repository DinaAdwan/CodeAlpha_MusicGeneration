"""Character-style LSTM over music tokens."""
from __future__ import annotations

import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len) -> logits: (batch, seq_len, vocab)
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        return self.fc(self.dropout(out)), hidden
