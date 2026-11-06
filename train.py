"""Train the LSTM on the prepared token dataset.

Usage:
    python train.py --data data/dataset.pkl --epochs 50 --seq_len 64
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import MusicLSTM


class NoteDataset(Dataset):
    """Sliding windows: input = tokens[i:i+L], target = tokens[i+1:i+L+1]."""

    def __init__(self, sequences: list[list[int]], seq_len: int, stride: int = 1):
        self.seq_len = seq_len
        self.windows: list[list[int]] = []
        for seq in sequences:
            for i in range(0, len(seq) - seq_len - 1, stride):
                self.windows.append(seq[i : i + seq_len + 1])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        return torch.tensor(w[:-1]), torch.tensor(w[1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.pkl")
    ap.add_argument("--out", default="checkpoints/music_lstm.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.data, "rb") as fh:
        blob = pickle.load(fh)
    vocab = blob["vocab"]

    dataset = NoteDataset(blob["sequences"], args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Vocab {len(vocab)} · {len(dataset):,} training windows")

    model = MusicLSTM(len(vocab), hidden_dim=args.hidden, num_layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, started = 0.0, time.time()

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, len(vocab)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # LSTMs love to explode
            optimizer.step()
            total += loss.item()

        avg = total / len(loader)
        scheduler.step(avg)
        ppl = torch.exp(torch.tensor(avg)).item()
        print(f"Epoch {epoch:3d}/{args.epochs} · loss {avg:.4f} · ppl {ppl:7.2f} · {time.time()-started:.1f}s")

        if avg < best:
            best = avg
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "token_to_id": blob["token_to_id"],
                    "config": {"hidden": args.hidden, "layers": args.layers},
                    "loss": avg,
                },
                args.out,
            )
            print(f"   ↳ saved (best) → {args.out}")


if __name__ == "__main__":
    main()
