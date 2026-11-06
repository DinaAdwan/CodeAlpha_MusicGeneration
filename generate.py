"""Sample new music from a trained checkpoint and write a MIDI file.

Usage:
    python generate.py --ckpt checkpoints/music_lstm.pt --length 300 --temperature 0.9
"""
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from music21 import chord, instrument, note, stream

from model import MusicLSTM


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {"hidden": 512, "layers": 2})
    model = MusicLSTM(len(ckpt["vocab"]), hidden_dim=cfg["hidden"], num_layers=cfg["layers"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["vocab"], ckpt["token_to_id"]


@torch.no_grad()
def sample(model, seed_ids, vocab, length=300, temperature=1.0, top_k=0, device="cpu"):
    """Autoregressive sampling with temperature and optional top-k filtering."""
    generated = list(seed_ids)
    hidden = None
    x = torch.tensor([generated], device=device)

    for _ in range(length):
        logits, hidden = model(x, hidden)
        logits = logits[0, -1] / max(temperature, 1e-6)

        if top_k > 0:
            kth = torch.topk(logits, top_k).values[-1]
            logits[logits < kth] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        generated.append(nxt)
        x = torch.tensor([[nxt]], device=device)  # feed only the new token; hidden carries context

    return [vocab[i] for i in generated]


def tokens_to_stream(tokens: list[str]) -> stream.Stream:
    part = stream.Part()
    part.insert(0, instrument.Piano())
    offset = 0.0

    for tok in tokens:
        try:
            kind, body = tok.split(":", 1)
            if kind == "R":
                dur = float(body)
                r = note.Rest(quarterLength=dur)
                part.insert(offset, r)
            else:
                pitches, dur_s = body.rsplit("_", 1)
                dur = float(dur_s)
                if kind == "N":
                    el = note.Note(int(pitches), quarterLength=dur)
                else:  # chord
                    el = chord.Chord([int(p) for p in pitches.split(".")], quarterLength=dur)
                el.volume.velocity = random.randint(70, 100)  # avoid a robotic flat dynamic
                part.insert(offset, el)
            offset += dur
        except Exception:
            continue  # malformed token - skip

    s = stream.Score()
    s.insert(0, part)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/music_lstm.pt")
    ap.add_argument("--data", default="data/dataset.pkl", help="Used to pick a real seed phrase.")
    ap.add_argument("--out", default="output/generated.mid")
    ap.add_argument("--length", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed_len", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, token_to_id = load_model(args.ckpt, device)

    # Seed with a real phrase from the corpus if available, else a random token.
    if Path(args.data).exists():
        with open(args.data, "rb") as fh:
            seqs = pickle.load(fh)["sequences"]
        src = random.choice(seqs)
        start = random.randint(0, max(0, len(src) - args.seed_len - 1))
        seed = src[start : start + args.seed_len]
    else:
        seed = [random.randrange(len(vocab))]

    tokens = sample(
        model, seed, vocab,
        length=args.length, temperature=args.temperature,
        top_k=args.top_k, device=device,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokens_to_stream(tokens).write("midi", fp=args.out)
    print(f"✅ {len(tokens)} tokens → {args.out}  (temperature={args.temperature})")


if __name__ == "__main__":
    main()
