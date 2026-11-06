"""Parse a folder of MIDI files into a token sequence + vocabulary.

Token format:
    N:60_0.5      -> single note (MIDI pitch 60, quarterLength 0.5)
    C:60.64.67_1.0 -> chord (pitches joined by '.')
    R:1.0         -> rest

Usage:
    python prepare_data.py --midi_dir data/midi --out data/dataset.pkl
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

from music21 import chord, converter, instrument, note


def quantize(length: float) -> float:
    """Snap durations to a 16th-note grid so the vocabulary stays small."""
    q = round(float(length) * 4) / 4
    return max(q, 0.25)


def midi_to_tokens(path: Path) -> list[str]:
    score = converter.parse(path)
    try:  # prefer a single instrument part (usually piano)
        parts = instrument.partitionByInstrument(score)
        elements = parts.parts[0].recurse() if parts else score.flatten().notesAndRests
    except Exception:
        elements = score.flatten().notesAndRests

    tokens: list[str] = []
    for el in elements:
        dur = quantize(el.duration.quarterLength)
        if isinstance(el, note.Note):
            tokens.append(f"N:{el.pitch.midi}_{dur}")
        elif isinstance(el, chord.Chord):
            pitches = ".".join(str(p.midi) for p in sorted(el.pitches))
            tokens.append(f"C:{pitches}_{dur}")
        elif isinstance(el, note.Rest) and dur >= 0.5:
            tokens.append(f"R:{dur}")
    return tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", default="data/midi")
    ap.add_argument("--out", default="data/dataset.pkl")
    ap.add_argument("--min_freq", type=int, default=2,
                    help="Drop tokens rarer than this (mapped to the closest common one).")
    args = ap.parse_args()

    files = sorted(Path(args.midi_dir).rglob("*.mid")) + sorted(Path(args.midi_dir).rglob("*.midi"))
    if not files:
        raise SystemExit(f"No MIDI files found in {args.midi_dir!r}. See README for datasets.")

    sequences: list[list[str]] = []
    for i, f in enumerate(files, 1):
        try:
            toks = midi_to_tokens(f)
            if len(toks) > 50:
                sequences.append(toks)
            print(f"[{i}/{len(files)}] {f.name}: {len(toks)} tokens")
        except Exception as exc:
            print(f"[{i}/{len(files)}] SKIP {f.name} ({exc})")

    counts = Counter(t for seq in sequences for t in seq)
    vocab = sorted(t for t, c in counts.items() if c >= args.min_freq)
    token_to_id = {t: i for i, t in enumerate(vocab)}

    # Encode, silently dropping rare tokens.
    encoded = [[token_to_id[t] for t in seq if t in token_to_id] for seq in sequences]
    encoded = [s for s in encoded if len(s) > 50]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"sequences": encoded, "vocab": vocab, "token_to_id": token_to_id}, fh)

    total = sum(len(s) for s in encoded)
    print(f"\n✅ {len(encoded)} sequences · {total:,} tokens · vocab {len(vocab)} → {args.out}")


if __name__ == "__main__":
    main()
