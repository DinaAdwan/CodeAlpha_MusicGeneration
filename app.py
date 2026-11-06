"""CodeAlpha Task 3 — Music Generation (Streamlit UI)."""
from __future__ import annotations

import pickle
import random
import tempfile
from pathlib import Path

import streamlit as st
import torch

from generate import load_model, sample, tokens_to_stream

st.set_page_config(page_title="AI Music Generation", page_icon="🎵", layout="centered")
st.title("🎵 AI Music Generation")
st.caption("CodeAlpha AI Internship — Task 3 · LSTM trained on MIDI note sequences")

CKPT = Path("checkpoints/music_lstm.pt")
DATA = Path("data/dataset.pkl")

if not CKPT.exists():
    st.error(
        "No trained checkpoint found.\n\n"
        "Run:\n```\npython prepare_data.py --midi_dir data/midi\npython train.py --epochs 50\n```"
    )
    st.stop()


@st.cache_resource
def _load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, _ = load_model(str(CKPT), device)
    seqs = pickle.load(open(DATA, "rb"))["sequences"] if DATA.exists() else []
    return model, vocab, seqs, device


model, vocab, seqs, device = _load()
st.success(f"Model loaded · vocabulary: {len(vocab)} tokens")

col1, col2 = st.columns(2)
length = col1.slider("Length (notes)", 50, 800, 300, 50)
temperature = col2.slider(
    "Temperature", 0.4, 1.6, 0.9, 0.1,
    help="Low = safe and repetitive. High = adventurous and chaotic.",
)
top_k = st.slider("Top-k filtering (0 = off)", 0, 50, 0)

if st.button("🎼 Generate", type="primary", use_container_width=True):
    with st.spinner("Composing…"):
        if seqs:
            src = random.choice(seqs)
            start = random.randint(0, max(0, len(src) - 33))
            seed = src[start : start + 32]
        else:
            seed = [random.randrange(len(vocab))]

        tokens = sample(model, seed, vocab, length=length,
                        temperature=temperature, top_k=top_k, device=device)

        out = Path(tempfile.gettempdir()) / "generated.mid"
        tokens_to_stream(tokens).write("midi", fp=str(out))
        midi_bytes = out.read_bytes()

    st.download_button(
        "⬇️ Download MIDI", midi_bytes, file_name="generated.mid",
        mime="audio/midi", use_container_width=True,
    )
    with st.expander("Generated token sequence"):
        st.code(" ".join(tokens[:120]) + (" …" if len(tokens) > 120 else ""))
    st.info("Open the .mid in any DAW / MuseScore, or render to WAV with FluidSynth (see README).")
