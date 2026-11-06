# CodeAlpha_MusicGeneration

**CodeAlpha AI Internship — Task 3**

An LSTM that learns note sequences from MIDI files and composes new music.

## Pipeline
```
MIDI files → music21 parsing → token sequence → LSTM training → sampling → MIDI out
```

### Tokenisation
Each musical event becomes one token:

| Token | Meaning |
|---|---|
| `N:60_0.5` | Note, MIDI pitch 60, half a quarter-note long |
| `C:60.64.67_1.0` | Chord (C-E-G), one quarter-note long |
| `R:1.0` | Rest |

Durations are quantized to a 16th-note grid to keep the vocabulary compact.

## Model
2-layer LSTM (512 hidden units) with an embedding layer, dropout 0.3, and a linear
head over the vocabulary. Trained with cross-entropy on next-token prediction and
gradient clipping at 5.0.

## Getting MIDI data
Drop `.mid` files into `data/midi/`. Good free sources:
- **Classical Piano MIDI** — `piano-midi.de`
- **MAESTRO** (Google Magenta) — ~200 hours of virtuoso piano
- **Lakh MIDI Dataset** — 176k files, multi-genre
- Kaggle: search "classical music midi"

50–150 files of one genre is plenty for a convincing demo.

## Run
```bash
pip install -r requirements.txt

python prepare_data.py --midi_dir data/midi --out data/dataset.pkl
python train.py --epochs 50 --seq_len 64
python generate.py --temperature 0.9 --length 300

streamlit run app.py     # interactive UI
```

## Rendering to audio
The output is a `.mid`. To hear it as a WAV:
```bash
# macOS/Linux
fluidsynth -ni FluidR3_GM.sf2 output/generated.mid -F output/generated.wav -r 44100
```
Or just open the MIDI in MuseScore / GarageBand / any DAW.

## Tuning the output
| Temperature | Result |
|---|---|
| 0.5–0.7 | Conservative, repetitive, very "in key" |
| 0.8–1.0 | **Sweet spot** — coherent but creative |
| 1.2+ | Experimental, often dissonant |

Top-k filtering (k ≈ 20) is a good way to keep temperature high while blocking
truly implausible notes.
