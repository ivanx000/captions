# captions.py

Generate subtitle files from video using [OpenAI Whisper](https://github.com/openai/whisper) — runs locally, no API key needed.

Outputs `.ass` by default (with position and outline styling pre-applied) or `.srt` via `--format srt`. Designed for use with **Sony Vegas Pro** on 1080×1920 portrait video.

---

## Prerequisites

### 1. Python 3.8+
Download from [python.org](https://www.python.org/downloads/).

### 2. ffmpeg (system install — required)
- **Windows**: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin/` folder to your PATH
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

Verify: `ffmpeg -version`

---

## Installation

```bash
# 1. Clone or copy this folder, then navigate to it
cd captions

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### GPU support (optional but recommended for speed)

The default `requirements.txt` installs CPU-only PyTorch. For CUDA GPU acceleration:

1. Find your CUDA version: `nvidia-smi`
2. Get the right install command from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
3. Run that command **before** `pip install -r requirements.txt`, e.g.:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

The script auto-detects CUDA and prints which device it's using.

---

## Usage

```
python captions.py <video_path> [--mode sentence|phrase|word]
                                [--model tiny|base|small|medium|large]
                                [--language CODE]
                                [--format srt|ass]
                                [--diarize --hf-token TOKEN [--speakers N]]
```

The output file is saved in the **same directory as the input video**, with the same base name.

### Examples

```bash
# Basic — phrase mode, small model, .ass output
python captions.py my_video.mp4

# Output plain .srt instead
python captions.py my_video.mp4 --format srt

# Word-by-word (karaoke / TikTok style)
python captions.py my_video.mp4 --mode word

# Use a more accurate model
python captions.py my_video.mp4 --model medium

# Non-English video
python captions.py my_video.mp4 --language ja

# Speaker diarization (labels each subtitle A:, B:, etc.)
python captions.py my_video.mp4 --diarize --hf-token hf_xxx

# Diarization with a known number of speakers
python captions.py my_video.mp4 --diarize --hf-token hf_xxx --speakers 2

# Combine options
python captions.py my_video.mp4 --mode phrase --model large --language fr
```

---

## Chunking modes

| Mode | Description | Best for |
|------|-------------|----------|
| `phrase` | Up to 3 words per line, breaks at natural pauses **(default)** | Fast speech, TikTok/Shorts style |
| `sentence` | One subtitle per Whisper segment | Dialogue, narration |
| `word` | Each word is its own subtitle event (≥100ms) | Karaoke, word-by-word captions |

---

## Model sizes

| Model | Speed | Accuracy | VRAM |
|-------|-------|----------|------|
| `tiny` | Fastest | Lowest | ~1 GB |
| `base` | Fast | Low | ~1 GB |
| `small` | Balanced **(default)** | Good | ~2 GB |
| `medium` | Slow | Better | ~5 GB |
| `large` | Slowest | Best | ~10 GB |

Models are downloaded automatically on first use and cached in `~/.cache/whisper/`.

---

## Supported input formats

`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`

---

## Output formats

### .ass (default)
Advanced SubStation Alpha format with styling pre-applied — Arial font, white text, black outline, centered near the bottom. Matches the Vegas Pro Subtitles preset for 1080×1920.

To adjust the style, edit the constants near the top of `captions.py`:

```python
ASS_FONT_SIZE    = 72     # increase/decrease text size
ASS_OUTLINE_SIZE = 4      # outline thickness in pixels
ASS_MARGIN_V     = 384    # pixels from the bottom of the frame
```

### .srt
Plain subtitle format with no styling. Use `--format srt` to get this instead.

```
1
00:00:01,000 --> 00:00:03,500
First subtitle line

2
00:00:03,500 --> 00:00:06,000
Second subtitle line
```

---

## Importing into Sony Vegas Pro

### .ass (recommended)
1. Open your project in Vegas Pro
2. Drag the `.ass` file onto the timeline
3. Vegas auto-generates the subtitle track with position and outline already applied — no manual styling needed per event

### .srt
1. Open your project in Vegas Pro
2. Drag the `.srt` file onto the timeline
3. Vegas auto-generates the subtitle track — style each event manually as needed

---

## Speaker diarization

Diarization identifies who is speaking and prefixes each subtitle with a letter label (`A:`, `B:`, etc.).

**Requirements:**
- `pip install pyannote.audio`
- A free [HuggingFace token](https://huggingface.co/settings/tokens)
- Accept the model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

```bash
python captions.py my_video.mp4 --diarize --hf-token hf_xxx
# or set the token as an env var to avoid typing it each time:
# HF_TOKEN=hf_xxx python captions.py my_video.mp4 --diarize
```

---

## Troubleshooting

**`ffmpeg not found`** — The ffmpeg binary is not on your PATH. See Prerequisites above.

**`openai-whisper is not installed`** — Run `pip install openai-whisper` inside your venv.

**Poor transcription accuracy** — Try a larger model (`--model medium` or `--model large`), or specify the language explicitly (`--language en`).

**Word timestamps unavailable in `word` mode** — Falls back to sentence mode automatically with a warning. This is rare but can occur with very short or silent audio.

**`pyannote.audio is not installed`** — Run `pip install pyannote.audio` to enable diarization.
