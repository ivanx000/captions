# captions.py

Generate `.srt` subtitle files from video files using [OpenAI Whisper](https://github.com/openai/whisper) — runs locally, no API key needed.

Designed for use with **Sony Vegas Pro 18**, which imports `.srt` files and converts each subtitle line into an individually editable text event on the timeline.

---

## Prerequisites

### 1. Python 3.8+
Download from [python.org](https://www.python.org/downloads/).

### 2. ffmpeg (system install — required)
The Python `ffmpeg-python` package is just bindings. You **must** also have the ffmpeg binary on your system PATH.

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin/` folder to your PATH. Or use winget: `winget install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg` (Debian/Ubuntu) or equivalent

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
python captions.py <video_path> [--mode sentence|phrase|word] [--model tiny|base|small|medium|large] [--language CODE]
```

The output `.srt` file is saved in the **same directory as the input video**, with the same base name.

### Examples

```bash
# Basic — sentence mode, small model, auto-detect language
python captions.py my_video.mp4

# Shorter caption chunks (~3-7 words), good for fast speech
python captions.py my_video.mp4 --mode phrase

# Word-by-word (karaoke / TikTok style)
python captions.py my_video.mp4 --mode word

# Use a more accurate model
python captions.py my_video.mp4 --model medium

# Non-English video
python captions.py my_video.mp4 --language ja

# Combine options
python captions.py my_video.mp4 --mode phrase --model large --language fr
```

---

## Chunking modes

| Mode | Description | Best for |
|------|-------------|----------|
| `sentence` | One subtitle per Whisper segment (default) | Dialogue, narration |
| `phrase` | ~3-7 words per line, breaks at natural pauses | Fast speech, dense content |
| `word` | Each word is its own subtitle event (≥100ms) | Karaoke, TikTok-style captions |

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

## Output

Standard `.srt` format with comma-separated milliseconds (required by the spec and by Vegas):

```
1
00:00:01,000 --> 00:00:03,500
First subtitle line here

2
00:00:03,500 --> 00:00:06,000
Second subtitle line here
```

---

## Importing into Sony Vegas Pro 18

1. Open your project in Vegas Pro
2. Go to **File > Import > Subtitles**  
   *(or drag the `.srt` file onto the timeline)*
3. Each subtitle line becomes an individual text event — fully editable

---

## Troubleshooting

**`ffmpeg not found`** — The ffmpeg binary is not on your PATH. See Prerequisites above.

**`openai-whisper is not installed`** — Run `pip install openai-whisper` inside your venv.

**Poor transcription accuracy** — Try a larger model (`--model medium` or `--model large`), or specify the language explicitly (`--language en`).

**Word timestamps unavailable in `word` mode** — Falls back to sentence mode automatically with a warning. This is rare but can occur with very short or silent audio.
