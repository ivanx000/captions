"""
captions.py — Generate .srt subtitle files from video using OpenAI Whisper.

Usage:
    python captions.py <video_path> [--mode sentence|phrase|word]
                                    [--model tiny|base|small|medium|large]
                                    [--language CODE]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

MODEL_NOTE = (
    "Note: Model sizes — "
    "tiny/base: fastest, least accurate | "
    "small: balanced (default) | "
    "medium/large: most accurate, slowest & most VRAM"
)


# ---------------------------------------------------------------------------
# Timestamp formatting
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm (comma, not period)."""
    ms = round((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str:
    """Return the ffmpeg executable path, checking PATH and known install locations."""
    if found := shutil.which("ffmpeg"):
        return found
    # winget installs ffmpeg here but doesn't always refresh the current shell's PATH
    winget_bin = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    for exe in winget_bin.glob("Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe"):
        return str(exe)
    sys.exit(
        "Error: ffmpeg not found. Install it from https://ffmpeg.org/download.html "
        "and make sure it's on your system PATH."
    )


def extract_audio(video_path: str, wav_path: str) -> None:
    """Extract audio from video to a 16kHz mono WAV file using ffmpeg."""
    cmd = [
        _find_ffmpeg(),
        "-i", video_path,
        "-ar", "16000",   # 16kHz sample rate (Whisper's expected rate)
        "-ac", "1",        # mono
        "-vn",             # no video
        "-f", "wav",
        wav_path,
        "-y",              # overwrite without asking
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        sys.exit(f"Error: ffmpeg failed to extract audio:\n{result.stderr.strip()}")


# ---------------------------------------------------------------------------
# Chunking modes
# ---------------------------------------------------------------------------

def chunk_sentence(result: dict) -> list:
    """One subtitle per Whisper segment (natural sentence/clause boundaries)."""
    subtitles = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if text:
            subtitles.append((seg["start"], seg["end"], text))
    return subtitles


def chunk_phrase(result: dict) -> list:
    """
    Group words into short phrases (up to 3 words).
    Breaks on natural pauses (gap > 0.4s between words) or when hitting 3 words.
    Falls back to segment-level chunks if word timestamps are unavailable.
    """
    # Flatten all words across segments
    all_words = []
    for seg in result["segments"]:
        words = seg.get("words")
        if words:
            all_words.extend(words)
        else:
            # Fallback: treat the whole segment as one chunk
            text = seg["text"].strip()
            if text:
                all_words.append({
                    "word": text,
                    "start": seg["start"],
                    "end": seg["end"],
                    "_segment_fallback": True,
                })

    if not all_words:
        return []

    subtitles = []
    chunk_words = []
    chunk_start = None

    for i, word in enumerate(all_words):
        word_text = word["word"].strip()
        if not word_text:
            continue

        if chunk_start is None:
            chunk_start = word["start"]

        chunk_words.append(word_text)

        next_word = all_words[i + 1] if i + 1 < len(all_words) else None

        gap_to_next = (next_word["start"] - word["end"]) if next_word else None
        natural_pause = gap_to_next is not None and gap_to_next > 0.4
        max_words_reached = len(chunk_words) >= 3
        is_last = next_word is None

        if natural_pause or max_words_reached or is_last:
            text = " ".join(chunk_words)
            subtitles.append((chunk_start, word["end"], text))
            chunk_words = []
            chunk_start = None

    return subtitles


def chunk_word(result: dict) -> list:
    """
    Each word as its own subtitle event.
    Enforces a minimum 100ms duration so Vegas doesn't create absurdly short events.
    """
    all_words = []
    for seg in result["segments"]:
        words = seg.get("words")
        if words:
            all_words.extend(words)
        else:
            # Fallback: no word timestamps, skip gracefully
            pass

    if not all_words:
        # No word timestamps at all — fall back to sentence mode with a warning
        print(
            "Warning: Word timestamps unavailable for this audio. "
            "Falling back to sentence mode."
        )
        return chunk_sentence(result)

    subtitles = []
    for i, word in enumerate(all_words):
        text = word["word"].strip()
        if not text:
            continue

        start = word["start"]
        end = word["end"]

        # Don't overlap the next word's start time
        if i + 1 < len(all_words):
            end = min(end, all_words[i + 1]["start"])

        # Enforce minimum 100ms duration
        end = max(end, start + 0.1)

        subtitles.append((start, end, text))

    return subtitles


# ---------------------------------------------------------------------------
# SRT output
# ---------------------------------------------------------------------------

def write_srt(subtitles: list, output_path: str) -> None:
    """Write a list of (start, end, text) tuples to a .srt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for index, (start, end, text) in enumerate(subtitles, start=1):
            text = text.replace(".", "")
            text = text[0].upper() + text[1:].lower() if text else text
            f.write(f"{index}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n")
            f.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate .srt subtitle files from video using OpenAI Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python captions.py video.mp4
  python captions.py video.mp4 --mode phrase
  python captions.py video.mp4 --mode word --model medium
  python captions.py video.mp4 --language ja
        """,
    )
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument(
        "--mode",
        choices=["sentence", "phrase", "word"],
        default="phrase",
        help="Subtitle chunking mode (default: phrase)",
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code, e.g. 'en', 'ja', 'fr' (default: auto-detect)",
    )
    args = parser.parse_args()

    # --- Validate input ---
    video_path = Path(args.video_path).resolve()

    if not video_path.exists():
        sys.exit(f"Error: File not found: {video_path}")

    if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        sys.exit(
            f"Error: Unsupported file format '{video_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    output_path = video_path.with_suffix(".srt")

    # --- Ensure ffmpeg is on PATH for this process (and all subprocesses,
    #     including Whisper's internal audio.py calls) ---
    ffmpeg_exe = _find_ffmpeg()
    ffmpeg_dir = str(Path(ffmpeg_exe).parent)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    # --- Detect device ---
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    print(f"Using device: {device}")
    print(MODEL_NOTE)
    print()

    # --- Extract audio to temp file ---
    tmp_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name

        print("Extracting audio...")
        extract_audio(str(video_path), tmp_wav)

        # --- Load Whisper model ---
        print(f"Loading Whisper model ({args.model})...")
        try:
            import whisper
        except ImportError:
            sys.exit(
                "Error: openai-whisper is not installed. "
                "Run: pip install openai-whisper"
            )

        model = whisper.load_model(args.model, device=device)

        # Word timestamps are needed for phrase and word modes
        need_word_timestamps = args.mode in ("phrase", "word")

        # --- Transcribe ---
        print("Transcribing...")
        transcribe_kwargs = {
            "word_timestamps": need_word_timestamps,
        }
        if args.language:
            transcribe_kwargs["language"] = args.language

        result = model.transcribe(tmp_wav, **transcribe_kwargs)

        # --- Chunk ---
        if args.mode == "sentence":
            subtitles = chunk_sentence(result)
        elif args.mode == "phrase":
            subtitles = chunk_phrase(result)
        else:  # word
            subtitles = chunk_word(result)

        if not subtitles:
            sys.exit("Error: No speech detected in the audio.")

        # --- Write .srt ---
        print("Writing .srt...")
        write_srt(subtitles, str(output_path))

        print(f"\nDone! {len(subtitles)} subtitle events saved to: {output_path}")

    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


if __name__ == "__main__":
    main()
