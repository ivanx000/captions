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
# ASS style constants (mirrors Vegas "Subtitles" preset on 1080x1920)
# ---------------------------------------------------------------------------

ASS_PLAY_RES_X   = 1080
ASS_PLAY_RES_Y   = 1920
ASS_FONT         = "Arial"
ASS_FONT_SIZE    = 72        # adjust if text appears too big/small in Vegas
ASS_TEXT_COLOR   = "&H00FFFFFF"   # white
ASS_OUTLINE_COLOR = "&H00000000"  # black
ASS_OUTLINE_SIZE = 4         # pixels; maps to Vegas outline width 10.000
ASS_MARGIN_V     = 384       # pixels from bottom (Vegas Y=0.20 on 1920px frame)


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


def format_ass_timestamp(seconds: float) -> str:
    """Convert seconds to ASS timestamp format H:MM:SS.cc (centiseconds)."""
    cs = round((seconds % 1) * 100)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


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
# Speaker diarization
# ---------------------------------------------------------------------------

def diarize_audio(audio_path: str, hf_token: str, num_speakers: int = None) -> list:
    """Run pyannote speaker diarization; returns [(start, end, speaker_label), ...]."""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        sys.exit(
            "Error: pyannote.audio is not installed. "
            "Run: pip install pyannote.audio"
        )
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = None

    print("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    if device is not None:
        pipeline = pipeline.to(device)

    print("Running speaker diarization...")
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    annotation = pipeline(audio_path, **kwargs)

    return [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def get_speaker_at(diarization: list, time: float) -> str:
    """Return the speaker label active at `time`, or None if no segment covers it."""
    for start, end, speaker in diarization:
        if start <= time <= end:
            return speaker
    return None


# ---------------------------------------------------------------------------
# Chunking modes
# ---------------------------------------------------------------------------

def chunk_sentence(result: dict) -> list:
    """One subtitle per Whisper segment (natural sentence/clause boundaries)."""
    subtitles = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if text:
            subtitles.append((seg["start"], seg["end"], text, None))
    return subtitles


def chunk_phrase(result: dict, diarization: list = None) -> list:
    """
    Group words into short phrases (up to 3 words).
    Breaks on natural pauses (>0.4s), sentence-ending punctuation, speaker changes,
    or when hitting 3 words. Falls back to segment-level chunks if word timestamps
    are unavailable.
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
    chunk_speaker = None

    for i, word in enumerate(all_words):
        word_text = word["word"].strip()
        if not word_text:
            continue

        word_start = word["start"]
        word_speaker = get_speaker_at(diarization, word_start) if diarization else None

        # Force a break when the speaker changes mid-stream
        if chunk_words and diarization and word_speaker != chunk_speaker:
            text = " ".join(chunk_words)
            subtitles.append((chunk_start, all_words[i - 1]["end"], text, chunk_speaker))
            chunk_words = []
            chunk_start = None
            chunk_speaker = None

        if chunk_start is None:
            chunk_start = word_start
            chunk_speaker = word_speaker

        chunk_words.append(word_text)

        next_word = all_words[i + 1] if i + 1 < len(all_words) else None

        gap_to_next = (next_word["start"] - word["end"]) if next_word else None
        natural_pause = gap_to_next is not None and gap_to_next > 0.4
        max_words_reached = len(chunk_words) >= 3
        sentence_end = word_text.rstrip().endswith((".", "?", "!"))
        is_last = next_word is None

        if natural_pause or max_words_reached or sentence_end or is_last:
            text = " ".join(chunk_words)
            subtitles.append((chunk_start, word["end"], text, chunk_speaker))
            chunk_words = []
            chunk_start = None
            chunk_speaker = None

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

        subtitles.append((start, end, text, None))

    return subtitles


# ---------------------------------------------------------------------------
# SRT / ASS output
# ---------------------------------------------------------------------------

def write_ass(subtitles: list, output_path: str) -> None:
    """Write subtitles to an .ass file with styling matching the Vegas Subtitles preset."""
    raw_speakers = [s[3] for s in subtitles if s[3] is not None]
    unique_speakers = sorted(set(raw_speakers))
    speaker_map = {sp: chr(ord("A") + i) for i, sp in enumerate(unique_speakers)}
    use_labels = bool(unique_speakers)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {ASS_PLAY_RES_X}\n"
        f"PlayResY: {ASS_PLAY_RES_Y}\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{ASS_FONT},{ASS_FONT_SIZE},"
        f"{ASS_TEXT_COLOR},&H000000FF,{ASS_OUTLINE_COLOR},&H00000000,"
        f"0,0,0,0,100,100,0,0,1,{ASS_OUTLINE_SIZE},0,"
        f"2,0,0,{ASS_MARGIN_V},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for start, end, text, speaker in subtitles:
            text = text.replace(".", "")
            text = text[0].upper() + text[1:].lower() if text else text
            if use_labels:
                label = speaker_map.get(speaker, "?")
                text = f"{label}: {text}"
            f.write(
                f"Dialogue: 0,{format_ass_timestamp(start)},{format_ass_timestamp(end)},"
                f"Default,,0,0,0,,{text}\n"
            )


def write_srt(subtitles: list, output_path: str) -> None:
    """Write a list of (start, end, text, speaker) tuples to a .srt file."""
    # Build a short letter label for each unique speaker (SPEAKER_00 -> A, etc.)
    raw_speakers = [s[3] for s in subtitles if s[3] is not None]
    unique_speakers = sorted(set(raw_speakers))
    speaker_map = {sp: chr(ord("A") + i) for i, sp in enumerate(unique_speakers)}
    use_labels = bool(unique_speakers)

    with open(output_path, "w", encoding="utf-8") as f:
        for index, (start, end, text, speaker) in enumerate(subtitles, start=1):
            text = text.replace(".", "")
            text = text[0].upper() + text[1:].lower() if text else text
            if use_labels:
                label = speaker_map.get(speaker, "?")
                text = f"{label}: {text}"
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
  python captions.py video.mp4 --diarize --hf-token hf_xxx
  python captions.py video.mp4 --diarize --hf-token hf_xxx --speakers 2
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
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (requires pyannote.audio and a HuggingFace token)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for pyannote models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Number of speakers (optional hint for diarization accuracy)",
    )
    parser.add_argument(
        "--format",
        choices=["srt", "ass"],
        default="ass",
        help="Output subtitle format (default: ass)",
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

    output_path = video_path.with_suffix(f".{args.format}")

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

        # --- Speaker diarization (optional) ---
        diarization = None
        if args.diarize:
            hf_token = args.hf_token or os.environ.get("HF_TOKEN")
            if not hf_token:
                sys.exit(
                    "Error: --diarize requires a HuggingFace token. "
                    "Pass --hf-token <token> or set the HF_TOKEN environment variable.\n"
                    "Get a free token at https://huggingface.co/settings/tokens"
                )
            diarization = diarize_audio(tmp_wav, hf_token, args.speakers)

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
            subtitles = chunk_phrase(result, diarization)
        else:  # word
            subtitles = chunk_word(result)

        if not subtitles:
            sys.exit("Error: No speech detected in the audio.")

        # --- Write output ---
        print(f"Writing .{args.format}...")
        if args.format == "ass":
            write_ass(subtitles, str(output_path))
        else:
            write_srt(subtitles, str(output_path))

        print(f"\nDone! {len(subtitles)} subtitle events saved to: {output_path}")

    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


if __name__ == "__main__":
    main()
