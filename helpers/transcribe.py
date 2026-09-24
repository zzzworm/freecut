"""Transcribe a video with a pluggable ASR backend.

Extracts mono 16kHz audio via ffmpeg, runs the selected backend, then writes
a normalized JSON to <edit_dir>/transcripts/<video_stem>.json.

The output shape is fixed and consumed downstream by pack_transcripts.py:

    {
      "words": [
        {"type": "word"|"spacing"|"audio_event",
         "text": str,
         "start": float (sec),
         "end":   float (sec),
         "speaker_id": "speaker_0" | "speaker_1" | ... }
      ]
    }

Backends:
    whisper     — DEFAULT. Local, free. Uses mlx-whisper on Apple Silicon if
                  available, otherwise faster-whisper. Single speaker (no
                  diarization) — every word gets speaker_id "speaker_0".
    vibevoice   — Calls an HTTP endpoint (env VIBEVOICE_ASR_URL) that serves
                  the microsoft/VibeVoice-ASR model. Multi-speaker diarization.
    funasr      — Calls a FunASR OpenAI-compatible endpoint (env FUN_ASR_URL,
                  e.g. http://192.168.50.79:5078) via
                  POST /v1/audio/transcriptions. Multi-speaker diarization
                  (CAM++), word-level timestamps via the forced aligner.
    elevenlabs  — Original ElevenLabs Scribe path. Needs ELEVENLABS_API_KEY.

Cached: if the output file already exists, transcription is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --backend whisper --model small
    python helpers/transcribe.py <video_path> --backend vibevoice
    python helpers/transcribe.py <video_path> --backend funasr
    python helpers/transcribe.py <video_path> --backend elevenlabs --num-speakers 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_BACKEND = "whisper"
DEFAULT_WHISPER_MODEL = "small"


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def _read_env_file() -> dict[str, str]:
    out: dict[str, str] = {}
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _env(name: str) -> str:
    return os.environ.get(name, "") or _read_env_file().get(name, "")


def load_api_key() -> str:
    """Return the ElevenLabs API key. Kept for import back-compat with
    transcribe_batch.py; only used by the elevenlabs backend."""
    v = _env("ELEVENLABS_API_KEY")
    if not v:
        sys.exit("ELEVENLABS_API_KEY not found in .env or environment")
    return v


# ---------------------------------------------------------------------------
# ffmpeg audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Shared helpers for the normalized shape
# ---------------------------------------------------------------------------

def _interleave_spacing(word_entries: list[dict]) -> list[dict]:
    """Insert 'spacing' entries between consecutive words so that
    pack_transcripts.py can detect silence gaps. Words are assumed sorted."""
    out: list[dict] = []
    prev_end: float | None = None
    for w in word_entries:
        start = float(w["start"])
        if prev_end is not None and start > prev_end:
            out.append({
                "type": "spacing",
                "text": " ",
                "start": prev_end,
                "end": start,
            })
        out.append(w)
        prev_end = float(w["end"])
    return out


# ---------------------------------------------------------------------------
# Backend: whisper (local, free, single-speaker)
# ---------------------------------------------------------------------------

def _mlx_whisper_available() -> bool:
    try:
        import mlx_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def transcribe_whisper(
    audio_path: Path,
    model: str = DEFAULT_WHISPER_MODEL,
    language: str | None = None,
) -> dict:
    """Local Whisper transcription with word-level timestamps.

    Prefers mlx-whisper on Apple Silicon; falls back to faster-whisper.
    Returns the normalized {"words": [...]} shape with speaker_id="speaker_0".
    """
    word_entries: list[dict] = []

    if _mlx_whisper_available():
        import mlx_whisper
        # mlx-whisper uses HuggingFace repo IDs. Map common short names.
        repo_map = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large": "mlx-community/whisper-large-v3-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        }
        path_or_repo = repo_map.get(model, model)
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=path_or_repo,
            language=language,
            word_timestamps=True,
        )
        for seg in result.get("segments", []):
            for w in seg.get("words", []) or []:
                text = (w.get("word") or "").strip()
                if not text:
                    continue
                word_entries.append({
                    "type": "word",
                    "text": text,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "speaker_id": "speaker_0",
                })

    elif _faster_whisper_available():
        from faster_whisper import WhisperModel
        wm = WhisperModel(model, device="auto", compute_type="auto")
        segments, _info = wm.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=False,
        )
        for seg in segments:
            for w in (seg.words or []):
                text = (w.word or "").strip()
                if not text:
                    continue
                word_entries.append({
                    "type": "word",
                    "text": text,
                    "start": float(w.start),
                    "end": float(w.end),
                    "speaker_id": "speaker_0",
                })

    else:
        sys.exit(
            "whisper backend needs one of:\n"
            "  pip install mlx-whisper       # Apple Silicon (recommended)\n"
            "  pip install faster-whisper    # NVIDIA / CPU fallback"
        )

    return {"words": _interleave_spacing(word_entries)}


# ---------------------------------------------------------------------------
# Shared segment → words conversion for HTTP ASR backends
# ---------------------------------------------------------------------------

def _first(d: dict, *keys: str, default: float = 0.0) -> float:
    """Return the first non-None value among `keys` as float, else `default`.

    Lets us accept both 'start'/'end' and 'start_time'/'end_time' conventions
    even when a key is present-but-null.
    """
    for k in keys:
        v = d.get(k)
        if v is not None:
            return float(v)
    return default


class _SpeakerMap:
    """Stable mapping from arbitrary speaker labels → speaker_0, speaker_1, ...

    Backends use different label conventions ("SPEAKER_00", "Speaker1",
    "说话人1"); downstream code only understands speaker_N.
    """

    def __init__(self) -> None:
        self._ids: dict[str, str] = {}

    def __call__(self, label: object) -> str:
        key = str(label) if label is not None else "0"
        if key not in self._ids:
            self._ids[key] = f"speaker_{len(self._ids)}"
        return self._ids[key]


def _seg_speaker(seg: dict) -> str:
    return (
        seg.get("speaker")
        or seg.get("speaker_id")
        or seg.get("who")
        or "0"
    )


def _segments_to_words(segments: list) -> list[dict]:
    """Convert HTTP-backend segment JSON to the normalized words[] shape.

    Accepts both key conventions seen in the wild:
      - VibeVoice: start/end, words[{text,start,end}], speaker|speaker_id|who
      - FunASR:    start_time/end_time, word_tokens[{text,start_time,end_time}],
                   speaker_id ("Speaker1", ...)

    Speaker labels are stably remapped to speaker_0, speaker_1, ...
    If per-word timestamps are absent, we linearly interpolate across the
    segment text so pack_transcripts still gets word-level entries.
    """
    sid = _SpeakerMap()

    word_entries: list[dict] = []
    for seg in segments:
        speaker_id = sid(_seg_speaker(seg))
        seg_start = _first(seg, "start", "start_time")
        seg_end = _first(seg, "end", "end_time", default=seg_start)
        seg_words = seg.get("words") or seg.get("word_tokens") or []

        if seg_words:
            for w in seg_words:
                text = (w.get("text") or w.get("word") or "").strip()
                if not text:
                    continue
                word_entries.append({
                    "type": "word",
                    "text": text,
                    "start": _first(w, "start", "start_time", default=seg_start),
                    "end": _first(w, "end", "end_time", default=seg_end),
                    "speaker_id": speaker_id,
                })
        else:
            # Fall back to linear interpolation across the segment text.
            tokens = (seg.get("text") or "").split()
            if not tokens:
                continue
            span = max(seg_end - seg_start, 1e-3)
            per = span / len(tokens)
            for i, tok in enumerate(tokens):
                word_entries.append({
                    "type": "word",
                    "text": tok,
                    "start": seg_start + i * per,
                    "end": seg_start + (i + 1) * per,
                    "speaker_id": speaker_id,
                })

    word_entries.sort(key=lambda w: w["start"])
    return word_entries


def _openai_words_to_normalized(words: list, segments: list) -> dict:
    """Convert an OpenAI-style top-level word list to the normalized shape.

    FunASR's verbose_json puts word timestamps in a top-level
    "words": [{"word": "Hello", "start": 0.0, "end": 0.48}, ...] array while
    "segments" only carry speaker labels and time spans. Words get their
    speaker attributed from the segment whose time range contains them
    (nearest segment if a word falls in a gap).
    """
    sid = _SpeakerMap()

    def speaker_at(t: float) -> str:
        best_label: object = None
        best_dist: float | None = None
        for seg in segments:
            s = _first(seg, "start", "start_time")
            e = _first(seg, "end", "end_time", default=s)
            if s <= t <= e:
                return sid(_seg_speaker(seg))
            d = min(abs(t - s), abs(t - e))
            if best_dist is None or d < best_dist:
                best_dist, best_label = d, _seg_speaker(seg)
        return sid(best_label)

    word_entries: list[dict] = []
    for w in words:
        if not isinstance(w, dict):
            continue
        text = (w.get("word") or w.get("text") or "").strip()
        if not text:
            continue
        start = _first(w, "start", "start_time")
        end = _first(w, "end", "end_time", default=start)
        word_entries.append({
            "type": "word",
            "text": text,
            "start": start,
            "end": end,
            "speaker_id": speaker_at((start + end) / 2),
        })

    word_entries.sort(key=lambda w: w["start"])
    return {"words": _interleave_spacing(word_entries)}


def _payload_to_normalized(payload: object) -> dict:
    """Normalize an HTTP-backend response to {"words": [...]}.

    Handles three shapes, in order:
      1. Already normalized {"words": [{type,text,start,end,speaker_id}]}.
      2. Top-level word-timestamp list + speaker-bearing segments
         (FunASR verbose_json with word_timestamps=true).
      3. Segments with or without per-word entries (VibeVoice / FunASR docs).
    """
    if not isinstance(payload, dict):
        return {"words": []}

    words = payload.get("words")
    segments = payload.get("segments") or payload.get("results") or []

    if isinstance(words, list) and words and isinstance(words[0], dict):
        first = words[0]
        if all(k in first for k in ("type", "text", "start", "end", "speaker_id")):
            return payload  # already normalized
        # OpenAI-style word list: {"word"|"text", "start", "end"} entries.
        if any(k in first for k in ("word", "text")) and "start" in first and "end" in first:
            return _openai_words_to_normalized(words, segments)

    return {"words": _interleave_spacing(_segments_to_words(segments))}


# ---------------------------------------------------------------------------
# Backend: vibevoice (HTTP endpoint, multi-speaker diarization)
# ---------------------------------------------------------------------------

def transcribe_vibevoice(
    audio_path: Path,
    url: str | None = None,
    language: str | None = None,
) -> dict:
    """Call a VibeVoice-ASR HTTP endpoint. The endpoint is expected to accept
    a multipart upload of a wav file and return either

        {"words": [...]}                                 # already normalized
    or  {"segments": [{"speaker": "SPEAKER_0",           # Who/When/What
                       "start": 0.12, "end": 3.4,
                       "text": "hello world",
                       "words": [{"text":"hello","start":0.12,"end":0.42}, ...]}]}
    """
    endpoint = url or _env("VIBEVOICE_ASR_URL")
    if not endpoint:
        sys.exit(
            "vibevoice backend requires VIBEVOICE_ASR_URL (env or .env).\n"
            "VibeVoice-ASR is CUDA-only; point this at a rented GPU box, "
            "Modal/RunPod deploy, or an Azure AI Foundry endpoint."
        )

    data: dict[str, str] = {}
    if language:
        data["language"] = language

    with open(audio_path, "rb") as f:
        resp = requests.post(
            endpoint,
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            timeout=1800,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"vibevoice endpoint returned {resp.status_code}: {resp.text[:500]}")
    return _payload_to_normalized(resp.json())


# ---------------------------------------------------------------------------
# Backend: funasr (OpenAI-compatible HTTP endpoint, multi-speaker diarization)
# ---------------------------------------------------------------------------

def _funasr_endpoint(base_url: str) -> str:
    """Build the /v1/audio/transcriptions URL from a base URL.

    Accepts a bare host (http://host:port), a /v1 base, or the full
    transcriptions path — all normalize to the same endpoint.
    """
    url = base_url.rstrip("/")
    if url.endswith("/v1/audio/transcriptions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/audio/transcriptions"
    return f"{url}/v1/audio/transcriptions"


def transcribe_funasr(
    audio_path: Path,
    url: str | None = None,
    language: str | None = None,
    word_timestamps: bool = True,
) -> dict:
    """Call a FunASR OpenAI-compatible endpoint (POST /v1/audio/transcriptions).

    Speaker diarization (CAM++) is on by default. Word-level timestamps are
    requested via the forced aligner; if the backend returns none, the shared
    segment converter interpolates across the segment text.
    """
    base = url or _env("FUN_ASR_URL")
    if not base:
        sys.exit(
            "funasr backend requires FUN_ASR_URL (env or .env).\n"
            "Point it at your FunASR server base URL, e.g. "
            "FUN_ASR_URL=http://192.168.50.79:5078"
        )
    endpoint = _funasr_endpoint(base)

    data: dict[str, str] = {
        "response_format": "verbose_json",
        "enable_speaker_diarization": "true",
        "word_timestamps": "true" if word_timestamps else "false",
    }
    if language:
        data["language"] = language

    with open(audio_path, "rb") as f:
        resp = requests.post(
            endpoint,
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            timeout=1800,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"funasr endpoint returned {resp.status_code}: {resp.text[:500]}")
    return _payload_to_normalized(resp.json())


# ---------------------------------------------------------------------------
# Backend: elevenlabs (Scribe — original path, still supported)
# ---------------------------------------------------------------------------

def call_scribe(
    audio_path: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
) -> dict:
    data: dict[str, str] = {
        "model_id": "scribe_v1",
        "diarize": "true",
        "tag_audio_events": "true",
        "timestamps_granularity": "word",
    }
    if language:
        data["language_code"] = language
    if num_speakers:
        data["num_speakers"] = str(num_speakers)

    with open(audio_path, "rb") as f:
        resp = requests.post(
            SCRIBE_URL,
            headers={"xi-api-key": api_key},
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            timeout=1800,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Scribe returned {resp.status_code}: {resp.text[:500]}")
    return resp.json()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _run_backend(
    backend: str,
    audio: Path,
    language: str | None,
    num_speakers: int | None,
    whisper_model: str,
    vibevoice_url: str | None,
    funasr_url: str | None,
    api_key: str | None,
) -> dict:
    if backend == "whisper":
        return transcribe_whisper(audio, model=whisper_model, language=language)
    if backend == "vibevoice":
        return transcribe_vibevoice(audio, url=vibevoice_url, language=language)
    if backend == "funasr":
        return transcribe_funasr(audio, url=funasr_url, language=language)
    if backend == "elevenlabs":
        key = api_key or load_api_key()
        return call_scribe(audio, key, language=language, num_speakers=num_speakers)
    sys.exit(f"unknown backend: {backend}")


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str | None = None,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
    backend: str = DEFAULT_BACKEND,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    vibevoice_url: str | None = None,
    funasr_url: str | None = None,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    `api_key` is only consulted by the elevenlabs backend; other backends
    ignore it. Kept in the signature for import back-compat with
    transcribe_batch.py.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        size_mb = audio.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"  {backend}: {video.stem}.wav ({size_mb:.1f} MB)", flush=True)
        payload = _run_backend(
            backend=backend,
            audio=audio,
            language=language,
            num_speakers=num_speakers,
            whisper_model=whisper_model,
            vibevoice_url=vibevoice_url,
            funasr_url=funasr_url,
            api_key=api_key,
        )

    out_path.write_text(json.dumps(payload, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and "words" in payload:
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video (pluggable backend)")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--backend",
        choices=["whisper", "vibevoice", "funasr", "elevenlabs"],
        default=DEFAULT_BACKEND,
        help=f"Transcription backend (default: {DEFAULT_BACKEND}).",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model size or HF repo (default: {DEFAULT_WHISPER_MODEL}). "
             "whisper backend only.",
    )
    ap.add_argument(
        "--vibevoice-url",
        default=None,
        help="Override VIBEVOICE_ASR_URL. vibevoice backend only.",
    )
    ap.add_argument(
        "--funasr-url",
        default=None,
        help="Override FUN_ASR_URL (base URL of the FunASR server). funasr backend only.",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers. elevenlabs backend only.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        api_key=None,
        language=args.language,
        num_speakers=args.num_speakers,
        backend=args.backend,
        whisper_model=args.model,
        vibevoice_url=args.vibevoice_url,
        funasr_url=args.funasr_url,
    )


if __name__ == "__main__":
    main()
