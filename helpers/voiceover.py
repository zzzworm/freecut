"""Generate OmniVoice narration and optionally mux it onto a video.

OmniVoice is an optional, separately installed text-to-speech engine. This
helper deliberately keeps its integration to command construction and safe
subprocess calls so it does not add an OmniVoice dependency to Freecut.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

DEFAULT_MODEL = "k2-fsa/OmniVoice"
DEFAULT_DUCK = 0.15
DEFAULT_VO_GAIN = 1.0
DEFAULT_OFFSET = 0.0
_SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".mkv"})
_MAX_STDERR_LENGTH = 1_024
_WORKDIR_PREFIX = ".freecut-voiceover-"
_FD_EXEC_WRAPPER = (
    "import os,sys;"
    "fd=int(sys.argv[1]);"
    "executable=sys.argv[2];"
    "argv=[executable,*sys.argv[3:]];"
    "os.fchdir(fd);"
    "os.close(fd);"
    "os.execv(executable,argv)"
)


class VoiceoverError(ValueError):
    """Raised when a voiceover request is invalid or cannot be prepared."""


@dataclass(frozen=True)
class VoiceoverRequest:
    """Validated settings for one OmniVoice generation request."""

    text: str
    output: Path
    video: Path | None = None
    edit_dir: Path | None = None
    model: str = DEFAULT_MODEL
    instruct: str | None = None
    ref_audio: Path | None = None
    ref_text: str | None = None
    language: str | None = None
    duration: float | None = None
    speed: float | None = None
    device: str | None = None
    dry_run: bool = False
    mix: bool = False
    duck: float = DEFAULT_DUCK
    vo_gain: float = DEFAULT_VO_GAIN
    offset: float = DEFAULT_OFFSET


@dataclass(frozen=True)
class VoiceoverPlan:
    """The executable commands that implement a voiceover request."""

    tts_command: tuple[str, ...]
    mux_command: tuple[str, ...] | None = None


@dataclass(frozen=True)
class _VideoProbe:
    """The ffprobe facts required to construct an audio filter graph."""

    duration: float
    has_audio: bool
    video_start_time: float | None = None
    audio_start_time: float | None = None


class _VoiceoverArgumentParser(argparse.ArgumentParser):
    """Use the public error type instead of terminating library callers."""

    def error(self, message: str) -> NoReturn:
        raise VoiceoverError(message)


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read the simple KEY=VALUE entries needed by this optional integration."""

    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        message = f"could not read environment file {path}: {error}"
        raise VoiceoverError(message) from error

    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("export "):
            entry = entry[7:].lstrip()
        if "=" not in entry:
            continue
        name, value = entry.split("=", maxsplit=1)
        name = name.strip()
        if not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name] = value
    return values


def load_voiceover_environment(
    *, environ: Mapping[str, str], cwd: Path, repo_dir: Path
) -> dict[str, str]:
    """Merge environment values with cwd and repository dotenv files."""

    resolved_cwd = cwd.expanduser().resolve()
    resolved_repo = repo_dir.expanduser().resolve()
    environment = _read_dotenv(resolved_repo / ".env")
    environment.update(_read_dotenv(resolved_cwd / ".env"))
    environment.update(dict(environ))
    return environment


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _configured_binary(value: str) -> Path:
    return Path(value).expanduser().resolve()


def resolve_omnivoice_binary(environment: Mapping[str, str], *, home: Path) -> Path:
    """Resolve OmniVoice in the documented deterministic order."""

    explicit_bin = environment.get("OMNIVOICE_BIN", "").strip()
    if explicit_bin:
        binary = _configured_binary(explicit_bin)
        if not _is_executable(binary):
            raise VoiceoverError(
                "OMNIVOICE_BIN must point to an executable omnivoice-infer binary: "
                f"{binary}"
            )
        return binary

    configured_home = environment.get("OMNIVOICE_HOME", "").strip()
    if configured_home:
        binary = (
            _configured_binary(configured_home) / ".venv" / "bin" / "omnivoice-infer"
        )
        if not _is_executable(binary):
            raise VoiceoverError(
                "OMNIVOICE_HOME must contain an executable "
                f".venv/bin/omnivoice-infer: {binary}"
            )
        return binary

    resolved_home = home.expanduser().resolve()
    for checkout in (
        resolved_home / "Developer" / "OmniVoice",
        resolved_home / "OmniVoice",
    ):
        binary = checkout / ".venv" / "bin" / "omnivoice-infer"
        if _is_executable(binary):
            return binary.resolve()

    on_path = shutil.which("omnivoice-infer")
    if on_path:
        binary = Path(on_path).resolve()
        if _is_executable(binary):
            return binary

    raise VoiceoverError(
        "OmniVoice not found. Set OMNIVOICE_BIN, set OMNIVOICE_HOME to its "
        "checkout, install it in ~/Developer/OmniVoice or ~/OmniVoice, or put "
        "omnivoice-infer on PATH."
    )


def _resolve_from_cwd(value: str | Path, cwd: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (cwd / path).resolve()


def _resolve_output_from_cwd(value: str | Path, cwd: Path) -> Path:
    """Canonicalize the parent without following the destination entry."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.parent.resolve() / path.name


def _parse_finite_number(
    value: str, name: str, *, minimum: float, maximum: float | None = None
) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise VoiceoverError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise VoiceoverError(f"{name} must be a finite number")
    if number < minimum or (maximum is not None and number > maximum):
        if maximum is None:
            raise VoiceoverError(f"{name} must be at least {minimum}")
        raise VoiceoverError(f"{name} must be between {minimum} and {maximum}")
    return number


def _uses_video_only_option(argv: Sequence[str]) -> bool:
    option_names = ("--mix", "--duck", "--vo-gain", "--offset")
    return any(
        argument == name or argument.startswith(f"{name}=")
        for argument in argv
        for name in option_names
    )


def _assert_does_not_overwrite_input(
    output: Path, inputs: Mapping[str, Path | None]
) -> None:
    for label, input_path in inputs.items():
        if input_path is None:
            continue
        if output == input_path:
            raise VoiceoverError(f"output must not overwrite input {label}: {output}")
        if output.exists():
            try:
                same_file = output.samefile(input_path)
            except OSError as error:
                message = f"could not compare output with input {label}: {error}"
                raise VoiceoverError(message) from error
            if same_file:
                raise VoiceoverError(
                    f"output must not overwrite input {label}: {output}"
                )


def _validate_video_output_suffix(output: Path) -> None:
    if output.suffix.lower() not in _SUPPORTED_VIDEO_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_VIDEO_SUFFIXES))
        raise VoiceoverError(
            f"video output must use a supported extension ({supported}): {output}"
        )


def parse_voiceover_args(argv: Sequence[str], *, cwd: Path) -> VoiceoverRequest:
    """Parse and validate the public voiceover CLI arguments without side effects."""

    parser = _VoiceoverArgumentParser(
        description=(
            "Generate an OmniVoice TTS voiceover and optionally mux it onto a video."
        )
    )
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--text-file", help="UTF-8 text file to synthesize")
    parser.add_argument("-o", "--output", required=True, help="Output WAV or video")
    parser.add_argument("--edit-dir", help="Edit directory for relative output")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="OmniVoice model id or path"
    )
    parser.add_argument("--instruct", help="Voice design instruction")
    parser.add_argument("--ref-audio", help="Reference audio for voice cloning")
    parser.add_argument("--ref-text", help="Optional transcript of reference audio")
    parser.add_argument("--language", help="Language name or code")
    parser.add_argument("--duration", help="Fixed output duration in seconds")
    parser.add_argument("--speed", help="Speaking-rate factor")
    parser.add_argument("--device", help="cuda, mps, or cpu")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    parser.add_argument("--onto", help="Video to receive the narration")
    parser.add_argument(
        "--mix", action="store_true", help="Mix original audio under narration"
    )
    parser.add_argument(
        "--duck", default=str(DEFAULT_DUCK), help="Background gain in [0, 1]"
    )
    parser.add_argument(
        "--vo-gain", default=str(DEFAULT_VO_GAIN), help="Linear narration gain"
    )
    parser.add_argument(
        "--offset", default=str(DEFAULT_OFFSET), help="Narration delay in seconds"
    )
    args = parser.parse_args(list(argv))

    resolved_cwd = cwd.expanduser().resolve()
    has_text = args.text is not None
    has_text_file = args.text_file is not None
    if has_text == has_text_file:
        raise VoiceoverError("exactly one of --text or --text-file is required")

    text_file: Path | None = None
    if has_text_file:
        text_file = _resolve_from_cwd(args.text_file, resolved_cwd)
        if not text_file.is_file():
            raise VoiceoverError(f"text-file does not exist: {text_file}")
        try:
            text = text_file.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as error:
            message = f"could not read text-file {text_file}: {error}"
            raise VoiceoverError(message) from error
    else:
        text = args.text.strip()
    if not text:
        raise VoiceoverError("text to synthesize must not be empty")

    video = _resolve_from_cwd(args.onto, resolved_cwd) if args.onto else None
    if video is not None and not video.is_file():
        raise VoiceoverError(f"onto video does not exist: {video}")
    if video is None and _uses_video_only_option(argv):
        raise VoiceoverError("video-only options require --onto")

    ref_audio = (
        _resolve_from_cwd(args.ref_audio, resolved_cwd) if args.ref_audio else None
    )
    if ref_audio is not None and not ref_audio.is_file():
        raise VoiceoverError(f"ref-audio does not exist: {ref_audio}")
    if args.ref_text is not None and ref_audio is None:
        raise VoiceoverError("--ref-text requires --ref-audio")
    if args.instruct is not None and ref_audio is not None:
        raise VoiceoverError("--instruct and --ref-audio are mutually exclusive")

    duration = (
        _parse_finite_number(args.duration, "duration", minimum=0.0)
        if args.duration is not None
        else None
    )
    if duration == 0.0:
        raise VoiceoverError("duration must be greater than 0")
    speed = (
        _parse_finite_number(args.speed, "speed", minimum=0.0)
        if args.speed is not None
        else None
    )
    if speed == 0.0:
        raise VoiceoverError("speed must be greater than 0")
    if duration is not None and speed is not None:
        raise VoiceoverError("duration and speed are mutually exclusive")

    duck = _parse_finite_number(args.duck, "duck", minimum=0.0, maximum=1.0)
    vo_gain = _parse_finite_number(args.vo_gain, "vo-gain", minimum=0.0)
    offset = _parse_finite_number(args.offset, "offset", minimum=0.0)

    edit_dir = (
        _resolve_from_cwd(args.edit_dir, resolved_cwd)
        if args.edit_dir is not None
        else None
    )
    if edit_dir is not None and edit_dir.exists() and not edit_dir.is_dir():
        raise VoiceoverError(f"edit-dir is not a directory: {edit_dir}")
    output_base = edit_dir if edit_dir is not None else resolved_cwd
    output = _resolve_output_from_cwd(args.output, output_base)
    if video is None:
        if output.suffix.lower() != ".wav":
            raise VoiceoverError(
                "output must use a .wav extension when --onto is not provided: "
                f"{output}"
            )
    else:
        _validate_video_output_suffix(output)

    sidecar = output.with_suffix(".vo.wav") if video is not None else None
    protected_inputs = {
        "onto video": video,
        "text-file": text_file,
        "ref-audio": ref_audio,
    }
    _assert_does_not_overwrite_input(output, protected_inputs)
    if sidecar is not None:
        _assert_does_not_overwrite_input(sidecar, protected_inputs)

    return VoiceoverRequest(
        text=text,
        output=output,
        video=video,
        edit_dir=edit_dir,
        model=args.model,
        instruct=args.instruct,
        ref_audio=ref_audio,
        ref_text=args.ref_text,
        language=args.language,
        duration=duration,
        speed=speed,
        device=args.device,
        dry_run=args.dry_run,
        mix=args.mix,
        duck=duck,
        vo_gain=vo_gain,
        offset=offset,
    )


def build_omnivoice_command(
    request: VoiceoverRequest, binary: Path | str, tts_output: Path
) -> list[str]:
    """Build the argument-vector passed to the upstream OmniVoice CLI."""

    command = [
        str(Path(binary).expanduser().resolve()),
        "--model",
        request.model,
        "--text",
        request.text,
        "--output",
        str(_resolve_output_from_cwd(tts_output, Path.cwd())),
    ]
    if request.instruct is not None:
        command.extend(("--instruct", request.instruct))
    if request.ref_audio is not None:
        command.extend(("--ref_audio", str(request.ref_audio)))
    if request.ref_text is not None:
        command.extend(("--ref_text", request.ref_text))
    if request.language is not None:
        command.extend(("--language", request.language))
    if request.duration is not None:
        command.extend(("--duration", str(request.duration)))
    if request.speed is not None:
        command.extend(("--speed", str(request.speed)))
    if request.device is not None:
        command.extend(("--device", request.device))
    return command


def _usable_duration(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        duration = float(value)
    except ValueError:
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration


def _usable_timestamp(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        timestamp = float(value)
    except ValueError:
        return None
    if not math.isfinite(timestamp):
        return None
    return timestamp


def _duration_from_timestamp(stream: Mapping[str, object]) -> float | None:
    duration_ts = _usable_duration(stream.get("duration_ts"))
    time_base = stream.get("time_base")
    if duration_ts is None or not isinstance(time_base, str):
        return None
    try:
        numerator, denominator = time_base.split("/", maxsplit=1)
        duration = duration_ts * (float(numerator) / float(denominator))
    except (ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration


def _duration_from_matroska_tag(stream: Mapping[str, object]) -> float | None:
    tags = stream.get("tags")
    if not isinstance(tags, dict):
        return None
    raw_duration = tags.get("DURATION")
    if not isinstance(raw_duration, str):
        return None
    parts = raw_duration.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (float(part) for part in parts)
    except ValueError:
        return None
    duration = hours * 3_600 + minutes * 60 + seconds
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration


def _bounded_stderr(error: subprocess.CalledProcessError) -> str:
    stderr = error.stderr
    if isinstance(stderr, bytes):
        text = stderr.decode(errors="replace")
    elif isinstance(stderr, str):
        text = stderr
    else:
        return ""
    text = text.strip()
    if len(text) > _MAX_STDERR_LENGTH:
        text = f"{text[:_MAX_STDERR_LENGTH]}..."
    return text


def _called_process_message(prefix: str, error: subprocess.CalledProcessError) -> str:
    detail = _bounded_stderr(error)
    message = f"{prefix} (exit status {error.returncode})"
    if detail:
        message = f"{message}: {detail}"
    return message


def _resolve_media_tool(name: str) -> Path:
    located = shutil.which(name)
    if located is None:
        raise VoiceoverError(f"required media tool not found on PATH: {name}")
    executable = Path(located).resolve()
    if not _is_executable(executable):
        raise VoiceoverError(f"required media tool is not executable: {executable}")
    return executable


def _run_json_probe(command: list[str], *, description: str) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
    except subprocess.CalledProcessError as error:
        raise VoiceoverError(_called_process_message(description, error)) from error
    except OSError as error:
        raise VoiceoverError(f"{description}: {error}") from error
    try:
        data = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise VoiceoverError(f"{description}: unexpected JSON output") from error
    if not isinstance(data, dict):
        raise VoiceoverError(f"{description}: unexpected JSON output")
    return data


def _packet_video_duration(
    ffprobe: Path,
    video: Path,
    *,
    video_start_time: float | None,
) -> float | None:
    data = _run_json_probe(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,dts_time,duration_time",
            "-of",
            "json",
            str(video),
        ],
        description=f"could not probe video packets {video}",
    )
    packets = data.get("packets")
    if not isinstance(packets, list):
        return None

    packet_starts: list[float] = []
    packet_ends: list[float] = []
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        timestamp = _usable_timestamp(packet.get("pts_time"))
        if timestamp is None:
            timestamp = _usable_timestamp(packet.get("dts_time"))
        if timestamp is None:
            continue
        packet_duration = _usable_timestamp(packet.get("duration_time"))
        if packet_duration is None or packet_duration < 0:
            packet_duration = 0.0
        packet_starts.append(timestamp)
        packet_ends.append(timestamp + packet_duration)
    if not packet_ends:
        return None

    start = video_start_time
    if start is None:
        start = min(packet_starts)
    return _usable_duration(max(packet_ends) - start)


def _probe_video(video: Path) -> _VideoProbe:
    video = video.expanduser().resolve()
    ffprobe = _resolve_media_tool("ffprobe")
    data = _run_json_probe(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration:"
                "stream=codec_type,duration,duration_ts,time_base,start_time:"
                "stream_tags=DURATION"
            ),
            "-of",
            "json",
            str(video),
        ],
        description=f"could not probe video {video}",
    )

    streams = data.get("streams")
    if not isinstance(streams, list):
        raise VoiceoverError(f"video has no video stream: {video}")
    video_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        None,
    )
    if video_stream is None:
        raise VoiceoverError(f"video has no video stream: {video}")
    audio_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ),
        None,
    )
    has_audio = audio_stream is not None
    video_start_time = _usable_timestamp(video_stream.get("start_time"))
    audio_start_time = (
        _usable_timestamp(audio_stream.get("start_time"))
        if audio_stream is not None
        else None
    )

    duration = _usable_duration(video_stream.get("duration"))
    if duration is None:
        duration = _duration_from_timestamp(video_stream)
    if duration is None:
        duration = _duration_from_matroska_tag(video_stream)
    if duration is None:
        try:
            duration = _packet_video_duration(
                ffprobe,
                video,
                video_start_time=video_start_time,
            )
        except VoiceoverError:
            if has_audio:
                raise
    if duration is None and not has_audio:
        format_info = data.get("format")
        if isinstance(format_info, dict):
            duration = _usable_duration(format_info.get("duration"))
    if duration is None:
        raise VoiceoverError(f"video has no usable video-stream duration: {video}")

    return _VideoProbe(
        duration=duration,
        has_audio=has_audio,
        video_start_time=video_start_time,
        audio_start_time=audio_start_time,
    )


def _narration_duration_from_probe(
    data: Mapping[str, object], narration: str | Path
) -> float:
    streams = data.get("streams")
    if not isinstance(streams, list):
        raise VoiceoverError(f"narration has no decodable audio stream: {narration}")
    audio_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ),
        None,
    )
    if audio_stream is None:
        raise VoiceoverError(f"narration has no decodable audio stream: {narration}")
    duration = _usable_duration(audio_stream.get("duration"))
    if duration is None:
        duration = _duration_from_timestamp(audio_stream)
    if duration is None:
        format_info = data.get("format")
        if isinstance(format_info, dict):
            duration = _usable_duration(format_info.get("duration"))
    if duration is None:
        raise VoiceoverError(f"narration has no usable duration: {narration}")
    return duration


def _narration_probe_arguments(narration: str) -> list[str]:
    return [
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,duration,duration_ts,time_base",
        "-of",
        "json",
        narration,
    ]


def _probe_narration_duration(narration: Path) -> float:
    narration = narration.expanduser().resolve()
    ffprobe = _resolve_media_tool("ffprobe")
    data = _run_json_probe(
        [str(ffprobe), *_narration_probe_arguments(str(narration))],
        description=f"could not probe narration {narration}",
    )
    return _narration_duration_from_probe(data, narration)


def _format_number(value: float) -> str:
    return str(value)


def _build_mux_command(
    video: Path,
    narration: Path,
    output: Path,
    *,
    mix: bool,
    duck: float,
    vo_gain: float,
    offset: float,
    probe: _VideoProbe,
    narration_duration: float | None,
) -> list[str]:
    duration = _format_number(probe.duration)
    delay_ms = round(offset * 1_000)
    voice_chain = (
        "[1:a]asetpts=PTS-STARTPTS,aresample=48000,"
        "aformat=channel_layouts=stereo,"
        f"volume={_format_number(vo_gain)},adelay={delay_ms}|{delay_ms},"
        f"apad,atrim=duration={duration}[vo]"
    )

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-i",
        str(narration),
    ]
    if mix and probe.has_audio:
        video_start = probe.video_start_time or 0.0
        audio_start = (
            probe.audio_start_time
            if probe.audio_start_time is not None
            else video_start
        )
        relative_audio_start = audio_start - video_start
        if relative_audio_start >= 0:
            background_start = (
                "[0:a:0]asetpts=PTS-STARTPTS,"
                f"adelay={round(relative_audio_start * 1_000)}|"
                f"{round(relative_audio_start * 1_000)},"
            )
        else:
            background_start = (
                f"[0:a:0]atrim=start={_format_number(-relative_audio_start)},"
                "asetpts=PTS-STARTPTS,"
            )
        duck_end = probe.duration
        if narration_duration is not None:
            duck_end = min(probe.duration, offset + narration_duration)
        background_chain = (
            f"{background_start}aresample=48000,"
            "aformat=channel_layouts=stereo,"
            f"volume={_format_number(duck)}:enable='between(t,{_format_number(offset)},"
            f"{_format_number(duck_end)})',apad,atrim=duration={duration}[bg]"
        )
    else:
        command.extend(
            [
                "-f",
                "lavfi",
                "-t",
                duration,
                "-i",
                "anullsrc=r=48000:cl=stereo",
            ]
        )
        background_chain = (
            f"[2:a]asetpts=PTS-STARTPTS,apad,atrim=duration={duration}[bg]"
        )
    filter_complex = (
        f"{voice_chain};{background_chain};"
        "[bg][vo]amix=inputs=2:duration=first:normalize=0,"
        "alimiter=limit=0.99,aresample=48000,"
        "aformat=channel_layouts=stereo[aout]"
    )

    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-t",
            duration,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-minrate",
            "192k",
            "-maxrate",
            "192k",
            "-bufsize",
            "384k",
            "-ar",
            "48000",
            "-ac",
            "2",
        ]
    )
    suffix = output.suffix.lower()
    if suffix != ".mkv":
        command.extend(("-movflags", "+faststart"))
    if suffix == ".m4v":
        command.extend(("-f", "mp4"))
    command.append(str(output))
    return command


def _require_secure_output_capabilities() -> None:
    required_constants = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    required_dir_fd_functions = (os.open, os.mkdir, os.stat, os.unlink, os.rmdir)
    if (
        os.name != "posix"
        or not callable(getattr(os, "geteuid", None))
        or not callable(getattr(os, "fchmod", None))
        or any(not hasattr(os, name) for name in required_constants)
        or any(
            function not in os.supports_dir_fd for function in required_dir_fd_functions
        )
        or os.listdir not in os.supports_fd
    ):
        raise VoiceoverError(
            "secure voiceover output requires POSIX directory-fd and nofollow support"
        )


def _directory_open_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    return flags


def _validate_directory_fd(
    descriptor: int,
    path: Path,
    *,
    newly_created: bool = False,
) -> os.stat_result:
    details = os.fstat(descriptor)
    effective_uid = os.geteuid()
    if not stat.S_ISDIR(details.st_mode):
        raise VoiceoverError(f"unsafe output directory {path}: not a directory")
    if details.st_uid not in {effective_uid, 0}:
        raise VoiceoverError(
            f"unsafe output directory {path}: owner must be current user or root"
        )
    writable_by_others = details.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if writable_by_others and not details.st_mode & stat.S_ISVTX:
        raise VoiceoverError(
            f"unsafe output directory {path}: group/world writable without sticky bit"
        )
    if newly_created:
        if details.st_uid != effective_uid:
            raise VoiceoverError(
                f"unsafe output directory {path}: newly created directory owner changed"
            )
        if stat.S_IMODE(details.st_mode) != 0o700:
            raise VoiceoverError(
                f"unsafe output directory {path}: newly created directory is not 0700"
            )
    return details


def _open_output_directory(output: Path, *, create_missing: bool) -> int:
    _require_secure_output_capabilities()
    directory = output.parent
    if not directory.is_absolute():
        raise VoiceoverError(f"output directory must be absolute: {directory}")

    flags = _directory_open_flags()
    try:
        descriptor = os.open("/", flags)
        _validate_directory_fd(descriptor, Path("/"))
    except (OSError, VoiceoverError) as error:
        if isinstance(error, VoiceoverError):
            raise
        raise VoiceoverError(f"unsafe output directory /: {error}") from error

    current_path = Path("/")
    try:
        for component in directory.parts[1:]:
            current_path /= component
            created = False
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create_missing:
                    raise VoiceoverError(
                        f"unsafe output directory {current_path}: does not exist"
                    ) from None
                try:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                        created = True
                    except FileExistsError:
                        pass
                    child = os.open(component, flags, dir_fd=descriptor)
                except OSError as error:
                    raise VoiceoverError(
                        f"unsafe output directory {current_path}: {error}"
                    ) from error
            except OSError as error:
                raise VoiceoverError(
                    f"unsafe output directory {current_path}: {error}"
                ) from error
            try:
                if created:
                    os.fchmod(child, 0o700)
                _validate_directory_fd(
                    child,
                    current_path,
                    newly_created=created,
                )
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_safe_output_directory(output: Path) -> None:
    """Validate an existing output namespace using nofollow directory handles."""

    descriptor = _open_output_directory(output, create_missing=False)
    os.close(descriptor)


def _output_directory_fd_opened(output: Path, output_dir_fd: int) -> None:
    """Test seam after binding the final directory and before creating workdir."""


def _run_in_workdir(
    executable: Path,
    arguments: Sequence[str],
    *,
    workdir_fd: int,
    capture_output: bool = False,
    text: bool = False,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    executable = executable.expanduser().resolve()
    if not executable.is_absolute() or not _is_executable(executable):
        raise VoiceoverError(f"external executable is not usable: {executable}")
    wrapper_python = Path(sys.executable).resolve()
    command = [
        str(wrapper_python),
        "-I",
        "-S",
        "-c",
        _FD_EXEC_WRAPPER,
        str(workdir_fd),
        str(executable),
        *arguments,
    ]
    return subprocess.run(
        command,
        check=True,
        capture_output=capture_output,
        text=text,
        shell=False,
        pass_fds=(workdir_fd,),
    )


def _remove_directory_contents(descriptor: int) -> None:
    flags = _directory_open_flags()
    for name in os.listdir(descriptor):
        details = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(details.st_mode):
            child = os.open(name, flags, dir_fd=descriptor)
            try:
                _remove_directory_contents(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)


class _OutputTransaction:
    """A private workdir and publication namespace bound to directory fds."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.output_dir_fd = -1
        self.workdir_fd = -1
        self.workdir_name = ""
        self._artifact_fds: dict[str, int] = {}

    def __enter__(self) -> _OutputTransaction:  # noqa: PYI034
        self.output_dir_fd = _open_output_directory(
            self.output,
            create_missing=True,
        )
        try:
            _output_directory_fd_opened(self.output, self.output_dir_fd)
            self._create_workdir()
        except BaseException:
            self._cleanup(suppress_errors=True)
            raise
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> Literal[False]:
        self._cleanup(suppress_errors=exception_type is not None)
        return False

    def _create_workdir(self) -> None:
        flags = _directory_open_flags()
        for _attempt in range(100):
            name = f"{_WORKDIR_PREFIX}{secrets.token_hex(16)}"
            try:
                os.mkdir(name, mode=0o700, dir_fd=self.output_dir_fd)
            except FileExistsError:
                continue
            except OSError as error:
                raise VoiceoverError(
                    f"could not create private voiceover workdir: {error}"
                ) from error
            self.workdir_name = name
            try:
                self.workdir_fd = os.open(
                    name,
                    flags,
                    dir_fd=self.output_dir_fd,
                )
                os.fchmod(self.workdir_fd, 0o700)
                _validate_directory_fd(
                    self.workdir_fd,
                    self.output.parent / name,
                    newly_created=True,
                )
                if sys.platform == "darwin":
                    try:
                        _run_in_workdir(
                            Path("/bin/chmod"),
                            ("-N", "."),
                            workdir_fd=self.workdir_fd,
                            capture_output=True,
                            text=True,
                        )
                    except subprocess.CalledProcessError as error:
                        raise VoiceoverError(
                            _called_process_message(
                                "could not clear inherited ACL from voiceover workdir",
                                error,
                            )
                        ) from error
                    except OSError as error:
                        raise VoiceoverError(
                            "could not clear inherited ACL from voiceover workdir: "
                            f"{error}"
                        ) from error
                    os.fchmod(self.workdir_fd, 0o700)
                    _validate_directory_fd(
                        self.workdir_fd,
                        self.output.parent / name,
                        newly_created=True,
                    )
                return
            except OSError as error:
                self._cleanup(suppress_errors=True)
                raise VoiceoverError(
                    f"could not secure private voiceover workdir: {error}"
                ) from error
            except BaseException:
                self._cleanup(suppress_errors=True)
                raise
        raise VoiceoverError("could not allocate a private voiceover workdir")

    def run(
        self,
        executable: Path,
        arguments: Sequence[str],
        *,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        return _run_in_workdir(
            executable,
            arguments,
            workdir_fd=self.workdir_fd,
            capture_output=capture_output,
            text=text,
        )

    def validate_artifact(self, name: str) -> os.stat_result:
        if Path(name).name != name:
            raise VoiceoverError(f"invalid work artifact name: {name}")
        existing_fd = self._artifact_fds.get(name)
        if existing_fd is not None:
            return os.fstat(existing_fd)
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = -1
        try:
            descriptor = os.open(name, flags, dir_fd=self.workdir_fd)
            details = os.fstat(descriptor)
        except OSError as error:
            if descriptor >= 0:
                os.close(descriptor)
            raise VoiceoverError(
                f"invalid voiceover work artifact {name}: {error}"
            ) from error
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_size <= 0
            or details.st_uid != os.geteuid()
        ):
            os.close(descriptor)
            raise VoiceoverError(
                f"invalid voiceover work artifact {name}: "
                "must be a non-empty regular file owned by current user"
            )
        self._artifact_fds[name] = descriptor
        return details

    def publish(self, source_name: str, destination_name: str) -> None:
        if Path(destination_name).name != destination_name:
            raise VoiceoverError(f"invalid output basename: {destination_name}")
        self.validate_artifact(source_name)
        artifact_fd = self._artifact_fds[source_name]
        artifact_details = os.fstat(artifact_fd)
        try:
            destination_details = os.stat(
                destination_name,
                dir_fd=self.output_dir_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            destination_details = None
        except OSError as error:
            raise VoiceoverError(
                f"could not inspect existing output {self.output.parent / destination_name}: "
                f"{error}"
            ) from error
        if destination_details is not None and stat.S_ISREG(
            destination_details.st_mode
        ):
            os.fchmod(artifact_fd, stat.S_IMODE(destination_details.st_mode))
            artifact_details = os.fstat(artifact_fd)
        try:
            os.replace(
                source_name,
                destination_name,
                src_dir_fd=self.workdir_fd,
                dst_dir_fd=self.output_dir_fd,
            )
            published = os.stat(
                destination_name,
                dir_fd=self.output_dir_fd,
                follow_symlinks=False,
            )
        except OSError as error:
            raise VoiceoverError(
                f"could not publish voiceover output "
                f"{self.output.parent / destination_name}: {error}"
            ) from error
        if (published.st_dev, published.st_ino) != (
            artifact_details.st_dev,
            artifact_details.st_ino,
        ):
            raise VoiceoverError(
                f"published voiceover output changed unexpectedly: "
                f"{self.output.parent / destination_name}"
            )

    def _cleanup(self, *, suppress_errors: bool) -> None:
        cleanup_error: OSError | None = None
        for descriptor in self._artifact_fds.values():
            try:
                os.close(descriptor)
            except OSError as error:
                cleanup_error = cleanup_error or error
        self._artifact_fds.clear()

        if self.workdir_fd >= 0:
            try:
                _remove_directory_contents(self.workdir_fd)
            except OSError as error:
                cleanup_error = cleanup_error or error
            try:
                os.close(self.workdir_fd)
            except OSError as error:
                cleanup_error = cleanup_error or error
            self.workdir_fd = -1
        if self.output_dir_fd >= 0 and self.workdir_name:
            try:
                os.rmdir(self.workdir_name, dir_fd=self.output_dir_fd)
            except FileNotFoundError:
                pass
            except OSError as error:
                cleanup_error = cleanup_error or error
            self.workdir_name = ""
        if self.output_dir_fd >= 0:
            try:
                os.close(self.output_dir_fd)
            except OSError as error:
                cleanup_error = cleanup_error or error
            self.output_dir_fd = -1
        if cleanup_error is not None and not suppress_errors:
            raise VoiceoverError(
                f"could not clean private voiceover workdir: {cleanup_error}"
            ) from cleanup_error


def _probe_workdir_narration(
    transaction: _OutputTransaction,
    ffprobe: Path,
    narration_name: str,
) -> float:
    try:
        result = transaction.run(
            ffprobe,
            _narration_probe_arguments(narration_name),
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise VoiceoverError(
            _called_process_message(
                f"could not probe narration {narration_name}",
                error,
            )
        ) from error
    except OSError as error:
        raise VoiceoverError(
            f"could not probe narration {narration_name}: {error}"
        ) from error
    if not isinstance(result.stdout, str):
        raise VoiceoverError(
            f"could not probe narration {narration_name}: unexpected output"
        )
    try:
        data = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise VoiceoverError(
            f"could not probe narration {narration_name}: unexpected JSON output"
        ) from error
    if not isinstance(data, dict):
        raise VoiceoverError(
            f"could not probe narration {narration_name}: unexpected JSON output"
        )
    return _narration_duration_from_probe(data, narration_name)


def _actual_tts_arguments(
    logical_command: Sequence[str],
    *,
    output_name: str,
) -> list[str]:
    arguments = list(logical_command[1:])
    output_index = arguments.index("--output") + 1
    arguments[output_index] = output_name
    return arguments


def _run_tts(
    transaction: _OutputTransaction,
    binary: Path,
    logical_command: Sequence[str],
    *,
    output_name: str,
) -> None:
    try:
        transaction.run(
            binary,
            _actual_tts_arguments(logical_command, output_name=output_name),
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise VoiceoverError(
            _called_process_message("OmniVoice generation failed", error)
        ) from error
    except OSError as error:
        raise VoiceoverError(f"OmniVoice generation failed: {error}") from error


def _run_ffmpeg(
    transaction: _OutputTransaction,
    ffmpeg: Path,
    command: Sequence[str],
) -> None:
    try:
        transaction.run(
            ffmpeg,
            command[1:],
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise VoiceoverError(_called_process_message("ffmpeg failed", error)) from error
    except OSError as error:
        raise VoiceoverError(f"ffmpeg failed: {error}") from error


def mux_voiceover(
    video: Path,
    narration: Path,
    output: Path,
    *,
    mix: bool,
    duck: float,
    vo_gain: float,
    offset: float,
) -> list[str]:
    """Mux narration into a video and return the logical ffmpeg command."""

    video = video.expanduser().resolve()
    narration = narration.expanduser().resolve()
    output = _resolve_output_from_cwd(output, Path.cwd())
    _validate_video_output_suffix(output)
    for value, name, minimum, maximum in (
        (duck, "duck", 0.0, 1.0),
        (vo_gain, "vo_gain", 0.0, None),
        (offset, "offset", 0.0, None),
    ):
        if (
            not math.isfinite(value)
            or value < minimum
            or (maximum is not None and value > maximum)
        ):
            raise VoiceoverError(f"invalid {name}")
    _assert_does_not_overwrite_input(
        output,
        {"onto video": video, "narration": narration},
    )
    if not video.is_file():
        raise VoiceoverError(f"onto video does not exist: {video}")
    if not narration.is_file() or narration.stat().st_size == 0:
        raise VoiceoverError(f"narration WAV does not exist or is empty: {narration}")

    probe = _probe_video(video)
    narration_duration = _probe_narration_duration(narration)
    ffmpeg = _resolve_media_tool("ffmpeg")
    logical_command = _build_mux_command(
        video,
        narration,
        output,
        mix=mix,
        duck=duck,
        vo_gain=vo_gain,
        offset=offset,
        probe=probe,
        narration_duration=narration_duration,
    )
    render_name = f"render{output.suffix.lower()}"
    execution_command = _build_mux_command(
        video,
        narration,
        Path(render_name),
        mix=mix,
        duck=duck,
        vo_gain=vo_gain,
        offset=offset,
        probe=probe,
        narration_duration=narration_duration,
    )
    with _OutputTransaction(output) as transaction:
        _run_ffmpeg(transaction, ffmpeg, execution_command)
        transaction.validate_artifact(render_name)
        transaction.publish(render_name, output.name)
    return logical_command


def run_voiceover(
    request: VoiceoverRequest,
    *,
    environ: Mapping[str, str],
    cwd: Path,
    repo_dir: Path,
) -> VoiceoverPlan:
    """Execute a request, or return and print its zero-write dry-run plan."""

    environment = load_voiceover_environment(
        environ=environ,
        cwd=cwd,
        repo_dir=repo_dir,
    )
    binary = resolve_omnivoice_binary(environment, home=Path.home())
    narration = (
        request.output.with_suffix(".vo.wav") if request.video else request.output
    )
    _assert_does_not_overwrite_input(
        request.output,
        {"onto video": request.video, "ref-audio": request.ref_audio},
    )
    _assert_does_not_overwrite_input(
        narration,
        {"onto video": request.video, "ref-audio": request.ref_audio},
    )
    tts_command = build_omnivoice_command(request, binary, narration)
    mux_command: tuple[str, ...] | None = None

    if request.dry_run:
        if request.video is not None:
            probe = _probe_video(request.video)
            mux_command = tuple(
                _build_mux_command(
                    request.video,
                    narration,
                    request.output,
                    mix=request.mix,
                    duck=request.duck,
                    vo_gain=request.vo_gain,
                    offset=request.offset,
                    probe=probe,
                    narration_duration=request.duration,
                )
            )
        plan = VoiceoverPlan(tuple(tts_command), mux_command)
        print(f"TTS: {shlex.join(plan.tts_command)}")
        if plan.mux_command is not None:
            print(f"ffmpeg: {shlex.join(plan.mux_command)}")
        return plan

    ffprobe = _resolve_media_tool("ffprobe")
    video_probe: _VideoProbe | None = None
    ffmpeg: Path | None = None
    if request.video is not None:
        _validate_video_output_suffix(request.output)
        video_probe = _probe_video(request.video)
        ffmpeg = _resolve_media_tool("ffmpeg")

    print(f"TTS: {shlex.join(tts_command)}")
    narration_name = "narration.wav"
    with _OutputTransaction(request.output) as transaction:
        _run_tts(
            transaction,
            binary,
            tts_command,
            output_name=narration_name,
        )
        transaction.validate_artifact(narration_name)
        narration_duration = _probe_workdir_narration(
            transaction,
            ffprobe,
            narration_name,
        )

        if request.video is None:
            transaction.publish(narration_name, request.output.name)
        else:
            assert video_probe is not None
            assert ffmpeg is not None
            logical_mux = _build_mux_command(
                request.video,
                narration,
                request.output,
                mix=request.mix,
                duck=request.duck,
                vo_gain=request.vo_gain,
                offset=request.offset,
                probe=video_probe,
                narration_duration=narration_duration,
            )
            mux_command = tuple(logical_mux)
            render_name = f"render{request.output.suffix.lower()}"
            execution_mux = _build_mux_command(
                request.video,
                Path(narration_name),
                Path(render_name),
                mix=request.mix,
                duck=request.duck,
                vo_gain=request.vo_gain,
                offset=request.offset,
                probe=video_probe,
                narration_duration=narration_duration,
            )
            print(f"ffmpeg: {shlex.join(logical_mux)}")
            _run_ffmpeg(transaction, ffmpeg, execution_mux)
            transaction.validate_artifact(render_name)
            transaction.publish(narration_name, narration.name)
            transaction.publish(render_name, request.output.name)

    return VoiceoverPlan(tuple(tts_command), mux_command)


def main() -> None:
    """Run the CLI and present validation failures as concise user errors."""

    cwd = Path.cwd()
    repo_dir = Path(__file__).resolve().parents[1]
    try:
        request = parse_voiceover_args(sys.argv[1:], cwd=cwd)
        run_voiceover(request, environ=os.environ, cwd=cwd, repo_dir=repo_dir)
    except VoiceoverError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
