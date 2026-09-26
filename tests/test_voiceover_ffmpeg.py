"""Integration contracts for the future ffmpeg-only voiceover mux API."""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
MEDIA_TOOLS_AVAILABLE = FFMPEG is not None and FFPROBE is not None
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not MEDIA_TOOLS_AVAILABLE,
        reason="ffmpeg and ffprobe are required for media integration tests",
    ),
]

if MEDIA_TOOLS_AVAILABLE:
    from helpers.voiceover import mux_voiceover


VIDEO_DURATION = 1.2
SOURCE_FREQUENCY = 440.0
NARRATION_FREQUENCY = 880.0
SAMPLE_RATE = 48_000


def _run_ffmpeg(*arguments: str) -> None:
    subprocess.run(
        [str(FFMPEG), "-y", "-v", "error", *arguments],
        check=True,
        capture_output=True,
    )


def _make_video(
    path: Path, *, with_audio: bool, duration: float = VIDEO_DURATION
) -> None:
    arguments = [
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x90:rate=24",
    ]
    if with_audio:
        arguments.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={SOURCE_FREQUENCY}:sample_rate={SAMPLE_RATE}",
            ]
        )
    arguments.extend(
        ["-t", f"{duration:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    )
    if with_audio:
        arguments.extend(
            ["-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", str(SAMPLE_RATE)]
        )
    arguments.append(str(path))
    _run_ffmpeg(*arguments)


def _make_video_with_audio_tail(
    path: Path, *, video_duration: float, audio_duration: float
) -> None:
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=160x90:rate=24:duration={video_duration:.3f}",
        "-f",
        "lavfi",
        "-i",
        (
            f"sine=frequency={SOURCE_FREQUENCY}:sample_rate={SAMPLE_RATE}:"
            f"duration={audio_duration:.3f}"
        ),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
        "-ar",
        str(SAMPLE_RATE),
        str(path),
    )


def _make_narration(path: Path, *, duration: float) -> None:
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={NARRATION_FREQUENCY}:sample_rate=24000",
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(path),
    )


def _probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,bit_rate,sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _duration(path: Path) -> float:
    probe = _probe(path)
    return float(probe["format"]["duration"])


def _audio_stream(path: Path) -> dict[str, object]:
    streams = _probe(path)["streams"]
    return next(stream for stream in streams if stream["codec_type"] == "audio")


def _stream_duration(path: Path, codec_type: str) -> float:
    streams = _probe(path)["streams"]
    stream = next(stream for stream in streams if stream["codec_type"] == codec_type)
    return float(stream["duration"])


def _decode_mono_pcm(path: Path) -> list[float]:
    result = subprocess.run(
        [
            str(FFMPEG),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return [sample[0] for sample in struct.iter_unpack("<f", result.stdout)]


def _sample_window(
    samples: list[float], *, start: float, duration: float
) -> list[float]:
    first = int(start * SAMPLE_RATE)
    last = first + int(duration * SAMPLE_RATE)
    return samples[first:last]


def _rms(samples: list[float]) -> float:
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _tone_level(samples: list[float], frequency: float) -> float:
    phase_step = 2 * math.pi * frequency / SAMPLE_RATE
    real = sum(
        sample * math.cos(index * phase_step) for index, sample in enumerate(samples)
    )
    imaginary = sum(
        sample * math.sin(index * phase_step) for index, sample in enumerate(samples)
    )
    return math.hypot(real, imaginary) / len(samples)


@pytest.mark.integration
def test_replace_with_source_audio_uses_narration_instead_of_source_track(
    tmp_path: Path,
) -> None:
    video = tmp_path / "with-source-audio.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "replace.mp4"
    _make_video(video, with_audio=True)
    _make_narration(narration, duration=VIDEO_DURATION)

    mux_voiceover(
        video, narration, output, mix=False, duck=0.25, vo_gain=1.0, offset=0.0
    )

    samples = _sample_window(_decode_mono_pcm(output), start=0.25, duration=0.5)
    assert (
        _tone_level(samples, NARRATION_FREQUENCY)
        > _tone_level(samples, SOURCE_FREQUENCY) * 5
    )


@pytest.mark.integration
def test_replace_with_silent_source_creates_a_narration_audio_track(
    tmp_path: Path,
) -> None:
    video = tmp_path / "silent-video.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "replace.mp4"
    _make_video(video, with_audio=False)
    _make_narration(narration, duration=0.6)

    mux_voiceover(
        video, narration, output, mix=False, duck=0.25, vo_gain=1.0, offset=0.0
    )

    assert _audio_stream(output)["codec_type"] == "audio"


@pytest.mark.integration
def test_replace_leaves_silence_until_narration_offset(tmp_path: Path) -> None:
    video = tmp_path / "with-source-audio.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "offset.mp4"
    _make_video(video, with_audio=True)
    _make_narration(narration, duration=0.5)

    mux_voiceover(
        video, narration, output, mix=False, duck=0.25, vo_gain=1.0, offset=0.4
    )

    samples = _decode_mono_pcm(output)
    assert _rms(_sample_window(samples, start=0.05, duration=0.2)) < 0.001
    assert _rms(_sample_window(samples, start=0.55, duration=0.2)) > 0.01


@pytest.mark.integration
def test_mix_with_source_audio_ducks_background_during_narration(
    tmp_path: Path,
) -> None:
    video = tmp_path / "with-source-audio.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "mixed.mp4"
    _make_video(video, with_audio=True)
    _make_narration(narration, duration=0.5)

    mux_voiceover(
        video, narration, output, mix=True, duck=0.2, vo_gain=1.0, offset=0.35
    )

    samples = _decode_mono_pcm(output)
    source_before = _tone_level(
        _sample_window(samples, start=0.05, duration=0.2), SOURCE_FREQUENCY
    )
    source_during = _tone_level(
        _sample_window(samples, start=0.5, duration=0.2), SOURCE_FREQUENCY
    )
    narration_during = _tone_level(
        _sample_window(samples, start=0.5, duration=0.2), NARRATION_FREQUENCY
    )
    assert source_before > source_during * 2
    assert narration_during > 0.01


@pytest.mark.integration
def test_mix_with_silent_source_uses_silence_as_the_background(tmp_path: Path) -> None:
    video = tmp_path / "silent-video.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "mixed.mp4"
    _make_video(video, with_audio=False)
    _make_narration(narration, duration=0.5)

    mux_voiceover(
        video, narration, output, mix=True, duck=0.25, vo_gain=1.0, offset=0.35
    )

    samples = _decode_mono_pcm(output)
    assert _rms(_sample_window(samples, start=0.05, duration=0.2)) < 0.001
    assert _rms(_sample_window(samples, start=0.55, duration=0.2)) > 0.01


@pytest.mark.integration
@pytest.mark.parametrize("narration_duration", [0.2, 2.0], ids=["short", "long"])
def test_mux_keeps_video_duration_for_short_and_long_narration(
    tmp_path: Path, narration_duration: float
) -> None:
    video = tmp_path / "video.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "muxed.mp4"
    _make_video(video, with_audio=True)
    _make_narration(narration, duration=narration_duration)

    mux_voiceover(
        video, narration, output, mix=False, duck=0.25, vo_gain=1.0, offset=0.0
    )

    video_duration = _stream_duration(video, "video")
    assert _duration(output) == pytest.approx(video_duration, abs=0.12)
    assert _stream_duration(output, "audio") <= video_duration + 0.12


@pytest.mark.integration
def test_mux_ignores_an_input_audio_tail_after_the_video_stream(tmp_path: Path) -> None:
    video = tmp_path / "audio-tail.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "muxed.mp4"
    source_audio_duration = 2.0
    _make_video_with_audio_tail(
        video,
        video_duration=VIDEO_DURATION,
        audio_duration=source_audio_duration,
    )
    _make_narration(narration, duration=VIDEO_DURATION)

    video_duration = _stream_duration(video, "video")
    assert _stream_duration(video, "audio") == pytest.approx(
        source_audio_duration, abs=0.12
    )
    assert _duration(video) == pytest.approx(source_audio_duration, abs=0.12)
    assert video_duration == pytest.approx(VIDEO_DURATION, abs=0.12)

    mux_voiceover(
        video, narration, output, mix=False, duck=0.25, vo_gain=1.0, offset=0.0
    )

    output_audio_duration = _stream_duration(output, "audio")
    assert _duration(output) == pytest.approx(video_duration, abs=0.12)
    assert output_audio_duration == pytest.approx(video_duration, abs=0.12)
    assert _duration(output) <= video_duration + 0.12
    assert output_audio_duration <= video_duration + 0.12


@pytest.mark.integration
def test_mux_copies_video_and_outputs_aac_192k_48khz_stereo(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "muxed.mp4"
    _make_video(video, with_audio=True)
    _make_narration(narration, duration=VIDEO_DURATION)

    mux_voiceover(
        video, narration, output, mix=False, duck=0.25, vo_gain=1.0, offset=0.0
    )

    probe = _probe(output)
    video_stream = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    audio_stream = _audio_stream(output)
    assert video_stream["codec_name"] == "h264"
    assert audio_stream["codec_name"] == "aac"
    assert audio_stream["sample_rate"] == "48000"
    assert audio_stream["channels"] == 2
    assert int(audio_stream["bit_rate"]) == pytest.approx(192_000, abs=15_000)


def _make_matroska_video_with_audio_tail(
    path: Path, *, video_duration: float, audio_duration: float
) -> None:
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=160x90:rate=24:duration={video_duration:.3f}",
        "-f",
        "lavfi",
        "-i",
        (
            f"sine=frequency={SOURCE_FREQUENCY}:sample_rate={SAMPLE_RATE}:"
            f"duration={audio_duration:.3f}"
        ),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
        "-ar",
        str(SAMPLE_RATE),
        str(path),
    )


def _make_video_with_delayed_source_audio(
    path: Path, *, video_duration: float, audio_duration: float, delay: float
) -> None:
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=160x90:rate=24:duration={video_duration:.3f}",
        "-f",
        "lavfi",
        "-i",
        (
            f"sine=frequency={SOURCE_FREQUENCY}:sample_rate={SAMPLE_RATE}:"
            f"duration={audio_duration:.3f}"
        ),
        "-filter_complex",
        f"[1:a]asetpts=PTS+{delay}/TB[source]",
        "-map",
        "0:v:0",
        "-map",
        "[source]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-t",
        f"{video_duration:.3f}",
        str(path),
    )


def _raw_duration_probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,duration,duration_ts,time_base",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _last_video_packet_end(path: Path) -> float:
    result = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,duration_time",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    packets = json.loads(result.stdout)["packets"]
    return max(
        float(packet["pts_time"]) + float(packet.get("duration_time", 0))
        for packet in packets
    )


def _stream_start_time(path: Path, stream_selector: str) -> float:
    result = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            stream_selector,
            "-show_entries",
            "stream=start_time",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["streams"][0]["start_time"])


@pytest.mark.integration
def test_mux_uses_matroska_video_packet_end_when_video_stream_duration_is_missing(
    tmp_path: Path,
) -> None:
    video = tmp_path / "audio-tail.mkv"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "muxed.mp4"
    _make_matroska_video_with_audio_tail(
        video,
        video_duration=VIDEO_DURATION,
        audio_duration=2.0,
    )
    _make_narration(narration, duration=VIDEO_DURATION)

    input_probe = _raw_duration_probe(video)
    input_video_stream = next(
        stream for stream in input_probe["streams"] if stream["codec_type"] == "video"
    )
    assert "duration" not in input_video_stream
    assert float(input_probe["format"]["duration"]) > VIDEO_DURATION + 0.5
    video_end = _last_video_packet_end(video)

    mux_voiceover(
        video, narration, output, mix=False, duck=0.25, vo_gain=1.0, offset=0.0
    )

    assert _duration(output) <= video_end + 0.08
    assert _stream_duration(output, "audio") <= video_end + 0.08


@pytest.mark.integration
def test_mix_preserves_delayed_source_audio_start_while_narration_has_an_offset(
    tmp_path: Path,
) -> None:
    video_duration = 1.4
    video = tmp_path / "delayed-source.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "mixed.mp4"
    _make_video_with_delayed_source_audio(
        video,
        video_duration=video_duration,
        audio_duration=0.6,
        delay=0.5,
    )
    _make_narration(narration, duration=0.3)

    assert _stream_start_time(video, "v:0") == pytest.approx(0.0, abs=0.001)
    assert _stream_start_time(video, "a:0") == pytest.approx(0.5, abs=0.08)

    mux_voiceover(video, narration, output, mix=True, duck=1.0, vo_gain=1.0, offset=0.9)

    samples = _decode_mono_pcm(output)
    source_before_its_input_timestamp = _tone_level(
        _sample_window(samples, start=0.10, duration=0.15), SOURCE_FREQUENCY
    )
    source_after_its_input_timestamp = _tone_level(
        _sample_window(samples, start=0.60, duration=0.15), SOURCE_FREQUENCY
    )
    narration_after_its_offset = _tone_level(
        _sample_window(samples, start=1.00, duration=0.15), NARRATION_FREQUENCY
    )

    assert source_before_its_input_timestamp < 0.003
    assert source_after_its_input_timestamp > 0.01
    assert narration_after_its_offset > 0.01
