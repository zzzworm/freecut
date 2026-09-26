"""Unit-level contract tests for the future OmniVoice render post-processor."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from helpers import voiceover
from helpers.voiceover import (
    VoiceoverError,
    build_omnivoice_command,
    load_voiceover_environment,
    mux_voiceover,
    parse_voiceover_args,
    resolve_omnivoice_binary,
    run_voiceover,
)


@pytest.fixture
def executable(tmp_path: Path) -> Path:
    path = tmp_path / "omnivoice-infer"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def writing_executable(tmp_path: Path) -> Path:
    path = tmp_path / "omnivoice-infer"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "import wave\n"
        "output = sys.argv[sys.argv.index('--output') + 1]\n"
        "with wave.open(output, 'wb') as generated:\n"
        "    generated.setnchannels(1)\n"
        "    generated.setsampwidth(2)\n"
        "    generated.setframerate(8000)\n"
        "    generated.writeframes(b'\\x00\\x00' * 80)\n"
    )
    path.chmod(0o755)
    return path


def _completed_command(
    executable: Path,
    arguments: Sequence[str],
    *,
    text: bool,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    output: str | bytes = "" if text else b""
    return subprocess.CompletedProcess(
        [str(executable), *arguments],
        0,
        stdout=output,
        stderr=output,
    )


def _write_workdir_artifact(
    workdir_fd: int,
    name: str,
    contents: bytes,
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o666,
        dir_fd=workdir_fd,
    )
    try:
        os.write(descriptor, contents)
    finally:
        os.close(descriptor)


@pytest.mark.unit
def test_load_environment_prefers_process_values_over_dotenv_files(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "cwd"
    repo_dir = tmp_path / "repo"
    cwd.mkdir()
    repo_dir.mkdir()
    (cwd / ".env").write_text("OMNIVOICE_HOME=/from-cwd\n")
    (repo_dir / ".env").write_text("OMNIVOICE_HOME=/from-repo\n")

    environment = load_voiceover_environment(
        environ={"OMNIVOICE_HOME": "/from-process"},
        cwd=cwd,
        repo_dir=repo_dir,
    )

    assert environment["OMNIVOICE_HOME"] == "/from-process"


@pytest.mark.unit
def test_load_environment_uses_cwd_dotenv_before_repo_dotenv(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    repo_dir = tmp_path / "repo"
    cwd.mkdir()
    repo_dir.mkdir()
    (cwd / ".env").write_text("OMNIVOICE_BIN=/from-cwd/bin\n")
    (repo_dir / ".env").write_text("OMNIVOICE_BIN=/from-repo/bin\n")

    environment = load_voiceover_environment(environ={}, cwd=cwd, repo_dir=repo_dir)

    assert environment["OMNIVOICE_BIN"] == "/from-cwd/bin"


@pytest.mark.unit
def test_load_environment_falls_back_to_repo_dotenv(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    repo_dir = tmp_path / "repo"
    cwd.mkdir()
    repo_dir.mkdir()
    (repo_dir / ".env").write_text("OMNIVOICE_HOME=/from-repo\n")

    environment = load_voiceover_environment(environ={}, cwd=cwd, repo_dir=repo_dir)

    assert environment["OMNIVOICE_HOME"] == "/from-repo"


@pytest.mark.unit
def test_load_environment_wraps_a_non_utf8_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_bytes(b"OMNIVOICE_BIN=\xff")

    with pytest.raises(VoiceoverError, match="could not read environment file"):
        load_voiceover_environment(environ={}, cwd=tmp_path, repo_dir=tmp_path)


@pytest.mark.unit
def test_resolve_binary_prefers_explicit_omnivoice_bin(
    tmp_path: Path, executable: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    project_binary = (
        home / "Developer" / "OmniVoice" / ".venv" / "bin" / "omnivoice-infer"
    )
    project_binary.parent.mkdir(parents=True)
    project_binary.write_text("#!/bin/sh\nexit 0\n")
    project_binary.chmod(0o755)

    binary = resolve_omnivoice_binary({"OMNIVOICE_BIN": str(executable)}, home=home)

    assert binary == executable.resolve()


@pytest.mark.unit
def test_resolve_binary_rejects_an_invalid_explicit_omnivoice_bin(
    tmp_path: Path,
) -> None:
    with pytest.raises(VoiceoverError, match="OMNIVOICE_BIN"):
        resolve_omnivoice_binary(
            {"OMNIVOICE_BIN": str(tmp_path / "does-not-exist")},
            home=tmp_path / "home",
        )


@pytest.mark.unit
def test_resolve_binary_uses_omnivoice_home_venv_before_checkout_locations(
    tmp_path: Path,
) -> None:
    configured_home = tmp_path / "configured"
    configured_binary = configured_home / ".venv" / "bin" / "omnivoice-infer"
    configured_binary.parent.mkdir(parents=True)
    configured_binary.write_text("#!/bin/sh\nexit 0\n")
    configured_binary.chmod(0o755)

    binary = resolve_omnivoice_binary(
        {"OMNIVOICE_HOME": str(configured_home)}, home=tmp_path / "user-home"
    )

    assert binary == configured_binary.resolve()


@pytest.mark.unit
def test_resolve_binary_uses_developer_checkout_before_home_checkout(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    developer_binary = (
        home / "Developer" / "OmniVoice" / ".venv" / "bin" / "omnivoice-infer"
    )
    fallback_binary = home / "OmniVoice" / ".venv" / "bin" / "omnivoice-infer"
    for path in (developer_binary, fallback_binary):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)

    binary = resolve_omnivoice_binary({}, home=home)

    assert binary == developer_binary.resolve()


@pytest.mark.unit
def test_resolve_binary_falls_back_to_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    path_binary = bin_dir / "omnivoice-infer"
    path_binary.write_text("#!/bin/sh\nexit 0\n")
    path_binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))

    binary = resolve_omnivoice_binary({}, home=home)

    assert binary == path_binary.resolve()


@pytest.mark.unit
def test_parse_resolves_input_output_and_edit_dir_to_absolute_paths(
    tmp_path: Path,
) -> None:
    input_video = tmp_path / "input.mp4"
    text_file = tmp_path / "script.txt"
    input_video.touch()
    text_file.write_text("A short narration.")

    request = parse_voiceover_args(
        [
            "--text-file",
            text_file.name,
            "--onto",
            input_video.name,
            "-o",
            "exports/final.mp4",
            "--edit-dir",
            "work",
        ],
        cwd=tmp_path,
    )

    assert request.text == "A short narration."
    assert request.video == input_video.resolve()
    assert request.output == (tmp_path / "work/exports/final.mp4").resolve()
    assert request.edit_dir == (tmp_path / "work").resolve()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("output_name", "is_valid"),
    [
        ("narration.wav", True),
        ("narration.WAV", True),
        ("narration.mp3", False),
        ("narration", False),
    ],
)
def test_parse_requires_wav_output_without_onto(
    tmp_path: Path, output_name: str, is_valid: bool
) -> None:
    arguments = ["--text", "Narration.", "-o", output_name]

    if is_valid:
        request = parse_voiceover_args(arguments, cwd=tmp_path)
        assert request.output == (tmp_path / output_name).resolve()
    else:
        with pytest.raises(VoiceoverError, match="output must use a .wav extension"):
            parse_voiceover_args(arguments, cwd=tmp_path)


@pytest.mark.unit
def test_parse_wraps_a_non_utf8_text_file(tmp_path: Path) -> None:
    text_file = tmp_path / "script.txt"
    text_file.write_bytes(b"Narration\xff")

    with pytest.raises(VoiceoverError, match="could not read text-file"):
        parse_voiceover_args(
            ["--text-file", text_file.name, "-o", "output.wav"],
            cwd=tmp_path,
        )


@pytest.mark.unit
def test_parse_rejects_an_edit_dir_that_is_not_a_directory(tmp_path: Path) -> None:
    edit_dir = tmp_path / "not-a-directory"
    edit_dir.write_text("not a directory")

    with pytest.raises(VoiceoverError, match="edit-dir is not a directory"):
        parse_voiceover_args(
            [
                "--text",
                "Narration.",
                "-o",
                "output.wav",
                "--edit-dir",
                edit_dir.name,
            ],
            cwd=tmp_path,
        )


@pytest.mark.unit
def test_parse_rejects_an_output_path_that_would_overwrite_input(
    tmp_path: Path,
) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.touch()

    with pytest.raises(VoiceoverError, match="output.*input"):
        parse_voiceover_args(
            ["--text", "Narration", "--onto", "input.mp4", "-o", "input.mp4"],
            cwd=tmp_path,
        )


@pytest.mark.unit
def test_mux_rejects_a_hard_link_to_the_input_video_before_execution(
    tmp_path: Path,
) -> None:
    video = tmp_path / "input.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "output.mp4"
    video.write_bytes(b"video source contents")
    narration.write_bytes(b"narration contents")
    os.link(video, output)
    source_hash = hashlib.sha256(video.read_bytes()).hexdigest()

    with pytest.raises(VoiceoverError, match="output.*onto video"):
        mux_voiceover(
            video,
            narration,
            output,
            mix=False,
            duck=0.15,
            vo_gain=1.0,
            offset=0.0,
        )

    assert hashlib.sha256(video.read_bytes()).hexdigest() == source_hash


@pytest.mark.unit
def test_mux_failure_preserves_existing_output_and_cleans_temporary_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "input.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "output.mp4"
    video.write_bytes(b"video source contents")
    narration.write_bytes(b"narration contents")
    output.write_bytes(b"existing output")
    monkeypatch.setattr(
        voiceover,
        "_probe_video",
        lambda _video: voiceover._VideoProbe(duration=1.0, has_audio=False),
    )
    monkeypatch.setattr(voiceover, "_probe_narration_duration", lambda _narration: 1.0)

    def fail_ffmpeg(
        executable: Path,
        arguments: Sequence[str],
        *,
        workdir_fd: int,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        if executable == Path("/bin/chmod"):
            return _completed_command(executable, arguments, text=text)
        raise subprocess.CalledProcessError(7, [str(executable), *arguments])

    monkeypatch.setattr(voiceover, "_run_in_workdir", fail_ffmpeg)

    with pytest.raises(VoiceoverError, match="ffmpeg failed"):
        mux_voiceover(
            video,
            narration,
            output,
            mix=False,
            duck=0.15,
            vo_gain=1.0,
            offset=0.0,
        )

    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(f"{voiceover._WORKDIR_PREFIX}*"))


@pytest.mark.unit
def test_parse_rejects_a_sidecar_hard_link_to_the_input_video(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    video.write_bytes(b"video source contents")
    os.link(video, output.with_suffix(".vo.wav"))

    with pytest.raises(VoiceoverError, match="output.*onto video"):
        parse_voiceover_args(
            ["--text", "Narration", "--onto", video.name, "-o", output.name],
            cwd=tmp_path,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            [
                "--text",
                "one",
                "--text-file",
                "script.txt",
                "--onto",
                "video.mp4",
                "-o",
                "out.mp4",
            ],
            "text",
        ),
        (
            [
                "--text",
                "one",
                "--ref-text",
                "reference words",
                "--onto",
                "video.mp4",
                "-o",
                "out.mp4",
            ],
            "ref-text",
        ),
        (
            [
                "--text",
                "one",
                "--ref-audio",
                "reference.wav",
                "--ref-text",
                "reference words",
                "--instruct",
                "warm",
                "--onto",
                "video.mp4",
                "-o",
                "out.mp4",
            ],
            "instruct",
        ),
        (
            [
                "--text",
                "one",
                "--duration",
                "0",
                "--onto",
                "video.mp4",
                "-o",
                "out.mp4",
            ],
            "duration",
        ),
        (
            ["--text", "one", "--speed", "nan", "--onto", "video.mp4", "-o", "out.mp4"],
            "speed",
        ),
        (
            [
                "--text",
                "one",
                "--offset",
                "-0.1",
                "--onto",
                "video.mp4",
                "-o",
                "out.mp4",
            ],
            "offset",
        ),
        (
            ["--text", "one", "--duck", "1.1", "--onto", "video.mp4", "-o", "out.mp4"],
            "duck",
        ),
        (
            [
                "--text",
                "one",
                "--vo-gain",
                "inf",
                "--onto",
                "video.mp4",
                "-o",
                "out.mp4",
            ],
            "vo-gain",
        ),
    ],
)
def test_parse_rejects_invalid_argument_combinations_and_numbers(
    tmp_path: Path, arguments: list[str], message: str
) -> None:
    (tmp_path / "script.txt").write_text("two")
    (tmp_path / "video.mp4").touch()
    (tmp_path / "reference.wav").touch()

    with pytest.raises(VoiceoverError, match=message):
        parse_voiceover_args(arguments, cwd=tmp_path)


@pytest.mark.unit
def test_parse_allows_reference_audio_without_reference_text(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    reference = tmp_path / "reference.wav"
    video.touch()
    reference.touch()

    request = parse_voiceover_args(
        [
            "--text",
            "Cloned narration.",
            "--onto",
            str(video),
            "--ref-audio",
            str(reference),
            "-o",
            "output.mp4",
        ],
        cwd=tmp_path,
    )

    assert request.ref_audio == reference.resolve()
    assert request.ref_text is None


@pytest.mark.unit
def test_parse_rejects_duration_and_speed_together(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()

    with pytest.raises(VoiceoverError, match="duration and speed"):
        parse_voiceover_args(
            [
                "--text",
                "Narration.",
                "--onto",
                str(video),
                "-o",
                "output.mp4",
                "--duration",
                "1.25",
                "--speed",
                "1.1",
            ],
            cwd=tmp_path,
        )


@pytest.mark.unit
def test_parse_allows_zero_ducking(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()

    request = parse_voiceover_args(
        [
            "--text",
            "Narration.",
            "--onto",
            str(video),
            "-o",
            "output.mp4",
            "--duck",
            "0",
        ],
        cwd=tmp_path,
    )

    assert request.duck == 0.0


@pytest.mark.unit
def test_build_omnivoice_command_uses_upstream_voice_cloning_flag_names(
    tmp_path: Path, executable: Path
) -> None:
    video = tmp_path / "video.mp4"
    reference = tmp_path / "reference.wav"
    video.touch()
    reference.touch()
    request = parse_voiceover_args(
        [
            "--text",
            "Cloned narration.",
            "--onto",
            str(video),
            "-o",
            str(tmp_path / "final.mp4"),
            "--ref-audio",
            str(reference),
            "--ref-text",
            "Reference transcript.",
            "--model",
            "local/omnivoice",
        ],
        cwd=tmp_path,
    )
    tts_output = tmp_path / "tts.wav"

    command = build_omnivoice_command(request, executable, tts_output)

    assert command == [
        str(executable.resolve()),
        "--model",
        "local/omnivoice",
        "--text",
        "Cloned narration.",
        "--output",
        str(tts_output.resolve()),
        "--ref_audio",
        str(reference.resolve()),
        "--ref_text",
        "Reference transcript.",
    ]


@pytest.mark.unit
def test_build_omnivoice_command_forwards_voice_design_controls(
    tmp_path: Path, executable: Path
) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    request = parse_voiceover_args(
        [
            "--text",
            "Designed narration.",
            "--onto",
            str(video),
            "-o",
            str(tmp_path / "final.mp4"),
            "--instruct",
            "Warm, quiet documentary tone",
            "--language",
            "en",
            "--duration",
            "1.25",
            "--device",
            "mps",
        ],
        cwd=tmp_path,
    )
    tts_output = tmp_path / "tts.wav"

    command = build_omnivoice_command(request, executable, tts_output)

    assert command == [
        str(executable.resolve()),
        "--model",
        "k2-fsa/OmniVoice",
        "--text",
        "Designed narration.",
        "--output",
        str(tts_output.resolve()),
        "--instruct",
        "Warm, quiet documentary tone",
        "--language",
        "en",
        "--duration",
        "1.25",
        "--device",
        "mps",
    ]


@pytest.mark.unit
def test_dry_run_writes_nothing_and_prints_tts_and_ffmpeg_plans(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    executable: Path,
) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.touch()
    edit_dir = tmp_path / "new-edit-dir"
    output = edit_dir / "exports" / "final.mp4"
    request = parse_voiceover_args(
        [
            "--text",
            "Narration only.",
            "--onto",
            str(input_video),
            "-o",
            "exports/final.mp4",
            "--edit-dir",
            str(edit_dir),
            "--dry-run",
        ],
        cwd=tmp_path,
    )
    monkeypatch.setattr(
        voiceover,
        "_probe_video",
        lambda _video: voiceover._VideoProbe(duration=1.0, has_audio=False),
    )

    plan = run_voiceover(
        request,
        environ={"OMNIVOICE_BIN": str(executable)},
        cwd=tmp_path,
        repo_dir=tmp_path,
    )

    captured = capsys.readouterr()
    assert plan.tts_command[0] == str(executable.resolve())
    assert plan.tts_command[-1] == str(output.with_suffix(".vo.wav"))
    assert plan.mux_command[0] == "ffmpeg"
    assert plan.mux_command[-1] == str(output)
    assert "TTS:" in captured.out
    assert "ffmpeg:" in captured.out
    assert not edit_dir.exists()
    assert not output.exists()
    assert not list(tmp_path.rglob("*.wav"))
    assert os.path.exists(input_video)


@pytest.mark.unit
def test_dry_run_propagates_video_probe_failure_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, executable: Path
) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.touch()
    edit_dir = tmp_path / "new-edit-dir"
    output = edit_dir / "exports" / "final.mp4"
    request = parse_voiceover_args(
        [
            "--text",
            "Narration only.",
            "--onto",
            str(input_video),
            "-o",
            "exports/final.mp4",
            "--edit-dir",
            str(edit_dir),
            "--dry-run",
        ],
        cwd=tmp_path,
    )

    def fail_probe(_video: Path) -> voiceover._VideoProbe:
        raise VoiceoverError("could not probe video")

    monkeypatch.setattr(voiceover, "_probe_video", fail_probe)

    with pytest.raises(VoiceoverError, match="could not probe video"):
        run_voiceover(
            request,
            environ={"OMNIVOICE_BIN": str(executable)},
            cwd=tmp_path,
            repo_dir=tmp_path,
        )

    assert not edit_dir.exists()
    assert not output.exists()
    assert not list(tmp_path.rglob("*.wav"))


@pytest.mark.unit
def test_run_voiceover_atomically_publishes_a_wav_output(
    tmp_path: Path, writing_executable: Path
) -> None:
    output = tmp_path / "narration.wav"
    output.write_bytes(b"previous narration")
    request = parse_voiceover_args(
        ["--text", "Narration only.", "-o", output.name],
        cwd=tmp_path,
    )

    plan = run_voiceover(
        request,
        environ={"OMNIVOICE_BIN": str(writing_executable)},
        cwd=tmp_path,
        repo_dir=tmp_path,
    )

    output_argument = plan.tts_command.index("--output") + 1
    assert plan.tts_command[output_argument] == str(output.resolve())
    assert plan.mux_command is None
    assert output.read_bytes().startswith(b"RIFF")
    assert not list(tmp_path.glob(f"{voiceover._WORKDIR_PREFIX}*"))


@pytest.mark.unit
def test_run_voiceover_wraps_tts_failure_and_cleans_temporary_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, executable: Path
) -> None:
    output = tmp_path / "narration.wav"
    output.write_bytes(b"existing narration")
    request = parse_voiceover_args(
        ["--text", "Narration only.", "-o", output.name],
        cwd=tmp_path,
    )

    def fail_tts(
        executable_path: Path,
        arguments: Sequence[str],
        *,
        workdir_fd: int,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        if executable_path == Path("/bin/chmod"):
            return _completed_command(executable_path, arguments, text=text)
        raise subprocess.CalledProcessError(
            7,
            [str(executable_path), *arguments],
        )

    monkeypatch.setattr(voiceover, "_run_in_workdir", fail_tts)

    with pytest.raises(VoiceoverError, match="OmniVoice generation failed"):
        run_voiceover(
            request,
            environ={"OMNIVOICE_BIN": str(executable)},
            cwd=tmp_path,
            repo_dir=tmp_path,
        )

    assert output.read_bytes() == b"existing narration"
    assert not list(tmp_path.glob(f"{voiceover._WORKDIR_PREFIX}*"))


@pytest.mark.unit
@pytest.mark.skipif(os.name == "nt", reason="requires POSIX directory permissions")
def test_run_voiceover_rejects_unsafe_output_directory_before_tts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, executable: Path
) -> None:
    output_directory = tmp_path / "unsafe-output"
    output_directory.mkdir()
    output = output_directory / "narration.wav"
    output.write_bytes(b"existing narration")
    request = parse_voiceover_args(
        ["--text", "Narration only.", "-o", str(output)],
        cwd=tmp_path,
    )

    def fail_if_tts_runs(command: list[str], *, check: bool) -> None:
        pytest.fail(f"unexpected external command: {command}")

    monkeypatch.setattr(voiceover.subprocess, "run", fail_if_tts_runs)
    output_directory.chmod(0o777)
    try:
        with pytest.raises(VoiceoverError, match="unsafe output directory"):
            run_voiceover(
                request,
                environ={"OMNIVOICE_BIN": str(executable)},
                cwd=tmp_path,
                repo_dir=tmp_path,
            )
    finally:
        output_directory.chmod(0o700)

    assert output.read_bytes() == b"existing narration"
    assert not list(output_directory.glob(f"{voiceover._WORKDIR_PREFIX}*"))


@pytest.mark.unit
@pytest.mark.skipif(os.name == "nt", reason="requires POSIX directory permissions")
def test_output_directory_check_allows_sticky_directory(tmp_path: Path) -> None:
    output_directory = tmp_path / "sticky-output"
    output_directory.mkdir()
    output_directory.chmod(0o1777)
    try:
        voiceover._assert_safe_output_directory(output_directory / "narration.wav")
    finally:
        output_directory.chmod(0o700)


@pytest.fixture
def valid_wav_executable(tmp_path: Path) -> Path:
    path = tmp_path / "valid-wav-omnivoice-infer"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "import wave\n"
        "output = sys.argv[sys.argv.index('--output') + 1]\n"
        "with wave.open(output, 'wb') as generated:\n"
        "    generated.setnchannels(1)\n"
        "    generated.setsampwidth(2)\n"
        "    generated.setframerate(8000)\n"
        "    generated.writeframes(b'\\x00\\x00' * 80)\n"
    )
    path.chmod(0o755)
    return path


@pytest.fixture
def corrupt_wav_executable(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt-wav-omnivoice-infer"
    path.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "--output" ]; then\n'
        "    output=$2\n"
        "    shift 2\n"
        "  else\n"
        "    shift\n"
        "  fi\n"
        "done\n"
        "printf 'this is not a WAV container' > \"$output\"\n"
    )
    path.chmod(0o755)
    return path


@pytest.mark.unit
def test_run_voiceover_preserves_existing_wav_when_tts_exits_successfully_with_corrupt_wav(
    tmp_path: Path, corrupt_wav_executable: Path
) -> None:
    output = tmp_path / "narration.wav"
    previous_output = b"previous valid narration"
    output.write_bytes(previous_output)
    request = parse_voiceover_args(
        ["--text", "Narration only.", "-o", output.name],
        cwd=tmp_path,
    )

    with pytest.raises(VoiceoverError):
        run_voiceover(
            request,
            environ={"OMNIVOICE_BIN": str(corrupt_wav_executable)},
            cwd=tmp_path,
            repo_dir=tmp_path,
        )

    assert output.read_bytes() == previous_output
    assert not list(tmp_path.glob(f"{voiceover._WORKDIR_PREFIX}*"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "output_name", ["narration.wav", "narration", "narration.webm"]
)
def test_parse_rejects_non_video_containers_for_video_output(
    tmp_path: Path, output_name: str
) -> None:
    video = tmp_path / "source.mp4"
    video.touch()

    with pytest.raises(VoiceoverError):
        parse_voiceover_args(
            ["--text", "Narration.", "--onto", video.name, "-o", output_name],
            cwd=tmp_path,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "output_name", ["final.mp4", "final.m4v", "final.mov", "final.mkv"]
)
def test_parse_allows_supported_video_containers_for_video_output(
    tmp_path: Path, output_name: str
) -> None:
    video = tmp_path / "source.mp4"
    video.touch()

    request = parse_voiceover_args(
        ["--text", "Narration.", "--onto", video.name, "-o", output_name],
        cwd=tmp_path,
    )

    assert request.output == (tmp_path / output_name).resolve()


@pytest.mark.unit
def test_mux_command_uses_container_specific_output_flags(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    narration = tmp_path / "narration.wav"
    probe = voiceover._VideoProbe(duration=1.0, has_audio=False)

    m4v_command = voiceover._build_mux_command(
        video,
        narration,
        tmp_path / "final.m4v",
        mix=False,
        duck=0.15,
        vo_gain=1.0,
        offset=0.0,
        probe=probe,
        narration_duration=0.5,
    )
    mkv_command = voiceover._build_mux_command(
        video,
        narration,
        tmp_path / "final.mkv",
        mix=False,
        duck=0.15,
        vo_gain=1.0,
        offset=0.0,
        probe=probe,
        narration_duration=0.5,
    )

    assert m4v_command[-3:-1] == ["-f", "mp4"]
    assert "-movflags" in m4v_command
    assert "-movflags" not in mkv_command


@pytest.mark.unit
def test_mux_command_trims_audio_that_starts_before_video_then_resets_pts(
    tmp_path: Path,
) -> None:
    command = voiceover._build_mux_command(
        tmp_path / "source.mp4",
        tmp_path / "narration.wav",
        tmp_path / "final.mp4",
        mix=True,
        duck=0.15,
        vo_gain=1.0,
        offset=0.0,
        probe=voiceover._VideoProbe(
            duration=1.0,
            has_audio=True,
            video_start_time=0.5,
            audio_start_time=0.0,
        ),
        narration_duration=0.5,
    )

    filter_complex = command[command.index("-filter_complex") + 1]
    assert "[0:a:0]atrim=start=0.5,asetpts=PTS-STARTPTS" in filter_complex


@pytest.mark.unit
def test_dry_run_mix_uses_requested_narration_duration_for_duck_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    executable: Path,
) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    request = parse_voiceover_args(
        [
            "--text",
            "Narration.",
            "--onto",
            video.name,
            "-o",
            "final.mp4",
            "--mix",
            "--offset",
            "0.15",
            "--duration",
            "0.4",
            "--dry-run",
        ],
        cwd=tmp_path,
    )
    monkeypatch.setattr(
        voiceover,
        "_probe_video",
        lambda _video: voiceover._VideoProbe(duration=1.0, has_audio=True),
    )

    plan = run_voiceover(
        request,
        environ={"OMNIVOICE_BIN": str(executable)},
        cwd=tmp_path,
        repo_dir=tmp_path,
    )

    assert plan.mux_command is not None
    filter_complex = plan.mux_command[plan.mux_command.index("-filter_complex") + 1]
    assert "between(t,0.15,0.55)" in filter_complex


@pytest.mark.unit
def test_probe_video_includes_bounded_ffprobe_stderr_in_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "broken.mp4"
    stderr = "malformed atom " + ("x" * 10_000) + " terminal marker"

    def fail_probe(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, command, stderr=stderr)

    monkeypatch.setattr(voiceover.subprocess, "run", fail_probe)

    with pytest.raises(VoiceoverError) as error_info:
        voiceover._probe_video(video)

    message = str(error_info.value)
    assert "malformed atom" in message
    assert "terminal marker" not in message
    assert len(message) < 2_000


@pytest.mark.unit
def test_probe_narration_includes_bounded_ffprobe_stderr_in_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    narration = tmp_path / "broken.wav"
    stderr = "invalid RIFF header " + ("x" * 10_000) + " terminal marker"

    def fail_probe(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, command, stderr=stderr)

    monkeypatch.setattr(voiceover.subprocess, "run", fail_probe)

    with pytest.raises(VoiceoverError) as error_info:
        voiceover._probe_narration_duration(narration)

    message = str(error_info.value)
    assert "invalid RIFF header" in message
    assert "terminal marker" not in message
    assert len(message) < 2_000


@pytest.mark.unit
def test_probe_video_uses_matroska_duration_tag_before_container_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "tagged.mkv"
    video.touch()
    calls: list[list[str]] = []
    probe_data = {
        "streams": [
            {
                "codec_type": "video",
                "start_time": "0.250",
                "tags": {"DURATION": "00:00:01.200000000"},
            },
            {"codec_type": "audio", "start_time": "0.500"},
        ],
        "format": {"duration": "4.000"},
    }

    def return_probe(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(probe_data),
            stderr="",
        )

    monkeypatch.setattr(
        voiceover, "_resolve_media_tool", lambda _name: Path("/usr/bin/true")
    )
    monkeypatch.setattr(voiceover.subprocess, "run", return_probe)

    probe = voiceover._probe_video(video)

    assert probe.duration == pytest.approx(1.2)
    assert probe.video_start_time == pytest.approx(0.25)
    assert probe.audio_start_time == pytest.approx(0.5)
    assert len(calls) == 1


@pytest.mark.unit
def test_probe_video_uses_packet_end_relative_to_video_stream_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "packet-duration.mkv"
    video.touch()
    responses = iter(
        [
            {
                "streams": [
                    {"codec_type": "video", "start_time": "5.000"},
                    {"codec_type": "audio", "start_time": "5.250"},
                ],
                "format": {"duration": "9.000"},
            },
            {
                "packets": [
                    {"pts_time": "5.000", "duration_time": "0.050"},
                    {"dts_time": "6.100", "duration_time": "0.100"},
                ]
            },
        ]
    )

    def return_probe(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(next(responses)),
            stderr="",
        )

    monkeypatch.setattr(
        voiceover, "_resolve_media_tool", lambda _name: Path("/usr/bin/true")
    )
    monkeypatch.setattr(voiceover.subprocess, "run", return_probe)

    probe = voiceover._probe_video(video)

    assert probe.duration == pytest.approx(1.2)
    assert probe.has_audio is True


@pytest.mark.unit
def test_probe_video_does_not_use_container_duration_when_audio_outlasts_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "unusable-packets.mkv"
    video.touch()
    responses = iter(
        [
            {
                "streams": [
                    {"codec_type": "video", "start_time": "0.000"},
                    {"codec_type": "audio", "start_time": "0.000"},
                ],
                "format": {"duration": "8.000"},
            },
            {"packets": []},
        ]
    )

    def return_probe(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(next(responses)),
            stderr="",
        )

    monkeypatch.setattr(
        voiceover, "_resolve_media_tool", lambda _name: Path("/usr/bin/true")
    )
    monkeypatch.setattr(voiceover.subprocess, "run", return_probe)

    with pytest.raises(VoiceoverError, match="video-stream duration"):
        voiceover._probe_video(video)


@pytest.mark.unit
def test_run_voiceover_uses_umask_derived_mode_for_new_wav_output(
    tmp_path: Path, valid_wav_executable: Path
) -> None:
    output = tmp_path / "narration.wav"
    request = parse_voiceover_args(
        ["--text", "Narration only.", "-o", output.name],
        cwd=tmp_path,
    )
    original_umask = os.umask(0o022)
    try:
        run_voiceover(
            request,
            environ={"OMNIVOICE_BIN": str(valid_wav_executable)},
            cwd=tmp_path,
            repo_dir=tmp_path,
        )
    finally:
        os.umask(original_umask)

    assert stat.S_IMODE(output.stat().st_mode) == 0o644


@pytest.mark.unit
def test_run_voiceover_preserves_existing_wav_output_mode(
    tmp_path: Path, valid_wav_executable: Path
) -> None:
    output = tmp_path / "narration.wav"
    output.write_bytes(b"previous narration")
    output.chmod(0o640)
    request = parse_voiceover_args(
        ["--text", "Narration only.", "-o", output.name],
        cwd=tmp_path,
    )

    run_voiceover(
        request,
        environ={"OMNIVOICE_BIN": str(valid_wav_executable)},
        cwd=tmp_path,
        repo_dir=tmp_path,
    )

    assert stat.S_IMODE(output.stat().st_mode) == 0o640


@pytest.mark.unit
@pytest.mark.skipif(os.name == "nt", reason="requires POSIX symlinks")
def test_run_voiceover_replaces_destination_symlink_without_writing_target(
    tmp_path: Path,
    valid_wav_executable: Path,
) -> None:
    target = tmp_path / "victim.wav"
    target.write_bytes(b"victim contents")
    output = tmp_path / "narration.wav"
    output.symlink_to(target)
    request = parse_voiceover_args(
        ["--text", "Narration only.", "-o", output.name],
        cwd=tmp_path,
    )

    plan = run_voiceover(
        request,
        environ={"OMNIVOICE_BIN": str(valid_wav_executable)},
        cwd=tmp_path,
        repo_dir=tmp_path,
    )

    output_argument = plan.tts_command.index("--output") + 1
    assert plan.tts_command[output_argument] == str(output)
    assert not output.is_symlink()
    assert output.read_bytes().startswith(b"RIFF")
    assert target.read_bytes() == b"victim contents"


@pytest.mark.unit
def test_mux_uses_umask_derived_mode_for_new_video_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "muxed.mp4"
    video.write_bytes(b"source video")
    narration.write_bytes(b"source narration")
    monkeypatch.setattr(
        voiceover,
        "_probe_video",
        lambda _video: voiceover._VideoProbe(duration=1.0, has_audio=False),
    )
    monkeypatch.setattr(voiceover, "_probe_narration_duration", lambda _narration: 0.5)

    def write_temporary_mux_output(
        executable_path: Path,
        arguments: Sequence[str],
        *,
        workdir_fd: int,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        if executable_path != Path("/bin/chmod"):
            _write_workdir_artifact(
                workdir_fd,
                arguments[-1],
                b"new muxed video",
            )
        return _completed_command(executable_path, arguments, text=text)

    monkeypatch.setattr(voiceover, "_run_in_workdir", write_temporary_mux_output)
    original_umask = os.umask(0o022)
    try:
        mux_voiceover(
            video,
            narration,
            output,
            mix=False,
            duck=0.15,
            vo_gain=1.0,
            offset=0.0,
        )
    finally:
        os.umask(original_umask)

    assert stat.S_IMODE(output.stat().st_mode) == 0o644


@pytest.mark.unit
def test_mux_preserves_existing_video_output_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mp4"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "muxed.mp4"
    video.write_bytes(b"source video")
    narration.write_bytes(b"source narration")
    output.write_bytes(b"previous muxed video")
    output.chmod(0o640)
    monkeypatch.setattr(
        voiceover,
        "_probe_video",
        lambda _video: voiceover._VideoProbe(duration=1.0, has_audio=False),
    )
    monkeypatch.setattr(voiceover, "_probe_narration_duration", lambda _narration: 0.5)

    def write_temporary_mux_output(
        executable_path: Path,
        arguments: Sequence[str],
        *,
        workdir_fd: int,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        if executable_path != Path("/bin/chmod"):
            _write_workdir_artifact(
                workdir_fd,
                arguments[-1],
                b"new muxed video",
            )
        return _completed_command(executable_path, arguments, text=text)

    monkeypatch.setattr(voiceover, "_run_in_workdir", write_temporary_mux_output)

    mux_voiceover(
        video,
        narration,
        output,
        mix=False,
        duck=0.15,
        vo_gain=1.0,
        offset=0.0,
    )

    assert stat.S_IMODE(output.stat().st_mode) == 0o640


def _race_output_parent_to_attacker_redirect(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sticky_parent: Path,
    attacker_redirect: Path,
) -> None:
    parked_parent = sticky_parent / "parked-safe-output"

    def swap_after_directory_fd_opened(
        checked_output: Path,
        output_dir_fd: int,
    ) -> None:
        assert stat.S_ISDIR(os.fstat(output_dir_fd).st_mode)
        checked_output.parent.rename(parked_parent)
        checked_output.parent.symlink_to(attacker_redirect, target_is_directory=True)

    monkeypatch.setattr(
        voiceover,
        "_output_directory_fd_opened",
        swap_after_directory_fd_opened,
    )


@pytest.mark.unit
@pytest.mark.skipif(
    os.name == "nt", reason="requires POSIX symlinks and sticky directories"
)
def test_tts_publish_does_not_follow_output_parent_symlink_swapped_after_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_wav_executable: Path,
) -> None:
    sticky_parent = tmp_path / "sticky"
    sticky_parent.mkdir(mode=0o1777)
    sticky_parent.chmod(0o1777)
    attacker_directory = tmp_path / "attacker"
    attacker_directory.mkdir()
    victim_directory = tmp_path / "victim"
    victim_directory.mkdir()
    victim_output = victim_directory / "narration.wav"
    victim_output.write_bytes(b"victim narration must remain unchanged")
    attacker_redirect = attacker_directory / "redirect"
    attacker_redirect.symlink_to(victim_directory, target_is_directory=True)
    output = sticky_parent / "missing-component" / "narration.wav"
    request = parse_voiceover_args(
        ["--text", "Narration only.", "-o", str(output)],
        cwd=tmp_path,
    )
    _race_output_parent_to_attacker_redirect(
        monkeypatch,
        sticky_parent=sticky_parent,
        attacker_redirect=attacker_redirect,
    )

    published = False
    try:
        run_voiceover(
            request,
            environ={"OMNIVOICE_BIN": str(valid_wav_executable)},
            cwd=tmp_path,
            repo_dir=tmp_path,
        )
        published = True
    except VoiceoverError:
        pass
    finally:
        if output.parent.is_symlink():
            output.parent.unlink()
        sticky_parent.chmod(0o700)

    assert not (attacker_directory / "narration.wav").exists()
    assert victim_output.read_bytes() == b"victim narration must remain unchanged"
    if published:
        assert (sticky_parent / "parked-safe-output" / "narration.wav").is_file()


@pytest.mark.unit
@pytest.mark.skipif(
    os.name == "nt", reason="requires POSIX symlinks and sticky directories"
)
def test_mux_publish_does_not_follow_output_parent_symlink_swapped_after_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sticky_parent = tmp_path / "sticky"
    sticky_parent.mkdir(mode=0o1777)
    sticky_parent.chmod(0o1777)
    attacker_directory = tmp_path / "attacker"
    attacker_directory.mkdir()
    victim_directory = tmp_path / "victim"
    victim_directory.mkdir()
    victim_output = victim_directory / "muxed.mp4"
    victim_output.write_bytes(b"victim video must remain unchanged")
    attacker_redirect = attacker_directory / "redirect"
    attacker_redirect.symlink_to(victim_directory, target_is_directory=True)
    output = sticky_parent / "missing-component" / "muxed.mp4"
    video = tmp_path / "source.mp4"
    narration = tmp_path / "narration.wav"
    video.write_bytes(b"source video")
    narration.write_bytes(b"source narration")
    monkeypatch.setattr(
        voiceover,
        "_probe_video",
        lambda _video: voiceover._VideoProbe(duration=1.0, has_audio=False),
    )
    monkeypatch.setattr(voiceover, "_probe_narration_duration", lambda _narration: 0.5)

    def write_temporary_mux_output(
        executable_path: Path,
        arguments: Sequence[str],
        *,
        workdir_fd: int,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        if executable_path != Path("/bin/chmod"):
            _write_workdir_artifact(
                workdir_fd,
                arguments[-1],
                b"new muxed video",
            )
        return _completed_command(executable_path, arguments, text=text)

    monkeypatch.setattr(voiceover, "_run_in_workdir", write_temporary_mux_output)
    _race_output_parent_to_attacker_redirect(
        monkeypatch,
        sticky_parent=sticky_parent,
        attacker_redirect=attacker_redirect,
    )

    published = False
    try:
        mux_voiceover(
            video,
            narration,
            output,
            mix=False,
            duck=0.15,
            vo_gain=1.0,
            offset=0.0,
        )
        published = True
    except VoiceoverError:
        pass
    finally:
        if output.parent.is_symlink():
            output.parent.unlink()
        sticky_parent.chmod(0o700)

    assert not (attacker_directory / "muxed.mp4").exists()
    assert victim_output.read_bytes() == b"victim video must remain unchanged"
    if published:
        assert (sticky_parent / "parked-safe-output" / "muxed.mp4").is_file()


@pytest.mark.integration
@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for media integration tests",
)
def test_run_voiceover_video_mode_publishes_valid_sidecar_and_muxed_video(
    tmp_path: Path,
    writing_executable: Path,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "published.mp4"
    sidecar = output.with_suffix(".vo.wav")
    subprocess.run(
        [
            str(shutil.which("ffmpeg")),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=24",
            "-t",
            "0.25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    output.write_bytes(b"previous invalid video")
    sidecar.write_bytes(b"previous invalid narration")
    request = parse_voiceover_args(
        ["--text", "Narration.", "--onto", source.name, "-o", output.name],
        cwd=tmp_path,
    )

    plan = run_voiceover(
        request,
        environ={"OMNIVOICE_BIN": str(writing_executable)},
        cwd=tmp_path,
        repo_dir=tmp_path,
    )

    assert plan.mux_command is not None
    assert plan.mux_command[0] == "ffmpeg"
    assert output.is_file()
    assert sidecar.is_file()
    assert voiceover._probe_video(output).has_audio is True
    assert voiceover._probe_narration_duration(sidecar) > 0
    assert not list(tmp_path.glob(f"{voiceover._WORKDIR_PREFIX}*"))
