"""Opt-in end-to-end contract against a real local OmniVoice installation."""

from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path

import pytest

RUN_OMNIVOICE_TESTS = os.environ.get("FREECUT_RUN_OMNIVOICE_TESTS") == "1"
pytestmark = [
    pytest.mark.omnivoice,
    pytest.mark.skipif(
        not RUN_OMNIVOICE_TESTS,
        reason="set FREECUT_RUN_OMNIVOICE_TESTS=1 to exercise a real OmniVoice installation",
    ),
]

if RUN_OMNIVOICE_TESTS:
    from helpers.voiceover import (
        load_voiceover_environment,
        parse_voiceover_args,
        resolve_omnivoice_binary,
        run_voiceover,
    )


@pytest.mark.omnivoice
def test_real_omnivoice_cli_supports_wrapper_contract_and_generates_short_wav(
    tmp_path: Path,
) -> None:
    environment = load_voiceover_environment(
        environ=os.environ,
        cwd=Path.cwd(),
        repo_dir=Path(__file__).resolve().parents[1],
    )
    binary = resolve_omnivoice_binary(environment, home=Path.home())
    help_result = subprocess.run(
        [str(binary), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for option in (
        "--model",
        "--text",
        "--output",
        "--duration",
        "--speed",
        "--device",
    ):
        assert option in help_result.stdout

    output = tmp_path / "omnivoice-smoke.wav"
    request = parse_voiceover_args(
        [
            "--text",
            "A brief test.",
            "-o",
            str(output),
            "--duration",
            "0.5",
            "--model",
            environment.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice"),
        ],
        cwd=tmp_path,
    )
    execution_environment = dict(environment)
    execution_environment["OMNIVOICE_BIN"] = str(binary)

    run_voiceover(
        request,
        environ=execution_environment,
        cwd=tmp_path,
        repo_dir=Path(__file__).resolve().parents[1],
    )

    assert output.exists()
    with wave.open(str(output)) as generated:
        assert generated.getnframes() > 0
