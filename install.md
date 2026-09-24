---
name: freecut-install
description: Install freecut into the current agent (Claude Code, Codex, Hermes, Openclaw, etc.) and wire up ffmpeg + a local Whisper backend so the user can start editing immediately. No paid API keys required.
---

# freecut install

Use this file only for first-time install or reconnect. For daily editing, read `SKILL.md`. Always read `helpers/` — that's where the scripts live.

## What you're doing

You're setting up a conversation-driven video editor for the user. After install, the user drops raw footage into any folder, runs their agent (`claude`, `codex`, etc.) there, and says "edit these into a launch video." You do the rest by reading `SKILL.md`.

Three things must exist on this machine:

1. The `freecut` repo cloned somewhere stable.
2. `ffmpeg` on `$PATH` (plus optional `yt-dlp` for online sources).
3. A local Whisper backend installed (`mlx-whisper` on Apple Silicon, `faster-whisper` elsewhere).

And one thing must be true about the current agent:

4. It can discover `SKILL.md` — either via a global skills directory (`~/.claude/skills/`, `~/.codex/skills/`) or via a `CLAUDE.md` / system-prompt import.

**No `.env` file is required for the default backend.** `.env` only matters if the user wants the optional VibeVoice or ElevenLabs backends (see step 6).

## Install prompt contract

- Do everything yourself. Only ask the user for things you cannot generate — confirmation before `brew install`, or an API key/URL if they explicitly want VibeVoice or ElevenLabs.
- Prefer a stable clone path like `~/Developer/freecut` (not `/tmp`, not `~/Downloads`).
- The skill references helpers by bare name (`transcribe.py`, `render.py`). That works because SKILL.md and `helpers/` ship together — keep them as siblings when you register the skill.
- After install, verify by running one real command. Don't declare success on file-existence checks alone.

## Steps

### 1. Clone to a stable path

```bash
test -d ~/Developer/freecut || git clone https://github.com/Moh4696/freecut ~/Developer/freecut
cd ~/Developer/freecut
```

If the repo is already there, `git pull --ff-only` and continue.

### 2. Install Python deps

```bash
# Prefer uv if available; fall back to pip.
command -v uv >/dev/null && uv sync || pip install -e .
```

`pyproject.toml` lists `requests`, `librosa`, `matplotlib`, `pillow`, `numpy`. No console scripts — helpers are invoked directly as `python helpers/<name>.py`.

### 3. Install a local Whisper backend

freecut's default transcription backend runs locally. Pick one:

```bash
# Apple Silicon — recommended, fastest.
python -c 'import platform,sys; sys.exit(0 if platform.system()=="Darwin" and platform.machine()=="arm64" else 1)' \
  && (command -v uv >/dev/null && uv pip install mlx-whisper || pip install mlx-whisper)

# Everywhere else (Linux, Windows, Intel Mac, CPU, NVIDIA):
# command -v uv >/dev/null && uv pip install faster-whisper || pip install faster-whisper
```

The first real transcription will download the model weights on demand (default: `small`, ~500 MB).

### 4. Install ffmpeg (+ optional yt-dlp)

`ffmpeg` and `ffprobe` are hard requirements. `yt-dlp` is only needed if the user wants to pull sources from URLs. Animation engines such as HyperFrames, Remotion, and Manim are installed lazily the first time a project actually needs them.

```bash
# macOS
command -v ffmpeg >/dev/null || brew install ffmpeg
command -v yt-dlp >/dev/null || brew install yt-dlp     # optional

# Debian / Ubuntu
# sudo apt-get update && sudo apt-get install -y ffmpeg
# pip install yt-dlp

# Arch
# sudo pacman -S ffmpeg yt-dlp
```

If `brew` / `apt` / `pacman` requires a sudo prompt, tell the user the exact command and wait. Do not invent a password.

### 5. Register the skill with the current agent

Figure out which agent you are running under, and register once. A symlink of the whole repo directory is the right shape — helpers/ needs to sit next to SKILL.md.

- **Claude Code** (`~/.claude/` present):

    ```bash
    mkdir -p ~/.claude/skills
    ln -sfn ~/Developer/freecut ~/.claude/skills/freecut
    ```

- **Codex** (`$CODEX_HOME` set, or `~/.codex/` present):

    ```bash
    mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
    ln -sfn ~/Developer/freecut "${CODEX_HOME:-$HOME/.codex}/skills/freecut"
    ```

- **Hermes / Openclaw / another agent with a skills directory**: symlink `~/Developer/freecut` into that agent's skills directory under the name `freecut`. If the agent has no skills directory, add a line to its system prompt / config pointing at `~/Developer/freecut/SKILL.md` (e.g. an `@~/Developer/freecut/SKILL.md` import in a `CLAUDE.md`-equivalent).

If you can't tell which agent you're in, ask the user once: "which agent am I running under — Claude Code, Codex, or something else?" Then pick the right target.

### 6. Optional backends (skip unless the user asks)

The default `whisper` backend needs no keys. Only run these steps if the user explicitly wants richer diarization or the original ElevenLabs path.

**Optional — VibeVoice-ASR (multi-speaker diarization, requires a GPU endpoint):**

VibeVoice-ASR is CUDA-only, so freecut talks to it over HTTP. The user provides a URL pointing at a rented GPU, a Modal/RunPod deploy, or Azure AI Foundry. Write it to `.env`:

```bash
cp ~/Developer/freecut/.env.example ~/Developer/freecut/.env
$EDITOR ~/Developer/freecut/.env       # VIBEVOICE_ASR_URL=https://…
chmod 600 ~/Developer/freecut/.env
```

Then batch-transcribe with `--backend vibevoice`.

**Optional — FunASR server (multi-speaker diarization, self-hosted):**

If the user runs a FunASR server with its OpenAI-compatible API, write its base URL to `.env`:

```bash
cp ~/Developer/freecut/.env.example ~/Developer/freecut/.env
$EDITOR ~/Developer/freecut/.env       # FUN_ASR_URL=http://192.168.50.79:5078
chmod 600 ~/Developer/freecut/.env
```

Then batch-transcribe with `--backend funasr`. The URL may be a bare host, a `/v1` base, or the full `/v1/audio/transcriptions` path.

**Optional — ElevenLabs Scribe (paid, original video-use backend):**

```bash
cp ~/Developer/freecut/.env.example ~/Developer/freecut/.env
printf 'ELEVENLABS_API_KEY=%s\n' "$KEY" >> ~/Developer/freecut/.env
chmod 600 ~/Developer/freecut/.env
```

Never echo the key back in tool output. Never commit `.env`.

### 7. Verify end-to-end

Run one real thing. Prefer the lightest verification that still proves the pipeline is wired up:

```bash
python ~/Developer/freecut/helpers/timeline_view.py --help >/dev/null && echo "helpers OK"
python ~/Developer/freecut/helpers/transcribe.py --help >/dev/null && echo "transcribe OK"
ffprobe -version | head -1
```

Full transcription test is optional at install time — even the free backend takes a few seconds and downloads a model. Better to wait until the user hands you their first clip.

### 8. Hand off

Tell the user, in one short message:

- Where the skill is installed (`~/Developer/freecut`).
- That they should `cd` into their footage folder and start their agent there (e.g. `claude`).
- That a good first message is: *"edit these into a launch video"* or *"inventory these takes and propose a strategy."*
- That all outputs land in `<videos_dir>/edit/` — the repo stays clean.
- (If relevant) that transcription is single-speaker by default; if they need speaker diarization, mention the `vibevoice` backend and the endpoint step.

## Keeping the skill current

- `cd ~/Developer/freecut && git pull --ff-only` pulls the latest code. The symlink auto-picks it up on the next run.
- If `pyproject.toml` changed deps, re-run `uv sync` / `pip install -e .` after pulling.

## Cold-start reminders

- Symlink the **whole directory**, not just `SKILL.md`. The helpers need to sit next to it.
- The default backend is `whisper`. No `.env` is needed. Don't nag the user for an API key at install.
- If `.env` exists but a required key/URL is empty for the backend the user picked, treat it the same as missing.
- `ffmpeg` from static builds works fine. Any modern (≥ 4.x) build is enough.
- `yt-dlp` is optional. Don't block install on it; install lazily the first time a user asks to pull from a URL.
- Node.js/npm are only needed for HyperFrames or Remotion slots. HyperFrames currently requires Node.js 22+.
- HyperFrames, Remotion, and Manim are optional animation engines. Don't install or prefer one globally during setup; pick the engine per animation slot in `SKILL.md`. HyperFrames can run through `npx --yes hyperframes ...` in the slot directory. Remotion can be scaffolded with `npx create-video@latest` or installed inside the slot before rendering.
- If the user is on Linux without a package manager Claude recognizes, print the manual `ffmpeg` install URL and wait rather than guessing.
