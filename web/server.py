"""Moodcut Web UI — FastAPI server.

Install dependencies:
    pip install fastapi uvicorn pyyaml --break-system-packages

Run from the project root:
    py -3.12 web/server.py

Then open: http://localhost:8000
"""

import asyncio
import os
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Project root is the parent of web/
ROOT = Path(__file__).parent.parent

app = FastAPI(title="Moodcut")

# ── Global pipeline state ──────────────────────────────────────────────────
_process: asyncio.subprocess.Process | None = None
_logs: list[str] = []
_running: bool = False
_stdin_pipe: bool = False        # True when process expects stdin (auth flow)
_current_channel: int = 0        # channel currently (or last) running

# ── Noise filter ────────────────────────────────────────────────────────────
# Library output that is always harmless and adds no value in the web terminal.
_NOISE = frozenset([
    "torchcodec is not installed correctly",
    "built-in audio decoding will fail",
    "use audio preloaded in-memory",
    "fix torchcodec installation",
    "Could not load libtorchcodec",
    "[start of libtorchcodec loading traceback]",
    "[end of libtorchcodec loading traceback]",
    "libtorchcodec_core",
    "FFmpeg version 4: Could not find module",
    "FFmpeg version 5: Could not find module",
    "FFmpeg version 6: Could not find module",
    "FFmpeg version 7: Could not find module",
    "Another runtime dependency; see exceptions below",
    "The following exceptions were raised as we tried to load",
    "constructor syntax.",
    "Lightning automatically upgraded your loaded checkpoint",
    "To apply the upgrade to your files permanently",
    "TensorFloat-32 (TF32) has been disabled",
    "reproducibility issues and lower accuracy",
    "It can be re-enabled by calling",
    ">>> import torch",
    ">>> torch.backends",
    "ReproducibilityWarning",
    "  warnings.warn(",
    "Likely causes:",
    "versions 4, 5, 6 and 7.",
    "PyTorch version (",
    "version compatibility",
])


def _is_noisy(line: str) -> bool:
    return any(s in line for s in _NOISE)


def _get_channel_cfg(channel: int) -> dict:
    """Return the yaml config dict for *channel*, or {} if not found."""
    try:
        with open(ROOT / "channels.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("canais", {}).get(channel, {})
    except Exception:
        return {}


def _check_password(channel: int, provided: str) -> str | None:
    """Return None if password is OK, or an error string if it is wrong."""
    expected = _get_channel_cfg(channel).get("senha", "")
    if expected and provided != expected:
        return "Senha incorreta para este canal."
    return None


# ── Models ─────────────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    url: str
    channel: int = 1
    skip_upload: bool = False
    skip_thumb: bool = False
    only_thumb: bool = False
    only_upload: bool = False
    password: str = ""


class AuthRequest(BaseModel):
    channel: int = 1


class StdinRequest(BaseModel):
    text: str


class StopRequest(BaseModel):
    channel: int = 0
    password: str = ""


# ── Routes ─────────────────────────────────────────────────────────────────


@app.get("/")
async def index():
    return FileResponse(
        Path(__file__).parent / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/channels")
async def api_channels():
    try:
        with open(ROOT / "channels.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("canais", {})
    except FileNotFoundError:
        return {}


@app.get("/api/status")
async def api_status():
    return {"running": _running, "log_lines": len(_logs), "channel": _current_channel}


@app.get("/api/channel-status")
async def api_channel_status():
    """Return auth token status for every channel in channels.yaml."""
    try:
        with open(ROOT / "channels.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        canais = data.get("canais", {})
    except FileNotFoundError:
        return []

    token_dir = ROOT / "auth"
    result = []
    for ch_id, ch_data in canais.items():
        token_path = token_dir / f"token_canal_{ch_id}.json"
        if not token_path.exists():
            status = "sem_token"
        else:
            try:
                import json as _json
                raw = _json.loads(token_path.read_text(encoding="utf-8"))
                # A refresh_token means credentials can self-renew
                if raw.get("refresh_token"):
                    status = "ok"
                elif raw.get("token"):
                    status = "sem_refresh"
                else:
                    status = "invalido"
            except Exception:
                status = "invalido"
        result.append({
            "id": ch_id,
            "nome": ch_data.get("nome", f"Canal {ch_id}"),
            "status": status,
            "has_password": bool(ch_data.get("senha")),
        })
    return result


@app.get("/api/logs")
async def api_logs(from_line: int = 0):
    return {
        "lines": _logs[from_line:],
        "total": len(_logs),
        "running": _running,
    }


@app.get("/api/thumbs")
async def api_thumbs():
    thumbs_dir = ROOT / "output" / "thumbs"
    if not thumbs_dir.exists():
        return []
    files = sorted(
        [f for f in thumbs_dir.glob("*.jpg") if not f.name.startswith(".")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return [f.name for f in files[:40]]


@app.get("/thumbs/{filename}")
async def serve_thumb(filename: str):
    if "/" in filename or "\\" in filename or not filename.endswith(".jpg"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = ROOT / "output" / "thumbs" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


@app.post("/api/run")
async def api_run(req: RunRequest):
    global _running, _logs, _stdin_pipe, _current_channel
    if _running:
        return {"error": "Pipeline já está rodando"}

    err = _check_password(req.channel, req.password)
    if err:
        return {"error": err}

    cmd = ["py", "-3.12", "main.py", req.url, f"--channel={req.channel}"]
    if req.skip_upload:
        cmd.append("--skip-upload")
    if req.skip_thumb:
        cmd.append("--skip-thumb")
    if req.only_thumb:
        cmd.append("--only-thumb")
    if req.only_upload:
        cmd.append("--only-upload")

    # Set state synchronously BEFORE creating the task so the SSE stream
    # can't race and send __DONE__ before the process starts.
    _running = True
    _stdin_pipe = False
    _current_channel = req.channel
    _logs = [f"$ {' '.join(cmd)}"]
    asyncio.create_task(_run_pipeline(cmd, with_stdin=False))
    return {"ok": True, "cmd": " ".join(cmd)}


@app.post("/api/auth")
async def api_auth(req: AuthRequest):
    global _running, _logs, _stdin_pipe
    if _running:
        return {"error": "Pipeline já está rodando"}

    cmd = ["py", "-3.12", "main.py", "--auth", f"--channel={req.channel}"]
    _running = True
    _stdin_pipe = True
    _logs = [f"$ {' '.join(cmd)}"]
    asyncio.create_task(_run_pipeline(cmd, with_stdin=True))
    return {"ok": True}


@app.post("/api/stdin")
async def api_stdin(req: StdinRequest):
    """Send a line of text to the running process's stdin (used for auth URL)."""
    global _process
    if not _process or _process.stdin is None:
        return {"error": "Sem processo ativo ou stdin não disponível"}
    try:
        _process.stdin.write((req.text + "\n").encode())
        await _process.stdin.drain()
        _logs.append(f"  [stdin] {req.text[:80]}")
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/stop")
async def api_stop(req: StopRequest = StopRequest()):
    global _process
    if req.channel:
        err = _check_password(req.channel, req.password)
        if err:
            return {"error": err}
    if _process and _process.returncode is None:
        _process.terminate()
        return {"ok": True}
    return {"error": "Nenhum processo ativo"}


@app.get("/api/stream")
async def api_stream(from_line: int = 0):
    """SSE endpoint — streams log lines from *from_line* onward."""

    async def generate():
        pos = from_line
        ticks = 0
        while True:
            while pos < len(_logs):
                line = _logs[pos].rstrip("\r\n")
                yield f"data: {line}\n\n"
                pos += 1
                ticks = 0

            if not _running and pos >= len(_logs):
                yield "data: __DONE__\n\n"
                return

            ticks += 1
            if ticks >= 20:
                yield ": keepalive\n\n"
                ticks = 0

            await asyncio.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Pipeline runner ─────────────────────────────────────────────────────────


async def _run_pipeline(cmd: list[str], *, with_stdin: bool = False) -> None:
    global _process, _running

    # Force unbuffered + UTF-8 output from the child Python process on Windows.
    # PYTHONUNBUFFERED=1 is equivalent to -u: disables block-buffering on pipes
    # so every print() appears immediately without waiting for the buffer to fill.
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        _process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE if with_stdin else None,
            cwd=str(ROOT),
            env=env,
        )
        assert _process.stdout is not None

        async for raw in _process.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not _is_noisy(line):
                _logs.append(line)

        await _process.wait()
        code = _process.returncode
        symbol = "✓" if code == 0 else "✗"
        _logs.append(f"{symbol} Processo encerrado (código {code})")
    except Exception as exc:  # pragma: no cover
        _logs.append(f"[ERRO INTERNO] {exc}")
    finally:
        _running = False


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  ▶  Moodcut Web UI")
    print("  →  http://localhost:8000")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
