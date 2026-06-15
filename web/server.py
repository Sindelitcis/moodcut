"""Moodcut Web UI — FastAPI server.

Install dependencies:
    pip install fastapi uvicorn pyyaml --break-system-packages

Run from the project root:
    py -3.12 web/server.py

Then open: http://localhost:8000
"""

import asyncio
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


# ── Models ─────────────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    url: str
    channel: int = 1
    skip_upload: bool = False
    skip_thumb: bool = False
    only_thumb: bool = False
    only_upload: bool = False


# ── Routes ─────────────────────────────────────────────────────────────────


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


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
    return {"running": _running, "log_lines": len(_logs)}


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
    # Security: no path traversal, only .jpg
    if "/" in filename or "\\" in filename or not filename.endswith(".jpg"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = ROOT / "output" / "thumbs" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


@app.post("/api/run")
async def api_run(req: RunRequest):
    global _running
    if _running:
        return {"error": "Pipeline já está rodando"}

    cmd = ["py", "-3.12", "main.py", req.url, f"--channel={req.channel}"]
    if req.skip_upload:
        cmd.append("--skip-upload")
    if req.skip_thumb:
        cmd.append("--skip-thumb")
    if req.only_thumb:
        cmd.append("--only-thumb")
    if req.only_upload:
        cmd.append("--only-upload")

    asyncio.create_task(_run_pipeline(cmd))
    return {"ok": True, "cmd": " ".join(cmd)}


@app.post("/api/stop")
async def api_stop():
    global _process
    if _process and _process.returncode is None:
        _process.terminate()
        return {"ok": True}
    return {"error": "Nenhum processo ativo"}


@app.get("/api/stream")
async def api_stream(from_line: int = 0):
    """SSE endpoint — streams log lines from *from_line* onward.

    The client connects once and receives new lines as they are appended.
    When the pipeline finishes, a final ``__DONE__`` event is sent.
    Keepalive comments are sent every ~2 s so proxies don't drop the connection.
    """

    async def generate():
        pos = from_line
        ticks = 0
        while True:
            # Flush all available lines
            while pos < len(_logs):
                line = _logs[pos].rstrip("\r\n")
                yield f"data: {line}\n\n"
                pos += 1
                ticks = 0

            # End of stream
            if not _running and pos >= len(_logs):
                yield "data: __DONE__\n\n"
                return

            # Keepalive every ~2 s
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


async def _run_pipeline(cmd: list[str]) -> None:
    global _process, _logs, _running
    _running = True
    _logs = [f"$ {' '.join(cmd)}"]

    try:
        _process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(ROOT),
        )
        assert _process.stdout is not None

        async for raw in _process.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            _logs.append(line)

        await _process.wait()
        code = _process.returncode
        symbol = "✓" if code == 0 else "✗"
        _logs.append(f"{symbol} Pipeline encerrado (código {code})")
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
