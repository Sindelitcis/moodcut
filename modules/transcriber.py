import json
import sys
from pathlib import Path

import torch
import whisper
import yaml


def transcribe(video_path: str) -> dict:
    """Transcribe a video file using Whisper and return segmented results.

    Args:
        video_path: Path to the video/audio file to transcribe.

    Returns:
        A dict with a ``"segments"`` key containing a list of
        ``{"start": float, "end": float, "text": str}``.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")

    # ── Load config ──────────────────────────────────────────────
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}"
        )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_name = config.get("whisper_model", "large-v3")

    # ── Cache path ───────────────────────────────────────────────
    output_dir = Path("output/transcriptions")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{video_path.stem}.json"

    # ── Return cached result if available ────────────────────────
    if json_path.exists():
        print(
            f"[*] Transcrição em cache: {json_path}",
            file=sys.stderr,
            flush=True,
        )
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    # ── Detect device ────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device.upper()}", file=sys.stderr, flush=True)
    print(
        f"[*] Carregando modelo Whisper '{model_name}'...",
        file=sys.stderr,
        flush=True,
    )

    model = whisper.load_model(model_name, device=device)

    # ── Transcribe ───────────────────────────────────────────────
    print(
        f"[*] Transcrevendo {video_path.name}...",
        file=sys.stderr,
        flush=True,
    )
    result = model.transcribe(str(video_path), verbose=True)

    # ── Build output ─────────────────────────────────────────────
    output = {
        "segments": [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            for seg in result["segments"]
        ]
    }

    # ── Save JSON ────────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] Transcrição salva em: {json_path}",
        file=sys.stderr,
        flush=True,
    )
    return output
