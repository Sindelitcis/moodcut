"""Cut highlight clips from a VOD using ffmpeg."""

import re
import subprocess
import sys
from pathlib import Path


def _parse_hms(hms: str) -> int:
    """Convert ``HH:MM:SS`` to total seconds."""
    parts = hms.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return int(parts[0])


def _sanitize_filename(name: str) -> str:
    """Strip special characters and truncate to 80 chars for a safe filename."""
    safe = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip()
    safe = re.sub(r"[-\s]+", "_", safe)
    return safe[:80] or "clip"


def cut_clips(video_path: str, segments: list) -> list[str]:
    """Cut each approved segment from the video into individual clip files.

    Uses ``ffmpeg -ss`` (fast keyframe seek) + ``-c copy`` (no re-encode).

    Args:
        video_path: Path to the downloaded VOD.
        segments: List of dicts with ``"inicio"``, ``"fim"``, ``"humor"``,
            and ``"titulo_sugerido"`` keys.

    Returns:
        List of absolute paths to the generated clip files.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")

    generated: list[str] = []

    for i, seg in enumerate(segments, 1):
        inicio: str = seg.get("inicio", "")
        fim: str = seg.get("fim", "")
        humor: str = seg.get("humor", "outros")
        titulo: str = seg.get("titulo_sugerido", f"clip_{i}")

        # ── Validate / compute duration ──────────────────────────
        if not inicio or not fim:
            print(
                f"  [AVISO] Segmento {i} sem timestamps, pulando",
                file=sys.stderr,
                flush=True,
            )
            continue

        start_sec = _parse_hms(inicio)
        end_sec = _parse_hms(fim)
        duration = end_sec - start_sec

        if duration <= 0:
            print(
                f"  [AVISO] Duração inválida ({duration}s) para segmento {i}, pulando",
                file=sys.stderr,
                flush=True,
            )
            continue

        # ── Build output path ────────────────────────────────────
        out_dir = Path("output") / humor
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_title = _sanitize_filename(titulo)
        out_path = out_dir / f"{i:02d}_{safe_title}.mp4"

        # ── ffmpeg: fast seek + stream copy ──────────────────────
        cmd = [
            "ffmpeg",
            "-ss",
            inicio,
            "-i",
            str(video_path),
            "-t",
            str(duration),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-y",
            str(out_path),
        ]

        print(
            f"  [ffmpeg] Cortando clipe {i}/{len(segments)}: {safe_title}.mp4",
            file=sys.stderr,
            flush=True,
        )

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(
                f"  [ERRO] ffmpeg falhou no clipe {i}: {result.stderr.strip()[:200]}",
                file=sys.stderr,
                flush=True,
            )
            continue

        if not out_path.exists():
            print(
                f"  [ERRO] Arquivo não gerado: {out_path}",
                file=sys.stderr,
                flush=True,
            )
            continue

        print(
            f"    -> {out_path}  ({duration}s)",
            file=sys.stderr,
            flush=True,
        )
        generated.append(str(out_path.resolve()))

    return generated
