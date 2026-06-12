"""Cut highlight clips from a VOD using ffmpeg."""

import re
import subprocess
import sys
from pathlib import Path

import yaml


def _parse_hms(hms: str) -> int:
    """Convert ``HH:MM:SS`` to total seconds."""
    parts = hms.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return int(parts[0])


def _format_hms(seconds: int) -> str:
    """Convert total seconds to ``HH:MM:SS``."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _sanitize_filename(name: str) -> str:
    """Strip special characters and truncate to 80 chars for a safe filename."""
    safe = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip()
    safe = re.sub(r"[-\s]+", "_", safe)
    return safe[:80] or "clip"


def _get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe falhou: {result.stderr.strip()}")
    return float(result.stdout.strip())


def cut_clips(video_path: str, segments: list) -> list[str]:
    """Cut each approved segment from the video into individual clip files.

    Uses ``ffmpeg -ss`` (fast keyframe seek) + ``-c copy`` (no re-encode).
    Padding values from ``config.yaml`` are applied to each segment's
    boundaries before cutting.

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

    # ── Load config ──────────────────────────────────────────────
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}"
        )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    padding_start = int(config.get("padding_start", 10))
    padding_end = int(config.get("padding_end", 8))

    # ── Get video duration ───────────────────────────────────────
    print(
        "  [ffprobe] Obtendo duração do vídeo...",
        file=sys.stderr,
        flush=True,
    )
    video_duration = _get_video_duration(video_path)
    print(
        f"  [ffprobe] Duração total: {video_duration:.1f}s",
        file=sys.stderr,
        flush=True,
    )

    generated: list[str] = []

    for i, seg in enumerate(segments, 1):
        inicio: str = seg.get("inicio", "")
        fim: str = seg.get("fim", "")
        humor: str = seg.get("humor", "outros")
        titulo: str = seg.get("titulo_sugerido", f"clip_{i}")

        # ── Validate timestamps ──────────────────────────────────
        if not inicio or not fim:
            print(
                f"  [AVISO] Segmento {i} sem timestamps, pulando",
                file=sys.stderr,
                flush=True,
            )
            continue

        raw_start = _parse_hms(inicio)
        raw_end = _parse_hms(fim)

        if raw_end <= raw_start:
            print(
                f"  [AVISO] Duração inválida ({raw_end - raw_start}s) "
                f"para segmento {i}, pulando",
                file=sys.stderr,
                flush=True,
            )
            continue

        # ── Apply padding ────────────────────────────────────────
        padded_start = max(0, raw_start - padding_start)
        padded_end = min(int(video_duration), raw_end + padding_end)
        duration = padded_end - padded_start

        if duration <= 0:
            print(
                f"  [AVISO] Duração após padding é zero para segmento {i}, pulando",
                file=sys.stderr,
                flush=True,
            )
            continue

        start_str = _format_hms(padded_start)
        dur_str = str(duration)

        # ── Build output path ────────────────────────────────────
        out_dir = Path("output") / humor
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_title = _sanitize_filename(titulo)
        out_path = out_dir / f"{i:02d}_{safe_title}.mp4"

        # ── ffmpeg: fast seek + stream copy ──────────────────────
        cmd = [
            "ffmpeg",
            "-ss",
            start_str,
            "-i",
            str(video_path),
            "-t",
            dur_str,
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-y",
            str(out_path),
        ]

        padding_info = f"[padding: -{padding_start}s / +{padding_end}s]"
        print(
            f"  [ffmpeg] Cortando clipe {i}/{len(segments)}: "
            f"{safe_title}.mp4 {padding_info} "
            f"(start {inicio}→{start_str}, {duration}s)",
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
