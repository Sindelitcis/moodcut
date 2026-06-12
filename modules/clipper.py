"""Cut highlight clips from a VOD using ffmpeg and burn subtitles."""

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


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp ``HH:MM:SS,mmm``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


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


def _build_srt_entries(
    transcript_segments: list[dict],
    clip_start_sec: int,
    clip_end_sec: int,
) -> list[str]:
    """Build a list of SRT entry strings for a given clip time range.

    Timestamps in the entries are relative to ``clip_start_sec``.
    Returns an empty list when no segments overlap the range.
    """
    entries: list[str] = []
    idx = 1
    for seg in transcript_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Skip segments outside the clip window
        if seg_end <= clip_start_sec or seg_start >= clip_end_sec:
            continue

        # Clip to boundaries and make relative
        rel_start = max(0.0, seg_start - clip_start_sec)
        rel_end = min(clip_end_sec - clip_start_sec, seg_end - clip_start_sec)

        if rel_end - rel_start < 0.1:
            continue

        entries.append(
            f"{idx}\n{_format_srt_time(rel_start)} --> {_format_srt_time(rel_end)}\n{text}\n"
        )
        idx += 1

    return entries


def cut_clips(
    video_path: str,
    segments: list,
    transcript_segments: list[dict],
) -> list[str]:
    """Cut each approved segment from the video into individual clip files.

    Each clip is first extracted with fast stream copy, then subtitles from
    the transcription are burned in using ``-vf subtitles=`` (white font,
    black outline, size 18, bottom-centre aligned).

    Args:
        video_path: Path to the downloaded VOD.
        segments: List of dicts with ``"inicio"``, ``"fim"``, ``"humor"``,
            and ``"titulo_sugerido"`` keys.
        transcript_segments: Raw transcript segments from the transcriber
            (``{"start": float, "end": float, "text": str}``).

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

        # ── Build output paths ───────────────────────────────────
        out_dir = Path("output") / humor
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_title = _sanitize_filename(titulo)
        base = f"{i:02d}_{safe_title}"
        temp_path = out_dir / f"{base}.tmp.mp4"
        final_path = out_dir / f"{base}.mp4"
        srt_path = out_dir / f"{base}.srt"

        # ── Cache check ──────────────────────────────────────────────
        if final_path.exists():
            print(
                f"  [cache] Clipe já existe, pulando corte: {final_path}",
                file=sys.stderr,
                flush=True,
            )
            generated.append(str(final_path.resolve()))
            continue

        # ── Step 1: cut with stream copy → temp file ─────────────
        cmd_cut = [
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
            str(temp_path),
        ]

        padding_info = f"[padding: -{padding_start}s / +{padding_end}s]"
        print(
            f"  [ffmpeg] Cortando clipe {i}/{len(segments)}: "
            f"{safe_title}.mp4 {padding_info} "
            f"(start {inicio}→{start_str}, {duration}s)",
            file=sys.stderr,
            flush=True,
        )

        result = subprocess.run(cmd_cut, capture_output=True, text=True)

        if result.returncode != 0:
            print(
                f"  [ERRO] ffmpeg (corte) falhou no clipe {i}: "
                f"{result.stderr.strip()[:200]}",
                file=sys.stderr,
                flush=True,
            )
            continue

        if not temp_path.exists():
            print(
                f"  [ERRO] Arquivo temporário não gerado: {temp_path}",
                file=sys.stderr,
                flush=True,
            )
            continue

        # ── Step 2: generate SRT ─────────────────────────────────
        srt_entries = _build_srt_entries(transcript_segments, padded_start, padded_end)

        if srt_entries:
            srt_path.write_text("\n".join(srt_entries), encoding="utf-8")

            # Use forward slashes so ffmpeg's subtitles filter parses
            # the path correctly even on Windows.
            srt_for_ffmpeg = str(srt_path).replace("\\", "/")

            # ── Step 3: burn subtitles into final file ───────────
            cmd_burn = [
                "ffmpeg",
                "-i",
                str(temp_path),
                "-vf",
                f"subtitles={srt_for_ffmpeg}:force_style="
                f"'FontSize=14,"
                f"PrimaryColour=&H00FFFFFF,"
                f"BackColour=&H80000000,"
                f"BorderStyle=4,"
                f"Outline=0,"
                f"Shadow=0,"
                f"Alignment=2,"
                f"MarginV=20'",
                "-c:v",
                "h264_nvenc",
                "-cq",
                "18",
                "-preset",
                "p4",
                "-c:a",
                "copy",
                "-y",
                str(final_path),
            ]

            print(
                f"  [ffmpeg] Gravando legendas: {base}.mp4",
                file=sys.stderr,
                flush=True,
            )

            result_burn = subprocess.run(cmd_burn, capture_output=True, text=True)

            if result_burn.returncode != 0:
                print(
                    f"  [ERRO] ffmpeg (legendas) falhou no clipe {i}: "
                    f"{result_burn.stderr.strip()[:200]}",
                    file=sys.stderr,
                    flush=True,
                )
                # Fall back to temp clip (no subs) rather than losing the clip
                temp_path.rename(final_path)
            else:
                # Clean up temp + SRT
                if srt_path.exists():
                    srt_path.unlink()
        else:
            # No subtitles to burn — just rename temp to final
            temp_path.rename(final_path)

        # Clean up temp file if it still exists
        if temp_path.exists():
            temp_path.unlink()

        if not final_path.exists():
            print(
                f"  [ERRO] Arquivo final não gerado: {final_path}",
                file=sys.stderr,
                flush=True,
            )
            continue

        print(
            f"    -> {final_path}  ({duration}s)",
            file=sys.stderr,
            flush=True,
        )
        generated.append(str(final_path.resolve()))

    return generated
