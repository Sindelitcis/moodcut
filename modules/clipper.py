"""Cut highlight clips from a VOD using ffmpeg — single-pass NVENC encode
with frame-accurate seek and optional subtitle burn."""

import logging
import math
import re
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


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


def _find_keyframe(video_path: Path, start_sec: int) -> tuple[float, float]:
    """Find the closest keyframe <= start_sec via ffprobe.

    Probes the interval ``[start_sec-10, start_sec+0.1]`` for keyframes
    and returns the largest ``pts_time`` that does not exceed ``start_sec``.

    Returns:
        ``(keyframe_pts, residual)`` where ``residual = start_sec - keyframe_pts``.

    Falls back to rounding down to the nearest multiple of 2 if ffprobe
    fails or returns no usable keyframe.
    """
    interval_start = max(0.0, start_sec - 10.0)
    interval_end = start_sec + 0.1

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-skip_frame",
            "nokey",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "csv=p=0",
            "-read_intervals",
            f"{interval_start}%{interval_end}",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 and result.stdout.strip():
        keyframes: list[float] = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line:
                try:
                    pts = float(line)
                    if pts <= start_sec:
                        keyframes.append(pts)
                except ValueError:
                    continue

        if keyframes:
            best = max(keyframes)
            return best, start_sec - best

    # Fallback: round down to nearest multiple of 2
    fallback = math.floor(start_sec / 2) * 2
    return float(fallback), start_sec - fallback


def _build_srt_entries(
    transcript_segments: list[dict],
    clip_start_sec: int,
    clip_end_sec: int,
    keyframe_seek: float,
) -> list[str]:
    """Build a list of SRT entry strings for a given clip time range.

    Timestamps in the entries are relative to ``keyframe_seek`` (the
    actual start of the output video).  Word filtering, however, is
    still anchored to ``clip_start_sec`` (= ``padded_start``) so that
    no words from before the intended clip boundary appear.

    Returns an empty list when no segments overlap the range.

    For segments that start before ``clip_start_sec``, word-level
    alignment data (``seg["words"]``) is used to filter out words
    whose timestamp falls before the clip boundary.
    """
    entries: list[str] = []
    idx = 1
    for seg in transcript_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])

        # ── Word-level filtering for segments that begin before the clip ──
        if seg_start < clip_start_sec and "words" in seg and seg["words"]:
            filtered_words = [
                w for w in seg["words"] if w.get("start", 0) >= clip_start_sec
            ]
            if not filtered_words:
                continue
            seg_start = filtered_words[0]["start"]
            text = " ".join(w.get("word", "").strip() for w in filtered_words).strip()
        else:
            text = seg.get("text", "").strip()

        if not text:
            continue

        # Skip segments outside the clip window
        if seg_end <= clip_start_sec or seg_start >= clip_end_sec:
            continue

        # Clip to boundaries and make relative to keyframe_seek
        rel_start = max(0.0, seg_start - keyframe_seek)
        rel_end = min(clip_end_sec - keyframe_seek, seg_end - keyframe_seek)

        if rel_end - rel_start < 0.1:
            continue

        entries.append(
            f"{idx}\n{_format_srt_time(rel_start)} --> {_format_srt_time(rel_end)}\n{text}\n"
        )
        idx += 1

    return entries


def _run_ffmpeg(cmd: list[str], label: str, i: int, total: int) -> bool:
    """Run an FFmpeg command with logging and return success status."""
    logger.debug("FFmpeg command: %s", " ".join(cmd))
    logger.info("  [ffmpeg] %s clipe %d/%d", label, i, total)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "  [ERRO] FFmpeg falhou no clipe %d: %s",
            i,
            result.stderr.strip()[:300],
        )
        return False
    return True


def cut_clips(
    video_path: str,
    segments: list,
    transcript_segments: list[dict],
) -> list[str]:
    """Cut each approved segment from the video into individual clip files.

    Single-pass NVENC encode with frame-accurate seek and optional subtitle
    burn.  No intermediate files are created — the VOD is the only input and
    the final clip is the only output per segment.

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
    max_clip_duration = int(config.get("max_clip_duration", 0))

    # ── Get video duration ───────────────────────────────────────
    logger.info("  [ffprobe] Obtendo duração do vídeo...")
    video_duration = _get_video_duration(video_path)
    logger.info("  [ffprobe] Duração total: %.1fs", video_duration)

    generated: list[str] = []

    for i, seg in enumerate(segments, 1):
        inicio: str = seg.get("inicio", "")
        fim: str = seg.get("fim", "")
        humor: str = seg.get("humor", "outros")
        titulo: str = seg.get("titulo_sugerido", f"clip_{i}")

        # ── Validate timestamps ──────────────────────────────────
        if not inicio or not fim:
            logger.warning("  [AVISO] Segmento %d sem timestamps, pulando", i)
            continue

        raw_start = _parse_hms(inicio)
        raw_end = _parse_hms(fim)

        if raw_end <= raw_start:
            logger.warning(
                "  [AVISO] Duração inválida (%ds) para segmento %d, pulando",
                raw_end - raw_start,
                i,
            )
            continue

        # ── Max clip duration guard ──────────────────────────────
        raw_duration = raw_end - raw_start
        if max_clip_duration > 0 and raw_duration > max_clip_duration:
            logger.warning(
                "  [AVISO] Segmento %d dura %ds, excede max_clip_duration=%ds, pulando",
                i,
                raw_duration,
                max_clip_duration,
            )
            continue

        # ── Apply padding ────────────────────────────────────────
        padded_start = max(0, raw_start - padding_start)
        padded_end = min(int(video_duration), raw_end + padding_end)

        if padded_end <= padded_start:
            logger.warning(
                "  [AVISO] Duração após padding é zero para segmento %d, pulando",
                i,
            )
            continue

        # ── Build output paths ───────────────────────────────────
        out_dir = Path("output/clips")
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_title = _sanitize_filename(titulo)
        base = f"{i:02d}_{humor}_{safe_title}"
        final_path = out_dir / f"{base}.mp4"
        srt_path = out_dir / f"{base}.srt"

        # ── Cache check ──────────────────────────────────────────
        if final_path.exists():
            logger.info(
                "  [cache] Clipe já existe, pulando corte: %s",
                final_path,
            )
            generated.append(str(final_path.resolve()))
            continue

        # ── Frame-accurate seek parameters ───────────────────────
        # Probe the actual keyframe nearest to padded_start via ffprobe,
        # falling back to the multiple-of-2 approximation on failure.
        keyframe_seek, residual = _find_keyframe(video_path, padded_start)

        # Single seek at input: output starts at keyframe_seek.
        # Duration covers up to padded_end.
        duration = padded_end - keyframe_seek

        # ── Build FFmpeg command ─────────────────────────────────
        cmd = [
            "ffmpeg",
            "-ss",
            str(keyframe_seek),
            "-i",
            str(video_path),
            "-t",
            str(duration),
        ]

        # ── Generate subtitle file if applicable ─────────────────
        srt_entries = _build_srt_entries(
            transcript_segments, padded_start, padded_end, keyframe_seek
        )

        has_subs = bool(srt_entries)
        if has_subs:
            srt_path.write_text("\n".join(srt_entries), encoding="utf-8")
            srt_for_ffmpeg = str(srt_path).replace("\\", "/")
            cmd.extend(
                [
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
                ]
            )

        # ── Encoder settings ─────────────────────────────────────
        encoder_args = [
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
        cmd.extend(encoder_args)

        # ── Execute FFmpeg ───────────────────────────────────────
        label = f"Gravando legenda: {base}.mp4" if has_subs else f"Cortando: {base}.mp4"

        success = _run_ffmpeg(cmd, label, i, len(segments))

        # ── Fallback: NVENC → libx264 ────────────────────────────
        if not success:
            logger.warning(
                "  [AVISO] NVENC falhou no clipe %d, tentando libx264...",
                i,
            )
            fallback_encoder = [
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "fast",
                "-c:a",
                "copy",
                "-y",
                str(final_path),
            ]
            # Swap encoder_args block in-place
            fallback_cmd = cmd[: -len(encoder_args)] + fallback_encoder

            success = _run_ffmpeg(
                fallback_cmd,
                f"libx264 (fallback): {base}.mp4",
                i,
                len(segments),
            )

        # ── Cleanup subtitle file ────────────────────────────────
        if srt_path.exists():
            srt_path.unlink()

        if not success:
            # Remove any partial/broken output
            if final_path.exists():
                final_path.unlink()
            continue

        if not final_path.exists():
            logger.error(
                "  [ERRO] Arquivo final não gerado: %s",
                final_path,
            )
            continue

        logger.info("    -> %s  (%ds)", final_path, duration)
        generated.append(str(final_path.resolve()))

    return generated
