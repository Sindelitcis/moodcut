"""Segment chunking logic — groups transcript segments into ~3-minute blocks."""

from __future__ import annotations

_CHUNK_DURATION = 180  # seconds (~3 minutes)


def _format_timestamp(seconds: float) -> str:
    """Convert a float in seconds to ``HH:MM:SS``."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def chunk_transcript(segments: list) -> list[str]:
    """Group transcript segments into blocks of approximately 3 minutes.

    Each block is a single string where every segment is formatted as::

        [HH:MM:SS] texto do segmento

    Args:
        segments: List of dicts with ``"start"``, ``"end"`` and ``"text"`` keys.

    Returns:
        List of chunk strings.
    """
    chunks: list[str] = []
    current_lines: list[str] = []
    chunk_start: float | None = None

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"].strip()

        if not seg_text:
            continue

        # First segment of a new chunk → initialise anchor
        if chunk_start is None:
            chunk_start = seg_start

        # If adding this segment would push the chunk past the boundary,
        # *and* there is already content in the chunk, finalise it.
        span = seg_end - chunk_start
        if span > _CHUNK_DURATION and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = []
            chunk_start = seg_start

        current_lines.append(f"[{_format_timestamp(seg_start)}] {seg_text}")

    # Flush the last chunk if any
    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks
