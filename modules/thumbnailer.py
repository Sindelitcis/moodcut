"""Generate thumbnails for clips by extracting frames from the VOD (no subtitles)."""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat


def generate_thumbnail(
    clip_path: str,
    titulo: str,
    video_path: str,
    start_sec: int,
) -> str:
    """Generate a thumbnail for the given clip.

    Extracts frames from the *original VOD* (clean, no subtitles) around
    *start_sec*, picks the one with highest pixel variance, and overlays
    the title with a drop-shadow effect.

    Args:
        clip_path: Absolute path of the clip ``.mp4`` file (used for output name).
        titulo: Title text to overlay on the thumbnail.
        video_path: Absolute path of the original downloaded VOD.
        start_sec: Timestamp (in seconds) of the highlight start in the VOD.

    Returns:
        Absolute path of the generated thumbnail (``.jpg``).
    """
    clip_path = Path(clip_path)
    thumb_dir = Path("output/thumbs")
    thumb_dir.mkdir(parents=True, exist_ok=True)

    thumb_path = thumb_dir / f"{clip_path.stem}.jpg"

    if thumb_path.exists():
        print(
            f"  [cache] Thumbnail já existe, pulando: {thumb_path}",
            file=sys.stderr,
            flush=True,
        )
        return str(thumb_path.resolve())

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"VOD não encontrado: {video_path}")

    with tempfile.TemporaryDirectory(dir=thumb_dir) as tmp_dir:
        tmp = Path(tmp_dir)

        # ── Extract 10 frames from the VOD (1 every 6s) ───────
        sample_start = start_sec + 180  # pula os primeiros 3 minutos do segmento
        durations = [str(sample_start + offset) for offset in range(0, 60, 6)]
        frame_files: list[Path] = []
        for i, ts in enumerate(durations):
            out = tmp / f"frame_{i:04d}.jpg"
            subprocess.run(
                [
                    "ffmpeg",
                    "-ss",
                    ts,
                    "-i",
                    str(video_path),
                    "-vframes",
                    "1",
                    "-q:v",
                    "2",
                    "-y",
                    str(out),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            if out.exists():
                frame_files.append(out)

        if not frame_files:
            raise RuntimeError("Nenhum frame extraído do VOD")

        # ── Score frames by average pixel stddev ─────────────
        best_frame: Path | None = None
        best_score = -1.0
        for f in frame_files:
            img = Image.open(f).convert("RGB")
            stat = ImageStat.Stat(img)
            avg_stddev = sum(stat.stddev[:3]) / 3
            if avg_stddev > best_score:
                best_score = avg_stddev
                best_frame = f

        if best_frame is None:
            raise RuntimeError("Nenhum frame válido encontrado")

        # ── Open best frame (original colours, no overlay) ────
        base = Image.open(best_frame).convert("RGB")
        draw = ImageDraw.Draw(base)
        width, height = base.size

        # ── Font ─────────────────────────────────────────────
        font_size = max(60, int(width * 0.07))
        font = _load_rounded_font(font_size)

        # ── Wrap text ────────────────────────────────────────
        lines = textwrap.wrap(titulo, width=16)
        if not lines:
            lines = [titulo]

        # ── Calculate text block dimensions ──────────────────
        line_h = _text_extent(draw, "Ag", font)[1]
        line_spacing = int(font_size * 1.2)
        block_h = len(lines) * line_spacing - (line_spacing - line_h)

        # ── Position: centred vertically ─────────────────────
        total_text_height = len(lines) * (font_size * 1.2)
        text_block_y = int((height - total_text_height) // 2)

        # ── Drop shadow pass ─────────────────────────────────
        shadow_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        y = text_block_y
        for line in lines:
            line_w = _text_extent(draw, line, font)[0]
            x = (width - line_w) // 2
            shadow_draw.text((x, y), line, fill="#000000", font=font)
            y += line_spacing
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=12))
        base = Image.alpha_composite(base.convert("RGBA"), shadow_layer).convert("RGB")
        draw = ImageDraw.Draw(base)

        # ── Main text pass ───────────────────────────────────
        y = text_block_y
        for line in lines:
            line_w = _text_extent(draw, line, font)[0]
            x = (width - line_w) // 2
            draw.text(
                (x, y),
                line,
                fill="#FFE600",
                font=font,
                stroke_width=4,
                stroke_fill="#000000",
            )
            y += line_spacing

        # ── Save ─────────────────────────────────────────────
        base.save(thumb_path, "JPEG", quality=95)

    return str(thumb_path.resolve())


def _load_rounded_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a rounded/bold font at *size*, searching recursively."""
    candidates = [
        "Nunito-Bold.ttf",
        "Baloo2-Bold.ttf",
        "Comfortaa-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    search_dirs = [
        Path("C:/Windows/Fonts"),
        Path.home() / ".fonts",
        Path("/usr/share/fonts"),
    ]
    for sd in search_dirs:
        if not sd.exists():
            continue
        for ttf in sd.rglob("*.ttf"):
            if ttf.name in candidates:
                return ImageFont.truetype(str(ttf), size)
    return ImageFont.load_default()


def _text_extent(draw: ImageDraw, text: str, font) -> tuple[int, int]:
    """Return ``(width, height)`` of *text* when rendered with *font*."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]
