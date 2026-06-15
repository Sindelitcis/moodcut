"""Generate thumbnails for clips — v2 pipeline.

Pipeline stages (each is optional, with graceful fallback):
  1. Frame selection by expression (mediapipe face detection + variance heuristic)
  2. Background removal with BEN2
  3. Artistic composition (canvas 1280×720, Pillow)
  4. Ultimate fallback: v1 variance-based overlay logic
"""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

# ── Config keys ───────────────────────────────────────────────────────────
_CONFIG_PATH = Path("config.yaml")
_THUMB_DIR = Path("output/thumbs")
_CACHE_DIR = _THUMB_DIR / ".cache"
_CANVAS_W, _CANVAS_H = 1280, 720

# ───────────────────────────────────────────────────────────────────────────
#  Public API  (signature must NOT change)
# ───────────────────────────────────────────────────────────────────────────


def generate_thumbnail(
    clip_path: str,
    titulo: str,
    video_path: str,
    start_sec: int,
    palavra_destaque: str = "",
) -> str:
    """Generate a thumbnail for the given clip.

    Args:
        clip_path: Absolute path of the clip ``.mp4`` file (used for output name).
        titulo: Title text to overlay on the thumbnail.
        video_path: Absolute path of the original downloaded VOD.
        start_sec: Timestamp (in seconds) of the highlight start in the VOD.
        palavra_destaque: Word to highlight with accent colour.

    Returns:
        Absolute path of the generated thumbnail (``.jpg``).
    """
    clip_path = Path(clip_path)
    _THUMB_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    thumb_path = _THUMB_DIR / f"{clip_path.stem}.jpg"

    # ── Disk cache ─────────────────────────────────────────────────────
    if thumb_path.exists():
        print(
            f"  [cache] Thumbnail já existe, pulando: {thumb_path}",
            file=sys.stderr,
            flush=True,
        )
        return str(thumb_path.resolve())

    video_path = Path(video_path)
    if not video_path.exists():
        print(
            f"[WARN] VOD não encontrado: {video_path}",
            file=sys.stderr,
            flush=True,
        )
        return _v1_fallback(clip_path, titulo, video_path, start_sec, thumb_path)

    # ── Load config ────────────────────────────────────────────────────
    cfg = _load_config()

    # ── Temporary directory ────────────────────────────────────────────
    with tempfile.TemporaryDirectory(dir=_THUMB_DIR) as tmp_dir:
        tmp = Path(tmp_dir)

        # ─── Step 1: extract frames ────────────────────────────────────
        frame_files = _extract_frames(video_path, start_sec, tmp)
        if not frame_files:
            print(
                "[WARN] Nenhum frame extraído — fallback v1",
                file=sys.stderr,
                flush=True,
            )
            return _v1_fallback(clip_path, titulo, video_path, start_sec, thumb_path)

        # ─── Step 1b: select best frame ────────────────────────────────
        best_frame, face_bbox = _select_best_frame(frame_files)
        if best_frame is None:
            print(
                "[WARN] Nenhum frame válido — fallback v1", file=sys.stderr, flush=True
            )
            return _v1_fallback(clip_path, titulo, video_path, start_sec, thumb_path)

        base_frame = Image.open(best_frame).convert("RGB")

        # ─── Step 2: background removal (BEN2) on full frame ────────────
        cutout = None
        if face_bbox is not None:
            cutout = _cutout_subject(base_frame, clip_path.stem)
            if cutout is not None:
                cutout = _zoom_cutout_to_face(cutout, face_bbox)
        else:
            # No face detected — force blurred background, skip BEN2 entirely
            cfg["thumb_bg_blur"] = True
            print(
                "  [WARN] Sem rosto detectado — usando fundo borrado",
                file=sys.stderr,
                flush=True,
            )

        # ─── Step 3: compose ──────────────────────────────────────────
        try:
            result = _compose_v2(base_frame, cutout, titulo, palavra_destaque, cfg)
            result.save(thumb_path, "JPEG", quality=95)
            print(
                f"  [OK] Thumbnail v2: {thumb_path}",
                file=sys.stderr,
                flush=True,
            )
            return str(thumb_path.resolve())
        except Exception as e:
            print(
                f"[WARN] Composição v2 falhou ({e}) — fallback v1",
                file=sys.stderr,
                flush=True,
            )
            return _v1_fallback(clip_path, titulo, video_path, start_sec, thumb_path)


# ───────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ───────────────────────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Load thumbnail-related config keys with defaults."""
    defaults = {
        "thumb_bg_color": "#1A1A2E",
        "thumb_bg_blur": False,
        "thumb_accent_color": "#FF4500",
        "thumb_text_color": "#FFFFFF",
    }
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for k in defaults:
            if k in raw:
                defaults[k] = raw[k]
    except Exception as e:
        print(
            f"[WARN] Config load failed ({e}) — using defaults",
            file=sys.stderr,
            flush=True,
        )
    return defaults


# ── Frame extraction ──────────────────────────────────────────────────────


def _extract_frames(video_path: Path, start_sec: int, tmp: Path) -> list[Path]:
    """Extract 15 frames (1 every 20s) from ``start_sec+60`` to ``start_sec+360``."""
    frames: list[Path] = []
    sample_start = start_sec + 60
    for i in range(15):
        ts = sample_start + i * 20
        out = tmp / f"frame_{i:04d}.jpg"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-ss",
                    str(ts),
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
                timeout=30,
            )
        except Exception as e:
            print(f"[WARN] Frame {i} em {ts}s falhou: {e}", file=sys.stderr, flush=True)
            continue
        if out.exists():
            frames.append(out)
    return frames


# ── Best-frame selection ──────────────────────────────────────────────────


def _select_best_frame(frame_files: list[Path]) -> tuple[Path | None, tuple | None]:
    """Try face detection; fallback to pixel variance.

    Returns:
        ``(path, face_bbox)`` where ``face_bbox`` is ``(x, y, w, h)`` or ``None``.
    """
    path, bbox = _select_by_expression(frame_files)
    if path is not None:
        return path, bbox
    return _select_by_variance(frame_files), None


def _select_by_expression(frame_files: list[Path]) -> tuple[Path | None, tuple | None]:
    """Use OpenCV Haar cascade to detect faces; pick frame with highest face-region variance.

    Returns:
        ``(path, (x, y, w, h))`` of the best frame and its largest face, or ``(None, None)``.
    """
    try:
        import cv2
    except ImportError:
        print(
            "[WARN] cv2 não instalado — pulando seleção por rosto",
            file=sys.stderr,
            flush=True,
        )
        return None, None

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        print(
            "[WARN] Haar cascade não encontrado — pulando seleção por rosto",
            file=sys.stderr,
            flush=True,
        )
        return None, None

    best_path: Path | None = None
    best_bbox: tuple | None = None
    best_score = -1.0

    for f in frame_files:
        try:
            img = Image.open(f).convert("RGB")
            img_np = np.array(img)
            import cv2

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            frame_h, frame_w = gray.shape
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            # Filter: face must be at least 8% of frame width (rejects false positives)
            faces = [face for face in faces if face[2] >= frame_w * 0.08]
            if len(faces) == 0:
                continue

            # Take up to 2 largest faces and compute their union bbox
            faces_sorted = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
            top_faces = faces_sorted[:2]

            ux1 = min(r[0] for r in top_faces)
            uy1 = min(r[1] for r in top_faces)
            ux2 = max(r[0] + r[2] for r in top_faces)
            uy2 = max(r[1] + r[3] for r in top_faces)
            union_bbox = (int(ux1), int(uy1), int(ux2 - ux1), int(uy2 - uy1))

            # Score: variance inside union region; bonus for frames with 2 faces
            face_region = img_np[uy1:uy2, ux1:ux2]
            score = float(np.var(face_region.astype(np.float32)))
            if len(top_faces) > 1:
                score *= 2.0  # strongly prefer frames that show multiple people

            if score > best_score:
                best_score = score
                best_path = f
                best_bbox = union_bbox
        except Exception as e:
            print(
                f"[WARN] Erro processando frame {f.name}: {e}",
                file=sys.stderr,
                flush=True,
            )
            continue

    if best_path is not None:
        print(
            f"  [face] Frame selecionado: {best_path.name} bbox={best_bbox} score={best_score:.1f}",
            file=sys.stderr,
            flush=True,
        )
    return best_path, best_bbox


def _crop_to_face(img: Image.Image, bbox: tuple, padding: float = 2.2) -> Image.Image:
    """Crop *img* around *bbox* (x, y, w, h) with padding to show upper body.

    ``padding`` multiplies the face size in each direction.  2.2 ≈ bust shot.
    """
    x, y, w, h = bbox
    iw, ih = img.size
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)
    return img.crop((x1, y1, x2, y2))


def _select_by_variance(frame_files: list[Path]) -> Path | None:
    """Pick frame with highest average pixel standard deviation (v1 fallback)."""
    best_path: Path | None = None
    best_score = -1.0
    for f in frame_files:
        try:
            img = Image.open(f).convert("RGB")
            stat = ImageStat.Stat(img)
            avg_stddev = sum(stat.stddev[:3]) / 3
        except Exception:
            continue
        if avg_stddev > best_score:
            best_score = avg_stddev
            best_path = f
    return best_path


# ── Background removal (BEN2) ─────────────────────────────────────────────

_ben2_model = None  # module-level cache — load once, reuse across thumbnails


def _get_ben2_model():
    """Load BEN_Base from HuggingFace (cached after first call)."""
    global _ben2_model
    if _ben2_model is not None:
        return _ben2_model
    try:
        import torch
        from ben2 import BEN_Base

        print(
            "  [ben2] Carregando modelo (primeira vez — baixa ~300MB se necessário)...",
            file=sys.stderr,
            flush=True,
        )
        model = BEN_Base.from_pretrained("PramaLLC/BEN2")
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("  [ben2] Modelo na GPU", file=sys.stderr, flush=True)
        _ben2_model = model
        return model
    except Exception as e:
        print(
            f"[WARN] ben2 falhou ao carregar modelo: {e}", file=sys.stderr, flush=True
        )
        return None


def _cutout_subject(frame: Image.Image, stem: str) -> Image.Image | None:
    """Remove background with BEN2; return RGBA cutout or ``None``."""
    cache_path = _CACHE_DIR / f"{stem}_cutout_full.png"
    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGBA")
        except Exception:
            pass

    model = _get_ben2_model()
    if model is None:
        return None

    try:
        result = model.inference(frame.convert("RGB"))

        # inference returns either (foreground_RGBA, mask) or just foreground_RGBA
        if isinstance(result, tuple):
            cutout = result[0]
        else:
            cutout = result

        if not isinstance(cutout, Image.Image):
            cutout = Image.fromarray(np.array(cutout))

        cutout = cutout.convert("RGBA")
        cutout.save(cache_path, "PNG")
        print(
            f"  [ben2] Cutout salvo em cache: {cache_path}", file=sys.stderr, flush=True
        )
        return cutout

    except Exception as e:
        print(f"[WARN] ben2 inference falhou: {e}", file=sys.stderr, flush=True)
        return None


# ── Post-BEN2 zoom ─────────────────────────────────────────────────


def _zoom_cutout_to_face(
    cutout: Image.Image, face_bbox: tuple | None, padding: float = 3.0
) -> Image.Image:
    """Crop the cutout RGBA around *face_bbox* with generous padding.

    ``padding`` = 3x the face size in each direction.
    Ensures crop is never smaller than 40% of the original cutout width
    (prevents excessive zoom on tiny detections).
    Returns the cropped RGBA, or the original cutout if *face_bbox* is ``None``.
    """
    if face_bbox is None:
        return cutout

    cw, ch = cutout.size
    x, y, w, h = face_bbox

    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(cw, x + w + pad_x)
    y2 = min(ch, y + h + pad_y)

    # Enforce minimum crop size: at least 25% of original cutout width
    min_w = int(cw * 0.25)
    crop_w = x2 - x1
    if crop_w < min_w:
        deficit = min_w - crop_w
        # Expand equally on both sides
        x1 = max(0, x1 - deficit // 2)
        x2 = min(cw, x2 + (deficit - deficit // 2))
        # If still too small after edge clamping, expand the other side
        crop_w = x2 - x1
        if crop_w < min_w:
            deficit = min_w - crop_w
            if x1 == 0:
                x2 = min(cw, x2 + deficit)
            else:
                x1 = max(0, x1 - deficit)

    return cutout.crop((x1, y1, x2, y2))


# ── v2 Composition ────────────────────────────────────────────────────────


def _compose_v2(
    frame: Image.Image,
    cutout: Image.Image | None,
    titulo: str,
    palavra_destaque: str,
    cfg: dict,
) -> Image.Image:
    """Build the 1280×720 composition canvas.

    Layers:
      1. Background (solid colour or blurred frame)
      2. Rim light behind subject
      3. Subject cutout (right side)
      4. Vignette
      5. Border
      6. Text (left third, with shadow + stroke)
      7. Highlighted word
    """
    bg_color = cfg["thumb_bg_color"]
    accent = cfg["thumb_accent_color"]
    text_color = cfg["thumb_text_color"]
    do_blur = cfg["thumb_bg_blur"]

    canvas = Image.new("RGB", (_CANVAS_W, _CANVAS_H), bg_color)

    # ── Layer 1: Background ────────────────────────────────────────────
    if do_blur:
        try:
            blurred = frame.resize((_CANVAS_W, _CANVAS_H), Image.LANCZOS)
            blurred = blurred.filter(ImageFilter.GaussianBlur(radius=25))
            # Increase saturation by ~40% (PIL doesn't have HSV; use ImageEnhance)
            from PIL import ImageEnhance

            blurred = ImageEnhance.Color(blurred).enhance(1.4)
            canvas.paste(blurred, (0, 0))
        except Exception as e:
            print(f"[WARN] Background blur falhou: {e}", file=sys.stderr, flush=True)
            # Keep solid colour

    # ── Layers 2-3: Rim light + Subject ────────────────────────────────
    cutout_layer = Image.new("RGBA", (_CANVAS_W, _CANVAS_H), (0, 0, 0, 0))

    if cutout is not None:
        try:
            # Scale cutout to 110% of canvas height (allows overflow — paste truncates)
            cw, ch = cutout.size
            scale = (_CANVAS_H * 1.10) / ch
            new_w = int(cw * scale)
            new_h = int(ch * scale)
            subject = cutout.resize((new_w, new_h), Image.LANCZOS)

            # Rim light: dilate alpha mask, blur, colour with accent
            alpha = subject.split()[3]
            rim_mask = alpha.filter(ImageFilter.MaxFilter(15))
            rim_mask = rim_mask.filter(ImageFilter.GaussianBlur(radius=20))
            rim = Image.new("RGBA", subject.size, _hex_to_rgba(accent, 180))
            rim.putalpha(rim_mask)

            # Position on right side (overflow equally at top and bottom)
            x_subj = _CANVAS_W - new_w + int(new_w * 0.25)
            y_subj = -36

            # Paste rim light behind subject
            cutout_layer.paste(rim, (x_subj, y_subj), rim)
            # Paste subject
            cutout_layer.paste(subject, (x_subj, y_subj), subject)

        except Exception as e:
            print(f"[WARN] Cutout paste falhou: {e}", file=sys.stderr, flush=True)

    canvas_rgba = Image.alpha_composite(canvas.convert("RGBA"), cutout_layer)
    canvas = canvas_rgba.convert("RGB")

    # ── Layer 4: Vignette ──────────────────────────────────────────────
    try:
        vignette = _make_vignette(_CANVAS_W, _CANVAS_H)
        canvas = Image.composite(
            canvas,
            Image.new("RGB", (_CANVAS_W, _CANVAS_H), (0, 0, 0)),
            vignette,
        )
    except Exception as e:
        print(f"[WARN] Vignette falhou: {e}", file=sys.stderr, flush=True)

    # ── Layer 5: Border (removida — mergia visualmente com bloco de keyword) ──
    draw = ImageDraw.Draw(canvas)

    # ── Layers 6-7: Text ───────────────────────────────────────────────
    try:
        canvas = _draw_text(
            canvas, titulo, palavra_destaque, text_color, accent, cutout is not None
        )
    except Exception as e:
        print(f"[WARN] Text overlay falhou: {e}", file=sys.stderr, flush=True)

    return canvas


# ── Vignette ──────────────────────────────────────────────────────────────


def _make_vignette(w: int, h: int) -> Image.Image:
    """Return a grayscale image where centre = white, edges = black."""
    x = np.linspace(-1, 1, w, dtype=np.float32)
    y = np.linspace(-1, 1, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(xx**2 + yy**2)
    # Gentle vignette: centre fully visible, edges at most 30% darkened
    alpha = np.clip(1.0 - dist * 0.45, 0.70, 1.0)
    arr = (alpha * 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


# ── Text rendering ────────────────────────────────────────────────────────


def _draw_text(
    canvas: Image.Image,
    titulo: str,
    palavra_destaque: str,
    text_color: str,
    accent: str,
    has_cutout: bool,
) -> Image.Image:
    """Draw title with body + keyword block. Returns modified canvas."""
    draw = ImageDraw.Draw(canvas)

    # ── Split into body and keyword ─────────────────────────────────────
    body = titulo.strip()
    keyword = palavra_destaque.upper().strip() if palavra_destaque else ""

    if keyword:
        idx = body.upper().find(keyword)
        if idx != -1:
            body = body[:idx] + body[idx + len(keyword) :]
        body = body.strip()

    # ── Max width ───────────────────────────────────────────────────────
    max_w = 680 if has_cutout else _CANVAS_W - 80

    # ── Font: body (170) ────────────────────────────────────────────────
    font_body = _load_condensed_font(170)
    font_body = _fit_font_width(draw, body.upper(), font_body, max_w)

    body_lines = textwrap.wrap(body.upper(), width=12)
    if not body_lines:
        body_lines = [body.upper()]

    # ── Font: keyword (210) ────────────────────────────────────────────
    kw_w = kw_h = 0
    font_kw = None
    if keyword:
        font_kw = _load_condensed_font(210)
        font_kw = _fit_font_width(draw, keyword, font_kw, max_w)
        kw_bbox = draw.textbbox((0, 0), keyword, font=font_kw)
        kw_w = kw_bbox[2] - kw_bbox[0]
        kw_h = kw_bbox[3] - kw_bbox[1]

    # ── Vertical positioning ───────────────────────────────────────────
    line_spacing_body = int(font_body.size * 1.2)

    total_lines_h = len(body_lines) * line_spacing_body
    keyword_h = int(font_kw.size * 1.15) if keyword else 0
    gap = 12
    total_h = total_lines_h + gap + keyword_h

    block_y = int(_CANVAS_H * 0.55) - total_h // 2

    # ── Shadow pass (body only) ─────────────────────────────────────────
    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    y = block_y
    for line in body_lines:
        lw = _text_width(draw, line, font_body)
        if has_cutout:
            x = 60
        else:
            x = (_CANVAS_W - lw) // 2
        shadow_draw.text((x + 8, y + 8), line, fill="#000000", font=font_body)
        y += line_spacing_body
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=6))
    canvas_rgba = Image.alpha_composite(canvas.convert("RGBA"), shadow_layer)
    canvas = canvas_rgba.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    # ── Main text pass: body ────────────────────────────────────────────
    y = block_y
    for line in body_lines:
        lw = _text_width(draw, line, font_body)
        if has_cutout:
            x = 60
        else:
            x = (_CANVAS_W - lw) // 2
        draw.text((x, y), line, fill=text_color, font=font_body)
        y += line_spacing_body

    # ── Keyword: accent block + white text ──────────────────────────────
    if keyword and font_kw:
        y_kw = block_y + total_lines_h + gap

        if has_cutout:
            x_kw = 60
        else:
            x_kw = (_CANVAS_W - kw_w) // 2

        bbox = draw.textbbox((x_kw, y_kw), keyword, font=font_kw)
        kw_rect = (
            bbox[0] - 10,
            bbox[1] - 6,
            bbox[2] + 10,
            bbox[3] + 6,
        )
        draw.rectangle(kw_rect, fill=accent)
        draw.text((x_kw, y_kw), keyword, fill="#FFFFFF", font=font_kw)

    return canvas


def _fit_font_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int
) -> ImageFont.FreeTypeFont:
    """Scale down *font* until *text* fits within *max_w* pixels."""
    sz = font.size
    while sz > 20 and _text_width(draw, text, font) > max_w:
        sz -= 2
        font = _load_condensed_font(sz)
    return font


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


# ── Font loading (v2: condensed/heavy first) ──────────────────────────────


def _load_condensed_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a condensed bold font, falling back through candidates."""
    candidates = [
        "Anton-Regular.ttf",
        "BebasNeue-Regular.ttf",
        "ArchivoBlack-Regular.ttf",
        "Montserrat-ExtraBold.ttf",
        "DejaVuSans-Bold.ttf",
        "Nunito-Bold.ttf",
        "Baloo2-Bold.ttf",
        "Comfortaa-Bold.ttf",
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
                try:
                    return ImageFont.truetype(str(ttf), size)
                except Exception:
                    continue
    return ImageFont.load_default()


# ── Colour helpers ────────────────────────────────────────────────────────


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)


# ───────────────────────────────────────────────────────────────────────────
#  v1 Fallback (original logic, kept as safety net)
# ───────────────────────────────────────────────────────────────────────────


def _v1_fallback(
    clip_path: Path,
    titulo: str,
    video_path: Path,
    start_sec: int,
    thumb_path: Path,
) -> str:
    """Run the original v1 thumbnail logic as a safety fallback."""
    print("  [v1] Usando pipeline original (fallback)", file=sys.stderr, flush=True)

    if not video_path.exists():
        raise FileNotFoundError(f"VOD não encontrado: {video_path}")

    with tempfile.TemporaryDirectory(dir=_THUMB_DIR) as tmp_dir:
        tmp = Path(tmp_dir)

        # Extract 10 frames (1 every 6s, starting at start_sec+180)
        sample_start = start_sec + 180
        durations = [str(sample_start + offset) for offset in range(0, 60, 6)]
        frame_files: list[Path] = []
        for i, ts in enumerate(durations):
            out = tmp / f"frame_{i:04d}.jpg"
            try:
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
                    timeout=30,
                )
            except Exception:
                continue
            if out.exists():
                frame_files.append(out)

        if not frame_files:
            raise RuntimeError("Nenhum frame extraído do VOD")

        # Score by variance
        best_frame: Path | None = None
        best_score = -1.0
        for f in frame_files:
            try:
                img = Image.open(f).convert("RGB")
                stat = ImageStat.Stat(img)
                avg_stddev = sum(stat.stddev[:3]) / 3
                if avg_stddev > best_score:
                    best_score = avg_stddev
                    best_frame = f
            except Exception:
                continue

        if best_frame is None:
            raise RuntimeError("Nenhum frame válido encontrado")

        base = Image.open(best_frame).convert("RGB")
        draw = ImageDraw.Draw(base)
        w, h = base.size

        font_size = max(60, int(w * 0.07))
        font = _load_condensed_font(font_size)

        lines = textwrap.wrap(titulo, width=16)
        if not lines:
            lines = [titulo]

        line_spacing = int(font_size * 1.2)
        total_text_height = len(lines) * (font_size * 1.2)
        text_block_y = int((h - total_text_height) // 2)

        # Shadow layer
        shadow_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        y = text_block_y
        for line in lines:
            line_w = _text_width(draw, line, font)
            x = (w - line_w) // 2
            shadow_draw.text((x, y), line, fill="#000000", font=font)
            y += line_spacing
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=12))
        base = Image.alpha_composite(base.convert("RGBA"), shadow_layer).convert("RGB")
        draw = ImageDraw.Draw(base)

        # Main text
        y = text_block_y
        for line in lines:
            line_w = _text_width(draw, line, font)
            x = (w - line_w) // 2
            draw.text(
                (x, y),
                line,
                fill="#FFE600",
                font=font,
                stroke_width=4,
                stroke_fill="#000000",
            )
            y += line_spacing

        base.save(thumb_path, "JPEG", quality=95)

    print(f"  [v1] Thumbnail: {thumb_path}", file=sys.stderr, flush=True)
    return str(thumb_path.resolve())
