"""Moodcut - orchestrated pipeline with per-stage error handling and timing."""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from modules.analyzer import _sanitize_title, analyze
from modules.chunker import chunk_transcript
from modules.clipper import _sanitize_filename, cut_clips
from modules.downloader import download_vod, get_vod_metadata
from modules.transcriber import transcribe


def load_from_cache(metadata: dict, video_path: str = "") -> tuple:
    """Load clips, thumbs and highlights from cache for partial runs.

    Returns:
        ``(clips, thumbs, highlights, video_path)``
    """
    safe_name = _sanitize_title(metadata.get("title", "analysis"))
    analysis_path = Path("output/analysis") / f"{safe_name}.json"

    if not analysis_path.exists():
        raise FileNotFoundError(f"Arquivo de análise não encontrado: {analysis_path}")

    with open(analysis_path, encoding="utf-8") as f:
        highlights = json.load(f)

    # ── Reconstruct clip and thumbnail paths from segment data ──
    clips: list[str] = []
    thumbs: list[str] = []
    for seg in highlights:
        humor = seg.get("humor", "outros")
        titulo = seg.get("titulo_sugerido", "clip")
        safe = _sanitize_filename(titulo)
        humor_dir = Path("output") / humor
        if not humor_dir.exists():
            clips.append("")
            thumbs.append("")
            continue

        clip_matches = sorted(humor_dir.glob(f"*{safe}.mp4"))
        clips.append(str(clip_matches[0].resolve()) if clip_matches else "")

        thumb_matches = sorted(humor_dir.glob(f"*{safe}.jpg"))
        thumbs.append(str(thumb_matches[0].resolve()) if thumb_matches else "")

    # ── Ensure video_path points to the VOD ───────────────────
    if not video_path:
        downloads_dir = Path("output/downloads")
        if downloads_dir.exists():
            safe_name = _sanitize_title(metadata.get("title", ""))
            mp4_files = sorted(downloads_dir.glob(f"*{safe_name}*.mp4"))
            if mp4_files:
                video_path = str(mp4_files[0].resolve())

    return clips, thumbs, highlights, video_path


def main():
    parser = argparse.ArgumentParser(description="Moodcut pipeline")
    parser.add_argument("url", help="URL do VOD")
    parser.add_argument(
        "--only-thumb", action="store_true", help="Gera apenas thumbnails"
    )
    parser.add_argument(
        "--only-upload", action="store_true", help="Faz apenas upload YouTube"
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="Pula o upload YouTube"
    )
    parser.add_argument(
        "--skip-thumb", action="store_true", help="Pula a geração de thumbnails"
    )
    args = parser.parse_args()

    only_thumb = args.only_thumb
    only_upload = args.only_upload
    skip_upload = args.skip_upload
    skip_thumb = args.skip_thumb
    url = args.url

    load_dotenv()
    total_start = time.time()

    # ── Metadata (always needed for cache or context) ──────────
    try:
        t0 = time.time()
        print(f"[*] Extraindo metadados de: {url}", file=sys.stderr)
        metadata = get_vod_metadata(url)
        print(f"    Título: {metadata['title']}", file=sys.stderr)
        print(f"    Canal:  {metadata['channel']}", file=sys.stderr)
        elapsed = time.time() - t0
        print(f"[OK] Metadados extraídos em {elapsed:.1f}s", file=sys.stderr)
    except Exception as e:
        print(f"[ERRO] Falha ao extrair metadados: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Pipeline completo ou carregar do cache ─────────────────
    run_full = not only_thumb and not only_upload

    if only_thumb or only_upload:
        clips, thumbs, highlights, cached_video = load_from_cache(metadata)
        video_path = cached_video or ""
    else:
        # ── Download ───────────────────────────────────────────
        try:
            t0 = time.time()
            print("[*] Baixando VOD...", file=sys.stderr)
            video_path = download_vod(url)
            print(f"[*] Vídeo salvo em: {video_path}", file=sys.stderr)
            elapsed = time.time() - t0
            print(f"[OK] Download concluído em {elapsed:.1f}s", file=sys.stderr)
        except Exception as e:
            print(f"[ERRO] Falha no download: {e}", file=sys.stderr)
            sys.exit(1)

        # ── Transcribe ─────────────────────────────────────────
        try:
            t0 = time.time()
            result = transcribe(video_path)
            n_segments = len(result["segments"])
            print(
                f"[OK] Transcrição: {n_segments} segmentos gerados",
                file=sys.stderr,
            )
            elapsed = time.time() - t0
            print(f"[OK] Transcrição concluída em {elapsed:.1f}s", file=sys.stderr)
        except Exception as e:
            print(f"[ERRO] Falha na transcrição: {e}", file=sys.stderr)
            sys.exit(1)

        # ── Chunk ──────────────────────────────────────────────
        try:
            t0 = time.time()
            chunks = chunk_transcript(result["segments"])
            print(
                f"[OK] Chunking: {len(chunks)} chunks gerados",
                file=sys.stderr,
            )
            if chunks:
                print(f"[Preview]\n{chunks[0]}", file=sys.stderr)
            elapsed = time.time() - t0
            print(f"[OK] Chunking concluído em {elapsed:.1f}s", file=sys.stderr)
        except Exception as e:
            print(f"[ERRO] Falha no chunking: {e}", file=sys.stderr)
            sys.exit(1)

        # ── Analyze ────────────────────────────────────────────
        try:
            t0 = time.time()
            highlights = analyze(chunks, metadata)
            print(
                f"[OK] Análise: {len(highlights)} segmento(s) aprovado(s)",
                file=sys.stderr,
            )
            elapsed = time.time() - t0
            print(f"[OK] Análise concluída em {elapsed:.1f}s", file=sys.stderr)
        except Exception as e:
            print(f"[ERRO] Falha na análise: {e}", file=sys.stderr)
            sys.exit(1)

        # ── Clip ───────────────────────────────────────────────
        try:
            t0 = time.time()
            if highlights:
                clips = cut_clips(video_path, highlights, result["segments"])
                print(
                    f"[OK] Clipes: {len(clips)} gerados",
                    file=sys.stderr,
                )
            else:
                clips = []
                print(
                    "[OK] Clipes: nenhum segmento aprovado para corte",
                    file=sys.stderr,
                )
            elapsed = time.time() - t0
            print(f"[OK] Clipes concluído em {elapsed:.1f}s", file=sys.stderr)
        except Exception as e:
            print(f"[ERRO] Falha ao cortar clipes: {e}", file=sys.stderr)
            sys.exit(1)

        # ── Thumbnail (skip if flagged) ────────────────────────
        thumbs = []
        if not skip_thumb:
            try:
                from modules.thumbnailer import generate_thumbnail

                t0 = time.time()
                for clip_path, seg in zip(clips, highlights):
                    titulo = seg.get("titulo_sugerido", "clip")
                    from modules.analyzer import _parse_timestamp

                    start_sec = _parse_timestamp(seg.get("inicio", "00:00:00"))
                    thumb = generate_thumbnail(clip_path, titulo, video_path, start_sec)
                    thumbs.append(thumb)
                    print(f"[*] Thumbnail: {thumb}", file=sys.stderr)
                elapsed = time.time() - t0
                print(
                    f"[OK] Thumbnails concluídas em {elapsed:.1f}s",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"[ERRO] Falha nas thumbnails: {e}", file=sys.stderr)
                thumbs = []

    # ── Thumbnail-only path ───────────────────────────────────
    if only_thumb:
        try:
            from modules.thumbnailer import generate_thumbnail

            t0 = time.time()
            thumbs = []
            for clip_path, seg in zip(clips, highlights):
                if not clip_path:
                    continue
                titulo = seg.get("titulo_sugerido", "clip")
                from modules.analyzer import _parse_timestamp

                start_sec = _parse_timestamp(seg.get("inicio", "00:00:00"))
                # Use the VOD if found, otherwise fall back to clip
                source = video_path if video_path else clip_path
                thumb = generate_thumbnail(clip_path, titulo, source, start_sec)
                thumbs.append(thumb)
                print(f"[*] Thumbnail: {thumb}", file=sys.stderr)
            elapsed = time.time() - t0
            print(
                f"[OK] Thumbnails concluídas em {elapsed:.1f}s",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[ERRO] Falha nas thumbnails: {e}", file=sys.stderr)

    # ── Upload YouTube ─────────────────────────────────────────
    if not skip_upload and not only_thumb:
        try:
            from modules.uploader import upload_video

            t0 = time.time()
            urls = []
            for clip_path, thumb, seg in zip(clips, thumbs, highlights):
                if not clip_path or not thumb:
                    print(
                        "  [AVISO] Pulando upload — clipe ou thumbnail não encontrado",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                titulo = seg.get("titulo_sugerido", "clip")
                descricao = seg.get("descricao_youtube", "")
                url = upload_video(clip_path, titulo, descricao, thumb)
                urls.append(url)
            elapsed = time.time() - t0
            print(
                f"[OK] Uploads concluídos em {elapsed:.1f}s",
                file=sys.stderr,
            )
            for url in urls:
                print(url)
        except Exception as e:
            print(f"[ERRO] Falha no upload: {e}", file=sys.stderr)

    # ── Summary ────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(
        f"[OK] Pipeline concluído em {total_elapsed:.1f}s",
        file=sys.stderr,
    )
    if run_full:
        print(video_path)


if __name__ == "__main__":
    main()
