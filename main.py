"""Moodcut - orchestrated pipeline with per-stage error handling and timing."""

import sys
import time

from dotenv import load_dotenv

from modules.analyzer import analyze
from modules.chunker import chunk_transcript
from modules.clipper import cut_clips
from modules.downloader import download_vod, get_vod_metadata
from modules.transcriber import transcribe


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <URL_DO_VOD>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    load_dotenv()

    total_start = time.time()

    # ── Metadata ─────────────────────────────────────────────────
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

    # ── Download ─────────────────────────────────────────────────
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

    # ── Transcribe ───────────────────────────────────────────────
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

    # ── Chunk ────────────────────────────────────────────────────
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

    # ── Analyze ──────────────────────────────────────────────────
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

    # ── Clip ─────────────────────────────────────────────────────
    try:
        t0 = time.time()
        if highlights:
            clips = cut_clips(video_path, highlights)
            print(
                f"[OK] Clipes: {len(clips)} gerados",
                file=sys.stderr,
            )
        else:
            print(
                "[OK] Clipes: nenhum segmento aprovado para corte",
                file=sys.stderr,
            )
        elapsed = time.time() - t0
        print(f"[OK] Clipes concluído em {elapsed:.1f}s", file=sys.stderr)
    except Exception as e:
        print(f"[ERRO] Falha ao cortar clipes: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(
        f"[OK] Pipeline concluído em {total_elapsed:.1f}s",
        file=sys.stderr,
    )
    print(video_path)


if __name__ == "__main__":
    main()
