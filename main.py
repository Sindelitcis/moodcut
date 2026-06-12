"""Moodcut - entry point."""

import sys

from dotenv import load_dotenv

from modules.analyzer import analyze
from modules.chunker import chunk_transcript
from modules.downloader import download_vod, get_vod_metadata
from modules.transcriber import transcribe


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <URL_DO_VOD>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]

    load_dotenv()

    # ── Metadata ─────────────────────────────────────────────────
    print(f"[*] Extraindo metadados de: {url}", file=sys.stderr)
    metadata = get_vod_metadata(url)
    print(
        f"    Título: {metadata['title']}",
        file=sys.stderr,
    )
    print(
        f"    Canal:  {metadata['channel']}",
        file=sys.stderr,
    )

    # ── Download ─────────────────────────────────────────────────
    print(f"[*] Baixando VOD...", file=sys.stderr)
    video_path = download_vod(url)
    print(f"[*] Vídeo salvo em: {video_path}", file=sys.stderr)

    # ── Transcribe ───────────────────────────────────────────────
    result = transcribe(video_path)
    n_segments = len(result["segments"])
    print(
        f"[OK] Transcrição: {n_segments} segmentos gerados",
        file=sys.stderr,
    )

    # ── Chunk ────────────────────────────────────────────────────
    chunks = chunk_transcript(result["segments"])
    print(
        f"[OK] Chunking: {len(chunks)} chunks gerados",
        file=sys.stderr,
    )
    if chunks:
        print(f"[Preview]\n{chunks[0]}", file=sys.stderr)

    # ── Analyze ──────────────────────────────────────────────────
    highlights = analyze(chunks, metadata)
    print(
        f"[OK] Análise: {len(highlights)} segmento(s) aprovado(s)",
        file=sys.stderr,
    )

    print(video_path)


if __name__ == "__main__":
    main()
