"""Moodcut - entry point."""

import sys

from dotenv import load_dotenv

from modules.downloader import download_vod
from modules.transcriber import transcribe


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <URL_DO_VOD>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]

    load_dotenv()

    # ── Download ─────────────────────────────────────────────────
    print(f"[*] Baixando VOD: {url}", file=sys.stderr)
    video_path = download_vod(url)
    print(f"[*] Vídeo salvo em: {video_path}", file=sys.stderr)

    # ── Transcribe ───────────────────────────────────────────────
    result = transcribe(video_path)
    n_segments = len(result["segments"])
    print(
        f"[OK] Transcrição: {n_segments} segmentos gerados",
        file=sys.stderr,
    )
    print(video_path)


if __name__ == "__main__":
    main()
