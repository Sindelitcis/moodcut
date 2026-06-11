"""Moodcut - entry point."""

import sys

from dotenv import load_dotenv

from modules.downloader import download_vod


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <URL_DO_VOD>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]

    load_dotenv()
    print(f"[*] Baixando VOD: {url}", file=sys.stderr)

    filepath = download_vod(url)
    print(filepath)


if __name__ == "__main__":
    main()
