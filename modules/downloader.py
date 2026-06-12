import os
import sys
from pathlib import Path

import yt_dlp


def get_vod_metadata(url: str) -> dict:
    """Extract title, channel and URL from a VOD without downloading.

    Args:
        url: The VOD URL.

    Returns:
        Dict with ``"title"``, ``"channel"`` and ``"url"`` keys.
    """
    ydl_opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title", ""),
            "channel": info.get("channel") or info.get("uploader", ""),
            "url": url,
        }


def download_vod(url: str) -> str:
    """Download the best available MP4 VOD from a URL (Twitch, YouTube, etc.).

    Args:
        url: The VOD URL to download.

    Returns:
        The absolute path to the downloaded file.
    """
    output_dir = Path("output/downloads")
    output_dir.mkdir(parents=True, exist_ok=True)

    def progress_hook(d):
        if d["status"] == "downloading":
            percent = d.get("_percent_str", "?.?%").strip()
            speed = d.get("_speed_str", "?").strip()
            eta = d.get("_eta_str", "?").strip()
            print(
                f"  [download] {percent} at {speed} ETA {eta}",
                file=sys.stderr,
                flush=True,
            )
        elif d["status"] == "finished":
            print(
                f"  [download] 100% - {d['filename']}",
                file=sys.stderr,
                flush=True,
            )

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output_dir / "%(title)s_%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "progress_hooks": [progress_hook],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = _resolve_filepath(ydl, info, output_dir)

    print(f"[OK] Download concluído: {filepath}", file=sys.stderr, flush=True)
    return filepath


def _resolve_filepath(ydl: yt_dlp.YoutubeDL, info: dict, output_dir: Path) -> str:
    """Determine the actual file path after download and post-processing."""
    # 1. Check requested_downloads (populated for multi-format downloads)
    if info.get("requested_downloads"):
        fp = info["requested_downloads"][0].get("filepath")
        if fp and os.path.exists(fp):
            return os.path.abspath(fp)

    # 2. Reconstruct from template and adjust extension (merge_output_format)
    fp = ydl.prepare_filename(info)
    base, _ = os.path.splitext(fp)
    fp = base + ".mp4"
    if os.path.exists(fp):
        return os.path.abspath(fp)

    # 3. Last resort — scan output_dir for a file containing the video ID
    vid_id = info.get("id", "")
    if vid_id:
        for f in output_dir.iterdir():
            if f.is_file() and vid_id in f.name:
                return str(f.resolve())

    raise FileNotFoundError(
        f"Não foi possível localizar o arquivo baixado em {output_dir.resolve()}"
    )
