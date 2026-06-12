"""Upload clips to YouTube via the Data API v3."""

import os
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRETS = Path("client_secrets.json")
TOKEN_DIR = Path("auth")
TOKEN_PATH = TOKEN_DIR / "youtube_token.json"


def _get_authenticated_service():
    """Return an authenticated YouTube API service object."""
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CLIENT_SECRETS.exists():
                raise FileNotFoundError(
                    f"Arquivo {CLIENT_SECRETS} não encontrado. "
                    "Gere um no Google Cloud Console e salve na raiz do projeto."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRETS), SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save token for next run
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return build("youtube", "v3", credentials=creds)


def upload_video(
    clip_path: str,
    titulo: str,
    descricao: str,
    thumb_path: str,
) -> str:
    """Upload a clip to YouTube and set its thumbnail.

    Args:
        clip_path: Absolute path to the clip ``.mp4`` file.
        titulo: Video title.
        descricao: Video description (optimised for YouTube).
        thumb_path: Absolute path to the thumbnail ``.jpg`` file.

    Returns:
        URL of the published video (``https://youtu.be/{video_id}``).
    """
    clip = Path(clip_path)
    if not clip.exists():
        raise FileNotFoundError(f"Clip não encontrado: {clip}")
    thumb = Path(thumb_path)
    if not thumb.exists():
        raise FileNotFoundError(f"Thumbnail não encontrada: {thumb}")

    youtube = _get_authenticated_service()

    # ── Upload ─────────────────────────────────────────────────
    body = {
        "snippet": {
            "title": titulo,
            "description": descricao,
            "categoryId": "24",  # Entertainment
        },
        "status": {
            "privacyStatus": "unlisted",
        },
    }

    media = MediaFileUpload(str(clip), chunksize=-1, resumable=True)

    print(
        f"  [YouTube] Enviando: {titulo[:60]}...",
        file=sys.stderr,
        flush=True,
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = request.execute()
    video_id = response["id"]
    video_url = f"https://youtu.be/{video_id}"

    # ── Set thumbnail ──────────────────────────────────────────
    youtube.thumbnails().set(
        videoId=video_id,
        media_body=MediaFileUpload(str(thumb)),
    ).execute()

    print(f"  [YouTube] {video_url}", file=sys.stderr, flush=True)
    return video_url
