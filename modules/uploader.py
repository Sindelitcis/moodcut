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


def _token_path(channel: int) -> Path:
    return TOKEN_DIR / f"token_canal_{channel}.json"


def _get_authenticated_service(channel: int = 1):
    """Return an authenticated YouTube API service object for *channel*."""
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    token_path = _token_path(channel)

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

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
            flow.redirect_uri = "http://localhost"
            auth_url, _ = flow.authorization_url(
                prompt="consent", access_type="offline"
            )
            print("\n" + "=" * 60, file=sys.stderr)
            print(f"  [auth] Canal {channel} — passo a passo:", file=sys.stderr)
            print(f"  1. Abra este link no navegador:", file=sys.stderr)
            print(f"\n  {auth_url}\n", file=sys.stderr)
            print(f"  2. Autorize a conta do YouTube.", file=sys.stderr)
            print(f"  3. O navegador vai dar erro — isso é normal.", file=sys.stderr)
            print(f"  4. Copie a URL COMPLETA da barra de endereços e cole aqui.", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            redirect_response = input("  Cole a URL completa aqui: ").strip()
            flow.fetch_token(authorization_response=redirect_response)
            creds = flow.credentials

        with open(token_path, "w") as f:
            f.write(creds.to_json())
        print(f"  [auth] Token salvo: {token_path}", file=sys.stderr)

    return build("youtube", "v3", credentials=creds)


def auth_channel(channel: int = 1) -> None:
    """Run OAuth flow for *channel* and save token. Use with --auth flag."""
    _get_authenticated_service(channel)
    print(f"[OK] Canal {channel} autorizado com sucesso.", file=sys.stderr)


def upload_video(
    clip_path: str,
    titulo: str,
    descricao: str,
    thumb_path: str,
    channel: int = 1,
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

    youtube = _get_authenticated_service(channel)

    # ── Upload ─────────────────────────────────────────────────
    body = {
        "snippet": {
            "title": titulo,
            "description": descricao,
            "categoryId": "24",  # Entertainment
        },
        "status": {
            "privacyStatus": "unlisted",
            "selfDeclaredMadeForKids": False,
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
