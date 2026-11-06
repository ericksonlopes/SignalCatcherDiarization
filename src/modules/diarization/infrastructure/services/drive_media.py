"""Resolve verified Drive media into a short-lived inference input file."""

import hashlib
import json
import re
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as UserCredentials
from sqlalchemy import text

from src.core.config.settings import settings
from src.core.database.connector import Session


def _load_credentials(path: str) -> service_account.Credentials | UserCredentials:
    try:
        with Path(path).open(encoding="utf-8") as source:
            payload = json.load(source)
        if not isinstance(payload, dict):
            raise ValueError("Google Drive credentials must be a JSON object.")
        kind = payload.get("type")
        if kind == "service_account":
            return service_account.Credentials.from_service_account_file(
                path, scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
        if kind == "authorized_user":
            # Keep the user's granted scopes when refreshing an existing token.
            return UserCredentials.from_authorized_user_file(path)
        raise ValueError("Unsupported Google Drive credential type.")
    except (OSError, ValueError) as exc:
        raise ValueError("Google Drive credentials are invalid or unreadable.") from exc


@contextmanager
def drive_media(uri: str, heartbeat: Callable[[], None]) -> Iterator[str]:
    file_id = uri.removeprefix("gdrive://")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
        raise ValueError("Invalid Google Drive media reference.")
    with Session() as db:
        source = db.execute(
            text(
                "SELECT b.md5, b.size FROM youtube_drive_backups b "
                "JOIN youtube_contents c ON c.id = b.content_id "
                "WHERE b.file_id = :file_id AND b.verified = true "
                "AND c.drive_file_id = b.file_id AND c.deletion_requested = false"
            ),
            {"file_id": file_id},
        ).first()
    if source is None:
        raise ValueError("This Drive media is not a verified YouTube copy.")
    heartbeat()
    credential = _load_credentials(settings.GOOGLE_DRIVE_CREDENTIALS_PATH)
    heartbeat()
    with (
        AuthorizedSession(credential, refresh_timeout=30) as session,
        tempfile.TemporaryDirectory(prefix="signalcatcher-drive-") as cache,
    ):
        path = Path(cache, "input.media")
        digest = hashlib.md5(usedforsecurity=False)
        size = 0
        with session.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            params={"alt": "media", "supportsAllDrives": "true"},
            stream=True,
            timeout=(10, 30),
        ) as media:
            media.raise_for_status()
            with path.open("wb") as target:
                for chunk in media.iter_content(1024 * 1024):
                    heartbeat()
                    if chunk:
                        target.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
        if size != source.size or digest.hexdigest() != source.md5:
            raise ValueError("Drive media failed size/checksum verification.")
        yield str(path)
