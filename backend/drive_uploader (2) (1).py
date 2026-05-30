"""
backend/drive_uploader.py — OAuth2 version (personal Google account)
"""

import os, io, json, threading
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

def _token_path() -> str:
    return os.getenv("GDRIVE_TOKEN", "")

def _folder_id() -> str:
    return os.getenv("GDRIVE_FOLDER_ID", "")

def _gdrive_enabled() -> bool:
    return bool(_token_path() and _folder_id())

GDRIVE_ENABLED = _gdrive_enabled()

_drive_service = None

def _get_drive_service():
    global _drive_service
    if _drive_service is not None:
        return _drive_service

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        p = Path(_token_path())
        if not p.exists():
            print(f"[Drive] ✗ Không tìm thấy token: {p}")
            return None

        data = json.loads(p.read_text())
        creds = Credentials(
            token=data["token"],
            refresh_token=data["refresh_token"],
            client_id=data["client_id"],
            client_secret=data["client_secret"],
            token_uri=data["token_uri"],
        )

        if creds.expired:
            creds.refresh(Request())
            data["token"] = creds.token
            p.write_text(json.dumps(data, indent=2))
            print("[Drive] ✓ Token đã được refresh tự động")

        _drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
        print("[Drive] ✓ OAuth2 OK")
        return _drive_service

    except Exception as e:
        print(f"[Drive] ✗ Lỗi: {e}")
        return None


def upload_image_to_drive(
    img_bytes: bytes,
    prefix: str = "physbot",
    session_id: str = "",
) -> str | None:
    if not _gdrive_enabled():
        print("[Drive] Chưa cấu hình — bỏ qua upload")
        return None

    service = _get_drive_service()
    if service is None:
        return None

    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sid      = f"_{session_id}" if session_id else ""
    filename = f"{prefix}{sid}_{ts}.jpg"

    try:
        from googleapiclient.http import MediaIoBaseUpload

        uploaded = service.files().create(
            body={"name": filename, "parents": [_folder_id()]},
            media_body=MediaIoBaseUpload(io.BytesIO(img_bytes), mimetype="image/jpeg"),
            fields="id, name, webViewLink",
        ).execute()

        file_id   = uploaded.get("id", "")
        view_link = uploaded.get("webViewLink", "")

        service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()

        print(f"[Drive] ✓ Upload OK: {filename}")
        print(f"[Drive]   Link: {view_link}")
        return view_link

    except Exception as e:
        print(f"[Drive] ✗ Upload thất bại: {e}")
        return None


def upload_image_background(
    img_bytes: bytes,
    prefix: str = "physbot",
    session_id: str = "",
):
    t = threading.Thread(
        target=upload_image_to_drive,
        args=(img_bytes,),
        kwargs={"prefix": prefix, "session_id": session_id},
        daemon=True,
    )
    t.start()
    return t


if __name__ == "__main__":
    if not _gdrive_enabled():
        print("⚠  GDRIVE_TOKEN hoặc GDRIVE_FOLDER_ID chưa đặt trong .env")
    else:
        print("Testing upload ảnh giả...")
        tiny_jpg = bytes([
            0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,
            0x01,0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xD9
        ])
        link = upload_image_to_drive(tiny_jpg, prefix="test", session_id="test001")
        if link:
            print(f"\n✅ Upload thành công!\n   Link: {link}")
        else:
            print("\n❌ Upload thất bại — xem log bên trên")