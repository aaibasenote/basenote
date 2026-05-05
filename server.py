"""
Basenote Server - FastAPI Backend
RAG-based Context Layer Platform

Requirements:
  pip install fastapi uvicorn python-multipart httpx aiofiles

Run:
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import re
import time
import httpx
import aiofiles
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
DATA_DIR = Path("basenote_data")
KB_DIR   = DATA_DIR / "knowledge_bases"   # per-user .txt chunks
SHARE_DIR = DATA_DIR / "shares"           # share relationship records
USERS_FILE = DATA_DIR / "users.json"

# Create directories
for d in [KB_DIR, SHARE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if not USERS_FILE.exists():
    USERS_FILE.write_text("{}")

# ──────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────
app = FastAPI(title="Basenote Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────
def load_users() -> dict:
    return json.loads(USERS_FILE.read_text())

def save_users(data: dict):
    USERS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def user_kb_dir(user_id: str) -> Path:
    p = KB_DIR / user_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def user_share_file(user_id: str) -> Path:
    return SHARE_DIR / f"{user_id}.json"

def load_shares(user_id: str) -> dict:
    """Returns {"sent": [...], "received": [...]}"""
    f = user_share_file(user_id)
    if not f.exists():
        return {"sent": [], "received": []}
    return json.loads(f.read_text())

def save_shares(user_id: str, data: dict):
    user_share_file(user_id).write_text(json.dumps(data, ensure_ascii=False, indent=2))

def can_access_kb(requester_id: str, owner_id: str) -> bool:
    if requester_id == owner_id:
        return True
    shares = load_shares(owner_id)
    return requester_id in shares["received"]

def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Simple character-level chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

async def download_gdrive_txt(share_url: str) -> str:
    """
    Converts Google Drive share link to direct download URL and fetches .txt content.
    Supports:
      - https://drive.google.com/file/d/{ID}/view
      - https://drive.google.com/open?id={ID}
    """
    # Extract file ID
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', share_url)
    if not match:
        match = re.search(r'id=([a-zA-Z0-9_-]+)', share_url)
    if not match:
        raise ValueError("Google Drive 링크에서 파일 ID를 찾을 수 없습니다.")

    file_id = match.group(1)
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        resp = await client.get(direct_url)
        if resp.status_code != 200:
            raise ValueError(f"파일 다운로드 실패 (HTTP {resp.status_code})")
        content_type = resp.headers.get("content-type", "")
        if "text" not in content_type and "octet-stream" not in content_type:
            # Try to handle virus-warning redirect page
            if "download_warning" in str(resp.url):
                confirm_url = str(resp.url) + "&confirm=t"
                resp = await client.get(confirm_url)
        return resp.text

# ──────────────────────────────────────────────
#  Models
# ──────────────────────────────────────────────
class RegisterRequest(BaseModel):
    user_id: str
    password: str

class LoginRequest(BaseModel):
    user_id: str
    password: str

class DriveConnectRequest(BaseModel):
    user_id: str
    drive_url: str
    doc_name: Optional[str] = None

class ShareRequest(BaseModel):
    from_user: str
    to_user: str

# ──────────────────────────────────────────────
#  Routes — Auth / User
# ──────────────────────────────────────────────
@app.post("/api/register")
async def register(req: RegisterRequest):
    users = load_users()
    uid = req.user_id.strip()
    pw = req.password.strip()
    if not uid:
        raise HTTPException(400, "아이디를 입력해주세요.")
    if not pw:
        raise HTTPException(400, "비밀번호를 입력해주세요.")
    if uid in users:
        raise HTTPException(409, "이미 존재하는 계정이에요")
    users[uid] = {"created_at": time.time(), "password": pw}
    save_users(users)
    return {"ok": True, "user_id": uid}

@app.post("/api/login")
async def login(req: LoginRequest):
    users = load_users()
    uid = req.user_id.strip()
    pw = req.password.strip()
    if not uid:
        raise HTTPException(400, "아이디를 입력해주세요.")
    if uid not in users:
        raise HTTPException(401, "아이디 또는 비밀번호가 올바르지 않아요")
    if users[uid].get("password", "") != pw:
        raise HTTPException(401, "아이디 또는 비밀번호가 올바르지 않아요")
    return {"ok": True, "user_id": uid}

@app.get("/api/users")
async def list_users():
    users = load_users()
    return {"users": list(users.keys())}

# ──────────────────────────────────────────────
#  Routes — Knowledge Base (Google Drive)
# ──────────────────────────────────────────────
@app.post("/api/kb/connect")
async def kb_connect(req: DriveConnectRequest):
    """Download .txt from Google Drive and store as chunks."""
    users = load_users()
    if req.user_id not in users:
        raise HTTPException(404, "등록되지 않은 사용자입니다.")

    try:
        text = await download_gdrive_txt(req.drive_url)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Save raw file
    kb_path = user_kb_dir(req.user_id)
    name = req.doc_name or f"doc_{int(time.time())}"
    # Sanitize name
    name = re.sub(r'[^\w가-힣\-]', '_', name)
    file_path = kb_path / f"{name}.txt"
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(text)

    chunks = chunk_text(text)
    return {
        "ok": True,
        "doc_name": name,
        "char_count": len(text),
        "chunk_count": len(chunks),
    }

@app.get("/api/kb/list/{user_id}")
async def kb_list(user_id: str):
    """List all documents in a user's KB."""
    kb_path = user_kb_dir(user_id)
    docs = []
    for f in kb_path.glob("*.txt"):
        stat = f.stat()
        docs.append({
            "name": f.stem,
            "size": stat.st_size,
            "modified_at": stat.st_mtime,
        })
    docs.sort(key=lambda x: x["modified_at"], reverse=True)
    return {"docs": docs}

@app.get("/api/kb/content/{requester_id}/{owner_id}/{doc_name}")
async def kb_content(requester_id: str, owner_id: str, doc_name: str):
    """Return raw document text if the requester can access the owner's KB."""
    users = load_users()
    if requester_id not in users:
        raise HTTPException(404, "등록되지 않은 사용자입니다.")
    if owner_id not in users:
        raise HTTPException(404, "지식베이스 소유자를 찾을 수 없습니다.")
    if not can_access_kb(requester_id, owner_id):
        raise HTTPException(403, "이 지식베이스에 접근할 수 없습니다.")

    safe_name = re.sub(r"[^\w가-힣\-]", "_", doc_name)
    file_path = user_kb_dir(owner_id) / f"{safe_name}.txt"
    if not file_path.exists():
        raise HTTPException(404, "문서를 찾을 수 없습니다.")

    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        raise HTTPException(500, "문서를 읽을 수 없습니다.")

    return {
        "owner_id": owner_id,
        "doc_name": safe_name,
        "text": text,
        "char_count": len(text),
    }

@app.delete("/api/kb/{user_id}/{doc_name}")
async def kb_delete(user_id: str, doc_name: str):
    kb_path = user_kb_dir(user_id)
    file_path = kb_path / f"{doc_name}.txt"
    if not file_path.exists():
        raise HTTPException(404, "문서를 찾을 수 없습니다.")
    file_path.unlink()
    return {"ok": True}

# ──────────────────────────────────────────────
#  Routes — Sharing
# ──────────────────────────────────────────────
@app.post("/api/share/send")
async def share_send(req: ShareRequest):
    users = load_users()
    if req.from_user not in users:
        raise HTTPException(404, f"사용자 '{req.from_user}'를 찾을 수 없습니다.")
    if req.to_user not in users:
        raise HTTPException(404, f"사용자 '{req.to_user}'를 찾을 수 없습니다.")
    if req.from_user == req.to_user:
        raise HTTPException(400, "자신에게는 공유할 수 없습니다.")

    # Update sender's share record
    sender_shares = load_shares(req.from_user)
    if req.to_user not in sender_shares["sent"]:
        sender_shares["sent"].append(req.to_user)
    save_shares(req.from_user, sender_shares)

    # Update receiver's share record
    receiver_shares = load_shares(req.to_user)
    if req.from_user not in receiver_shares["received"]:
        receiver_shares["received"].append(req.from_user)
    save_shares(req.to_user, receiver_shares)

    return {"ok": True, "message": f"'{req.to_user}'에게 지식베이스를 공유했습니다."}

@app.post("/api/share/revoke")
async def share_revoke(req: ShareRequest):
    sender_shares = load_shares(req.from_user)
    if req.to_user in sender_shares["sent"]:
        sender_shares["sent"].remove(req.to_user)
    save_shares(req.from_user, sender_shares)

    receiver_shares = load_shares(req.to_user)
    if req.from_user in receiver_shares["received"]:
        receiver_shares["received"].remove(req.from_user)
    save_shares(req.to_user, receiver_shares)

    return {"ok": True}

@app.get("/api/share/{user_id}")
async def share_info(user_id: str):
    return load_shares(user_id)

# ──────────────────────────────────────────────
#  Health
# ──────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
