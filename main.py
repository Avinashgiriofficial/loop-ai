import os
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ai_service import answer_from_vault, extract_document_fields
from auth import STORAGE_BUCKET, get_current_user, supabase_admin
from schemas import (
    AskRequest,
    FeedbackRequest,
    PermissionsUpdate,
    RecordCreate,
    WaitlistRequest,
)

IST = timezone(timedelta(hours=5, minutes=30))
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
    "audio/webm",
    "audio/mpeg",
    "audio/wav",
    "audio/mp4",
    "audio/ogg",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app = FastAPI(title="LOOP Personal Archive API", version="0.1.0-pilot")

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def enrich_record(rec: dict) -> dict:
    """Compute IST daysFromToday and formatted display date dynamically."""
    today = datetime.now(IST).date()
    due = date.fromisoformat(rec["due_date"])
    rec["daysFromToday"] = max(0, (due - today).days)
    rec["dueDateDisplay"] = f"{due.day} {due.strftime('%b %Y')}"
    return rec


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "loop-api",
        "date_ist": datetime.now(IST).date().isoformat(),
    }


# ------------------------------------------------------------------
# PROFILE & PERMISSIONS
# ------------------------------------------------------------------
@app.get("/api/me")
def get_profile(user=Depends(get_current_user)):
    res = supabase_admin.table("profiles").select("*").eq("id", user.id).single().execute()
    return res.data


@app.put("/api/permissions")
def update_permissions(payload: PermissionsUpdate, user=Depends(get_current_user)):
    if not payload.documentAccess:
        raise HTTPException(
            status_code=400,
            detail="Document extraction permission is required to use LOOP.",
        )

    updated = (
        supabase_admin.table("profiles")
        .update(
            {
                "permissions_granted": payload.permissions_granted,
                "permissions": {
                    "documentAccess": payload.documentAccess,
                    "notifications": payload.notifications,
                    "microphone": payload.microphone,
                    "leadTime": payload.leadTime,
                },
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .eq("id", user.id)
        .execute()
    )
    return updated.data[0]


# ------------------------------------------------------------------
# LIVE EXTRACTION (PDF / IMAGE / VOICE / TEXT)
# ------------------------------------------------------------------
@app.post("/api/extract")
async def extract_document(
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    user=Depends(get_current_user),
):
    if not file and (not text or len(text.strip()) < 8):
        raise HTTPException(
            status_code=400,
            detail="Upload a file or paste at least 8 characters of text.",
        )

    file_bytes = None
    mime_type = None
    stored_path = None
    source_type = "Pasted Text"

    if file:
        mime_type = file.content_type or "application/octet-stream"
        if mime_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime_type}")

        file_bytes = await file.read()
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="File exceeds the 10 MB pilot limit.")

        ext = os.path.splitext(file.filename or "upload")[1] or ".bin"
        stored_path = f"{user.id}/{uuid.uuid4().hex}{ext}"

        supabase_admin.storage.from_(STORAGE_BUCKET).upload(
            path=stored_path,
            file=file_bytes,
            file_options={"content-type": mime_type},
        )

        if mime_type == "application/pdf":
            source_type = "PDF Document"
        elif mime_type.startswith("image/"):
            source_type = "Photo / Receipt"
        elif mime_type.startswith("audio/"):
            source_type = "Voice Note"

    extracted = extract_document_fields(
        file_bytes=file_bytes,
        mime_type=mime_type,
        raw_text=text,
    )

    return {
        "extracted": extracted.model_dump(),
        "file_path": stored_path,
        "source_type": source_type,
    }


# ------------------------------------------------------------------
# VAULT RECORDS CRUD + SIGNED ORIGINAL VIEW
# ------------------------------------------------------------------
@app.get("/api/records")
def list_records(include_paid: bool = False, user=Depends(get_current_user)):
    query = supabase_admin.table("records").select("*").eq("user_id", user.id)
    if not include_paid:
        query = query.eq("status", "Confirmed")
    res = query.order("due_date", desc=False).execute()
    return [enrich_record(r) for r in res.data]


@app.post("/api/records")
def create_record(payload: RecordCreate, user=Depends(get_current_user)):
    # Enforce Free Pilot document limit (prepares for Plus/Pro/Family upgrades)
    profile = (
        supabase_admin.table("profiles")
        .select("plan_tier, max_active_records")
        .eq("id", user.id)
        .single()
        .execute()
        .data
    )
    count_res = (
        supabase_admin.table("records")
        .select("id", count="exact")
        .eq("user_id", user.id)
        .eq("status", "Confirmed")
        .execute()
    )
    if profile["plan_tier"] == "free_pilot" and (count_res.count or 0) >= profile["max_active_records"]:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Free Pilot limit of {profile['max_active_records']} active records reached. "
                "Mark old items paid or join the Plus waitlist."
            ),
        )

    row = payload.model_dump()
    row["user_id"] = user.id
    row["sections"] = [s.model_dump() for s in payload.sections]

    res = supabase_admin.table("records").insert(row).execute()
    return enrich_record(res.data[0])


@app.patch("/api/records/{record_id}/pay")
def mark_record_paid(record_id: str, user=Depends(get_current_user)):
    res = (
        supabase_admin.table("records")
        .update({"status": "Paid"})
        .eq("id", record_id)
        .eq("user_id", user.id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Record not found.")
    return {"status": "Paid", "id": record_id}


@app.get("/api/records/{record_id}/original")
def get_original_signed_url(record_id: str, user=Depends(get_current_user)):
    rec = (
        supabase_admin.table("records")
        .select("id, file_path")
        .eq("id", record_id)
        .eq("user_id", user.id)
        .single()
        .execute()
        .data
    )
    if not rec:
        raise HTTPException(status_code=404, detail="Record not found.")
    if not rec.get("file_path"):
        return {"signed_url": None}

    signed = supabase_admin.storage.from_(STORAGE_BUCKET).create_signed_url(rec["file_path"], 300)
    return {"signed_url": signed.get("signedURL") or signed.get("signedUrl")}


# ------------------------------------------------------------------
# ASK LOOP (GROUNDED VAULT Q&A)
# ------------------------------------------------------------------
@app.post("/api/ask")
def ask_loop(payload: AskRequest, user=Depends(get_current_user)):
    records = (
        supabase_admin.table("records")
        .select("*")
        .eq("user_id", user.id)
        .order("due_date", desc=False)
        .execute()
        .data
    )
    ai_out = answer_from_vault(payload.question, records)
    source_record = next(
        (enrich_record(r) for r in records if r["id"] == ai_out.source_record_id),
        None,
    )

    return {
        "found": ai_out.found,
        "answer": ai_out.answer,
        "source": source_record,
    }


# ------------------------------------------------------------------
# PRIVACY EXPORT, HARD DELETE, WAITLIST & PILOT FEEDBACK
# ------------------------------------------------------------------
@app.get("/api/export")
def export_archive(user=Depends(get_current_user)):
    profile = supabase_admin.table("profiles").select("*").eq("id", user.id).single().execute().data
    records = supabase_admin.table("records").select("*").eq("user_id", user.id).execute().data
    return {
        "exported_at": datetime.now(IST).isoformat(),
        "profile": profile,
        "records": records,
    }


@app.delete("/api/account/records")
def delete_all_records(user=Depends(get_current_user)):
    records = supabase_admin.table("records").select("file_path").eq("user_id", user.id).execute().data
    paths = [r["file_path"] for r in records if r.get("file_path")]
    if paths:
        supabase_admin.storage.from_(STORAGE_BUCKET).remove(paths)
    supabase_admin.table("records").delete().eq("user_id", user.id).execute()
    return {"deleted": True}


@app.post("/api/waitlist")
def join_waitlist(payload: WaitlistRequest):
    supabase_admin.table("waitlist").insert(payload.model_dump()).execute()
    return {"status": "joined", "tier": payload.tier}


@app.post("/api/feedback")
def submit_feedback(payload: FeedbackRequest, user=Depends(get_current_user)):
    row = payload.model_dump()
    row["user_id"] = user.id
    row["email"] = user.email
    supabase_admin.table("pilot_feedback").insert(row).execute()
    return {"status": "received"}
