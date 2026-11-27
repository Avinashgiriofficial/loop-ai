import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from schemas import AskAIOutput, ExtractedDocument

load_dotenv()

IST = timezone(timedelta(hours=5, minutes=30))
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def today_ist_iso() -> str:
    return datetime.now(IST).date().isoformat()


def extract_document_fields(
    file_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None,
    raw_text: Optional[str] = None,
) -> ExtractedDocument:
    today_str = today_ist_iso()
    prompt = f"""
    You are LOOP, a careful personal archive assistant built for Indian households.
    Today's date in IST is {today_str}.

    Extract the structured record details from the provided document, image, text, or voice note.
    Rules:
    1. Default currency is INR (₹) unless another currency is explicitly printed.
    2. NEVER guess an ambiguous date silently. If a numeric date like 03/12/2026 could reasonably mean
       either DD/MM/YYYY or MM/DD/YYYY (day <= 12 and month <= 12 without month name), OR if both a
       billing cutoff date and due date appear, set is_date_ambiguous=true, set due_date=null, explain
       why in ambiguity_reason, and provide exactly 2 candidate dates in date_choices.
    3. If the date is unambiguous (e.g., '28 Nov 2026' or '24/11/2026'), set is_date_ambiguous=false,
       populate due_date in YYYY-MM-DD format, and leave date_choices empty.
    4. Categorize strictly as 'Bill', 'Booking', or 'Warranty'.
    5. Populate 'sections' with 3 concise factual highlights directly quoted/grounded in the source.
    """

    contents = [prompt]
    if file_bytes and mime_type:
        contents.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
    if raw_text:
        contents.append(f"User-provided text:\n{raw_text}")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractedDocument,
            temperature=0.1,
        ),
    )
    return ExtractedDocument.model_validate_json(response.text)


def answer_from_vault(question: str, records: List[dict]) -> AskAIOutput:
    if not records:
        return AskAIOutput(
            found=False,
            answer="No saved records exist in your Vault yet. Add and approve a document first.",
            source_record_id=None,
        )

    today_str = today_ist_iso()
    compact_vault = [
        {
            "id": r["id"],
            "title": r["title"],
            "issuer": r["issuer"],
            "category": r["category"],
            "amount": r["amount"],
            "due_date": r["due_date"],
            "status": r["status"],
            "snippet": r["snippet"],
            "sections": r.get("sections", []),
        }
        for r in records
    ]

    prompt = f"""
    You are Ask LOOP. Today's date in IST is {today_str}.
    Answer the user's question using ONLY the verified Vault records below.
    Never invent numbers, dates, or accounts not present in the records.
    If no record answers the question, set found=false and source_record_id=null.

    Vault records (JSON):
    {json.dumps(compact_vault, ensure_ascii=False)}

    User question: {question}
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AskAIOutput,
            temperature=0.0,
        ),
    )
    return AskAIOutput.model_validate_json(response.text)
