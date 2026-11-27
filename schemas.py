from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    label: str = Field(description="Short field label, e.g., 'Issuing authority' or 'Amount due'")
    text: str = Field(description="Extracted value and brief context from the document")


class AmbiguousDateChoice(BaseModel):
    id: str = Field(description="Short unique slug, e.g., 'choice-ddmm' or 'choice-cutoff'")
    label: str = Field(description="Human-readable choice, e.g., '12 Dec 2026 — DD/MM format'")
    due_date: str = Field(description="ISO date YYYY-MM-DD")


class ExtractedDocument(BaseModel):
    title: str = Field(description="Concise record title, e.g., 'BESCOM Electricity Bill'")
    issuer: str = Field(description="Company, bank, utility, or property name")
    category: Literal["Bill", "Booking", "Warranty"]
    amount: str = Field(description="Formatted amount with currency symbol, e.g., '₹3,420'")
    numeric_amount: float = Field(description="Numeric value only, e.g., 3420")
    due_date: Optional[str] = Field(
        default=None,
        description="Confirmed ISO date YYYY-MM-DD if unambiguous; null if ambiguous",
    )
    is_date_ambiguous: bool = Field(
        description="True if date could be DD/MM vs MM/DD or multiple competing due/cutoff dates exist"
    )
    ambiguity_reason: Optional[str] = Field(
        default=None,
        description="Short explanation when is_date_ambiguous is True",
    )
    date_choices: List[AmbiguousDateChoice] = Field(
        default_factory=list,
        description="2 candidate dates when is_date_ambiguous is True, otherwise empty",
    )
    doc_header: str = Field(description="Formal document title")
    doc_subheader: str = Field(description="Account number, confirmation code, or billing period")
    reference_code: str = Field(description="Short invoice/booking/serial code")
    sections: List[DocumentSection] = Field(description="3 concise evidence highlights")
    snippet: str = Field(description="One-sentence summary grounded strictly in the document")


class RecordCreate(BaseModel):
    title: str
    issuer: str
    amount: str
    numeric_amount: float = 0
    due_date: str  # YYYY-MM-DD
    category: Literal["Bill", "Booking", "Warranty"]
    reminder_timing: str = "2 days before"
    source_type: str = "PDF Document"
    file_path: Optional[str] = None
    doc_header: Optional[str] = None
    doc_subheader: Optional[str] = None
    reference_code: Optional[str] = None
    sections: List[DocumentSection] = Field(default_factory=list)
    snippet: str = ""


class PermissionsUpdate(BaseModel):
    permissions_granted: bool = True
    documentAccess: bool = True
    notifications: bool = True
    microphone: bool = False
    leadTime: Literal["1 day before", "2 days before", "3 days before", "1 week before"] = "2 days before"


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class AskAIOutput(BaseModel):
    found: bool = Field(description="True only if an answer exists in the provided records")
    answer: str = Field(description="Concise 1-2 sentence answer strictly grounded in the records")
    source_record_id: Optional[str] = Field(
        default=None,
        description="The exact UUID of the primary supporting record, or null if not found",
    )


class WaitlistRequest(BaseModel):
    email: str
    tier: Literal["LOOP Plus", "LOOP Pro", "LOOP Family"]


class FeedbackRequest(BaseModel):
    screen: Optional[str] = "general"
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    message: str = Field(min_length=3, max_length=2000)
