import os
from datetime import date, datetime, timedelta, timezone

from dotenv import load_dotenv
import resend
from supabase import create_client

load_dotenv()

IST = timezone(timedelta(hours=5, minutes=30))
LEAD_DAYS = {
    "1 day before": 1,
    "2 days before": 2,
    "3 days before": 3,
    "1 week before": 7,
}

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
resend.api_key = os.environ["RESEND_API_KEY"]
EMAIL_FROM = os.environ["EMAIL_FROM"]


def run_daily_reminders():
    today = datetime.now(IST).date()

    res = (
        supabase.table("records")
        .select("*, profiles(email, full_name, permissions, channels)")
        .eq("status", "Confirmed")
        .is_("reminder_sent_at", "null")
        .execute()
    )

    sent_count = 0
    for rec in res.data:
        profile = rec.get("profiles") or {}
        perms = profile.get("permissions") or {}
        channels = profile.get("channels") or {}
        if not perms.get("notifications", True) or not channels.get("emailDigest", True):
            continue

        due = date.fromisoformat(rec["due_date"])
        days_left = (due - today).days
        trigger_days = LEAD_DAYS.get(rec.get("reminder_timing"), 2)

        if 0 <= days_left <= trigger_days:
            resend.Emails.send(
                {
                    "from": EMAIL_FROM,
                    "to": [profile["email"]],
                    "subject": f"LOOP Reminder: {rec['title']} due in {days_left} day(s)",
                    "html": f"""
                        <div style="font-family:sans-serif;max-width:520px;margin:auto;padding:24px;background:#0A1228;color:#EDEFF5;border-radius:12px;">
                          <h2 style="color:#D4AE5E;margin-top:0;">LOOP Reminder</h2>
                          <p>Hi {profile.get('full_name') or 'there'},</p>
                          <p>Your approved reminder for <strong>{rec['title']}</strong> ({rec['issuer']}) is coming up:</p>
                          <ul>
                            <li><strong>Amount:</strong> {rec['amount']}</li>
                            <li><strong>Due date:</strong> {due.strftime('%d %b %Y')} ({days_left} day(s) left)</li>
                          </ul>
                          <p style="font-size:12px;color:#9AA6C0;">You approved this reminder in LOOP. Sign in anytime to mark it paid.</p>
                        </div>
                    """,
                }
            )
            supabase.table("records").update(
                {"reminder_sent_at": datetime.now(timezone.utc).isoformat()}
            ).eq("id", rec["id"]).execute()
            sent_count += 1

    print(f"[{today.isoformat()}] Reminders dispatched: {sent_count}")


if __name__ == "__main__":
    run_daily_reminders()
