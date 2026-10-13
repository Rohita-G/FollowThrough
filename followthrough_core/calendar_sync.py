"""Google Calendar sync module for FollowThrough.

NOTE: Calendar integration and automated reminder functionality are legacy / out-of-scope
for the active FollowThrough project, pending final deletion decision.

Uses minimal Calendar scope (calendar.events).
Local desktop OAuth; tokens kept outside Git and never logged.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
DEFAULT_CREDENTIALS_DIR = Path("data/credentials")
DEFAULT_CREDENTIALS_PATH = DEFAULT_CREDENTIALS_DIR / "credentials.json"
DEFAULT_TOKEN_PATH = DEFAULT_CREDENTIALS_DIR / "token.json"

NOTIFICATION_DISCLAIMER = (
    "Notification behavior (pop-up vs email) depends on your personal Google Calendar account settings."
)


class CalendarSyncManager:
    def __init__(
        self,
        credentials_path: Path = DEFAULT_CREDENTIALS_PATH,
        token_path: Path = DEFAULT_TOKEN_PATH,
    ):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.credentials_path.parent.mkdir(parents=True, exist_ok=True)

    def is_connected(self) -> bool:
        """Check if an authorized, refreshable Google OAuth token exists."""
        if not self.token_path.is_file():
            return False
        try:
            from google.oauth2.credentials import Credentials
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
            if creds and creds.valid:
                return True
            if creds and creds.expired and creds.refresh_token:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
                with self.token_path.open("w", encoding="utf-8") as f:
                    f.write(creds.to_json())
                return True
            return False
        except Exception:
            return False

    def disconnect(self) -> bool:
        """Disconnect and purge saved OAuth token."""
        if self.token_path.is_file():
            try:
                os.remove(self.token_path)
                return True
            except OSError:
                return False
        return True

    def get_credentials(self):
        """Load credentials from token file without logging secrets."""
        from google.oauth2.credentials import Credentials
        if not self.token_path.is_file():
            raise FileNotFoundError("Google Calendar is not connected. Token file not found.")
        creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            with self.token_path.open("w", encoding="utf-8") as f:
                f.write(creds.to_json())
        return creds

    def start_desktop_oauth(self, secrets_path: Optional[Path] = None) -> Tuple[bool, str]:
        """Launch local desktop OAuth server flow. Does not log tokens."""
        target_path = secrets_path or self.credentials_path
        if not target_path.is_file():
            return False, f"Client secrets file not found at {target_path}. Please place credentials.json first."

        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            flow = InstalledAppFlow.from_client_secrets_file(str(target_path), SCOPES)
            # Runs local desktop server on available port; opens browser
            creds = flow.run_local_server(port=0)
            with self.token_path.open("w", encoding="utf-8") as f:
                f.write(creds.to_json())
            return True, "Successfully connected to Google Calendar."
        except Exception as e:
            return False, f"OAuth flow failed: {type(e).__name__}"

    def get_service(self):
        """Build Google Calendar v3 service."""
        from googleapiclient.discovery import build
        creds = self.get_credentials()
        return build("calendar", "v3", credentials=creds)

    @staticmethod
    def prepare_event_payload(
        title: str,
        description: str,
        start_date: str,
        is_all_day: bool = True,
        end_date: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        timezone: str = "UTC",
        reminder_minutes: int = 30,
    ) -> Dict[str, Any]:
        """Create and validate event payload before writing.

        Rules:
        - Date-only deadlines remain all-day unless user supplies a time.
        - Timed events require start and end times; never invent duration.
        """
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("Event title cannot be empty.")

        clean_start_date = start_date.strip()
        if not clean_start_date:
            raise ValueError("Start date cannot be empty (format YYYY-MM-DD).")

        payload: Dict[str, Any] = {
            "summary": clean_title,
            "description": description.strip(),
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "popup", "minutes": int(reminder_minutes)},
                ],
            },
        }

        if is_all_day:
            eff_end_date = (end_date or clean_start_date).strip()
            payload["start"] = {"date": clean_start_date}
            payload["end"] = {"date": eff_end_date}
        else:
            if not start_time or not start_time.strip():
                raise ValueError("Timed events require an explicit start time (HH:MM).")
            if not end_time or not end_time.strip():
                raise ValueError("Timed events require an explicit end time / duration; do not invent one.")
            
            s_time = start_time.strip()
            e_time = end_time.strip()
            start_dt = f"{clean_start_date}T{s_time}:00"
            end_dt = f"{(end_date or clean_start_date).strip()}T{e_time}:00"

            payload["start"] = {"dateTime": start_dt, "timeZone": timezone}
            payload["end"] = {"dateTime": end_dt, "timeZone": timezone}

        return payload

    def sync_event(
        self,
        calendar_id: str,
        payload: Dict[str, Any],
        existing_event_id: Optional[str] = None,
        service: Optional[Any] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Write to Google Calendar. Updates existing event if ID provided, else inserts."""
        srv = service or self.get_service()
        try:
            if existing_event_id:
                # Idempotent update to same event
                res = srv.events().update(
                    calendarId=calendar_id,
                    eventId=existing_event_id,
                    body=payload,
                ).execute()
                return True, existing_event_id, res
            else:
                # Create new event
                res = srv.events().insert(
                    calendarId=calendar_id,
                    body=payload,
                ).execute()
                return True, res.get("id", ""), res
        except Exception as e:
            return False, f"API write error: {type(e).__name__}", {}

    def delete_event(
        self,
        calendar_id: str,
        event_id: str,
        service: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """Delete event from Google Calendar upon explicit user confirmation."""
        srv = service or self.get_service()
        try:
            srv.events().delete(
                calendarId=calendar_id,
                eventId=event_id,
            ).execute()
            return True, "Event deleted from Google Calendar."
        except Exception as e:
            return False, f"API deletion error: {type(e).__name__}"
