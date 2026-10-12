"""SQLite Persistence Store for FollowThrough.

Provides:
1. `tasks` table storing user-confirmed task objects, source evidence, and offsets.
2. `task_history` table tracking all sequential state transitions and evidence.
3. Content-hashing (SHA-256) to prevent duplicate task saves on retries or reruns.
4. Additive migration supporting Google Calendar event ID linkage without data loss.
"""

import datetime
import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB_PATH = Path("data/followthrough_tasks.db")


class TaskStore:
    """Manages SQLite database connections, schema migrations, and state history."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a SQLite connection with Row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initialize database tables with additive column migrations."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 1. Main tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    action_description TEXT NOT NULL,
                    owner TEXT,
                    deadline TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    evidence_text TEXT NOT NULL,
                    start_offset INTEGER,
                    end_offset INTEGER,
                    content_hash TEXT UNIQUE,
                    calendar_id TEXT,
                    calendar_event_id TEXT
                )
            """)
            # 2. Immutable state transition ledger
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    update_type TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    evidence_text TEXT,
                    followup_text TEXT,
                    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
                )
            """)
            # Additive migration: safely add calendar columns if upgrading an older database
            cursor.execute("PRAGMA table_info(tasks)")
            cols = {col["name"] for col in cursor.fetchall()}
            if "calendar_id" not in cols:
                cursor.execute("ALTER TABLE tasks ADD COLUMN calendar_id TEXT")
            if "calendar_event_id" not in cols:
                cursor.execute("ALTER TABLE tasks ADD COLUMN calendar_event_id TEXT")
            conn.commit()

    @staticmethod
    def compute_hash(source_text: str, action_description: str) -> str:
        """Compute SHA-256 fingerprint from source text and action description to prevent duplicates."""
        payload = f"{source_text.strip()}||{action_description.strip()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save_task(self, task_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Save a user-confirmed task. Rejects exact duplicate saves."""
        src = task_data.get("source_text", "").strip()
        act = task_data.get("action_description", "").strip()
        content_hash = self.compute_hash(src, act)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT task_id FROM tasks WHERE content_hash = ?", (content_hash,))
            existing = cursor.fetchone()
            if existing:
                return False, f"Duplicate task already exists (ID: {existing['task_id']})"

            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            task_id = f"task-{hashlib.md5((content_hash + now_iso).encode('utf-8')).hexdigest()[:8]}"

            cursor.execute("""
                INSERT INTO tasks (
                    task_id, created_at, source_text, action_description,
                    owner, deadline, status, evidence_text, start_offset, end_offset, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
            """, (
                task_id,
                now_iso,
                src,
                act,
                task_data.get("owner", "Unspecified"),
                task_data.get("deadline", "None specified"),
                task_data.get("evidence_text", ""),
                task_data.get("start_offset", 0),
                task_data.get("end_offset", 0),
                content_hash,
            ))

            # Record creation entry in task history ledger
            cursor.execute("""
                INSERT INTO task_history (
                    task_id, timestamp, update_type, old_value, new_value, evidence_text, followup_text
                ) VALUES (?, ?, 'created', '', ?, ?, ?)
            """, (
                task_id,
                now_iso,
                act,
                task_data.get("evidence_text", ""),
                src,
            ))
            conn.commit()
            return True, task_id

    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List tasks ordered by most recently created."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC", (status,))
            else:
                cursor.execute("SELECT * FROM tasks ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single task record by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_history(self, task_id: str) -> List[Dict[str, Any]]:
        """Retrieve sequential change history for a task."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM task_history WHERE task_id = ? ORDER BY history_id ASC", (task_id,))
            return [dict(r) for r in cursor.fetchall()]

    def apply_update(
        self,
        task_id: str,
        update_type: str,
        new_value: str,
        evidence_text: str,
        followup_text: str,
    ) -> bool:
        """Apply a confirmed update and record old and new values in history.
        
        Supports deadline_revised, owner_revised, completed, cancelled, and followup_linked.
        """
        task = self.get_task(task_id)
        if not task:
            return False

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        old_val = ""

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if update_type == "deadline_revised":
                old_val = task["deadline"] or "None"
                cursor.execute("UPDATE tasks SET deadline = ? WHERE task_id = ?", (new_value, task_id))
            elif update_type == "owner_revised":
                old_val = task["owner"] or "Unspecified"
                cursor.execute("UPDATE tasks SET owner = ? WHERE task_id = ?", (new_value, task_id))
            elif update_type in ("completed", "cancelled"):
                old_val = task["status"]
                cursor.execute("UPDATE tasks SET status = ? WHERE task_id = ?", (update_type, task_id))
            elif update_type == "followup_linked":
                old_val = "linked"
            else:
                return False

            cursor.execute("""
                INSERT INTO task_history (
                    task_id, timestamp, update_type, old_value, new_value, evidence_text, followup_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                now_iso,
                update_type,
                old_val,
                new_value,
                evidence_text,
                followup_text,
            ))
            conn.commit()
            return True

    def update_task_calendar(self, task_id: str, calendar_id: str, calendar_event_id: str) -> bool:
        """Link a Google Calendar event ID to an existing task."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE tasks SET calendar_id = ?, calendar_event_id = ? WHERE task_id = ?",
                (calendar_id, calendar_event_id, task_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear_task_calendar(self, task_id: str) -> bool:
        """Clear Google Calendar event linkage from a task upon event deletion."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE tasks SET calendar_id = NULL, calendar_event_id = NULL WHERE task_id = ?",
                (task_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
