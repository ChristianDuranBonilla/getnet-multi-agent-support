"""SQLite-backed conversations and tickets.

One file holds both tables. Every method opens its own connection, so instances are
safe to share between the API threads; the file runs in WAL mode so readers never block
on the writer.
"""

import secrets
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.repositories.protocols import ConflictError, Priority, Resolution, TicketStatus
from app.state import Conversation, Turn

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    merchant_id TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    summary_through_turn INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    index_in_conversation INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    agent TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (conversation_id, index_in_conversation)
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    merchant_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    thread_id TEXT NOT NULL UNIQUE,
    summary TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    claimed_by TEXT,
    resolved_at TEXT,
    operator_reply TEXT,
    resolution TEXT,
    reason TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS tickets_by_status ON tickets(status, created_at);
CREATE INDEX IF NOT EXISTS tickets_by_conversation ON tickets(conversation_id, created_at);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SqliteStore:
    """Connection and schema management shared by the two repositories."""

    def __init__(self, path: Path) -> None:
        """Remember the file path and create the schema on first use.

        Args:
            path: SQLite file; parent directories are created when missing.
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    @contextmanager
    def connect(self):
        """Yield a connection with WAL mode and row access by name; commits on success."""
        connection = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            connection.execute("BEGIN")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def ensure_schema(self) -> None:
        """Create the tables when they do not exist."""
        connection = sqlite3.connect(self._path, timeout=30)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(tickets)")}
            if "reason" not in columns:
                connection.execute("ALTER TABLE tickets ADD COLUMN reason TEXT NOT NULL DEFAULT ''")
                connection.commit()
        finally:
            connection.close()


class SqliteConversationRepository:
    """`ConversationRepository` over the shared SQLite file."""

    def __init__(self, store: SqliteStore) -> None:
        """Bind to a store."""
        self._store = store

    def create(self, merchant_id: str) -> str:
        """Create an empty conversation and return its id."""
        conversation_id = f"cnv_{secrets.token_hex(6)}"
        with self._store.connect() as connection:
            connection.execute(
                "INSERT INTO conversations (conversation_id, merchant_id, created_at)"
                " VALUES (?, ?, ?)",
                (conversation_id, merchant_id, _now()),
            )
        return conversation_id

    def get(self, conversation_id: str, merchant_id: str) -> Conversation | None:
        """Return the conversation when it exists and belongs to the merchant."""
        with self._store.connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ? AND merchant_id = ?",
                (conversation_id, merchant_id),
            ).fetchone()
            if row is None:
                return None
            turns = connection.execute(
                "SELECT role, content, agent, created_at FROM turns"
                " WHERE conversation_id = ? ORDER BY index_in_conversation",
                (conversation_id,),
            ).fetchall()
        return Conversation(
            conversation_id=row["conversation_id"],
            merchant_id=row["merchant_id"],
            summary=row["summary"],
            summary_through_turn=row["summary_through_turn"],
            created_at=datetime.fromisoformat(row["created_at"]),
            turns=[
                Turn(
                    role=turn["role"],
                    content=turn["content"],
                    agent=turn["agent"],
                    created_at=datetime.fromisoformat(turn["created_at"]),
                )
                for turn in turns
            ],
        )

    def append(self, conversation_id: str, turn: Turn) -> None:
        """Append one turn.

        The position is computed and inserted in one transaction; the uniqueness
        constraint on (conversation, position) turns a concurrent append into a retry
        instead of a duplicate position.
        """
        for attempt in range(3):
            try:
                with self._store.connect() as connection:
                    index = connection.execute(
                        "SELECT COALESCE(MAX(index_in_conversation), 0) + 1 FROM turns"
                        " WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()[0]
                    connection.execute(
                        "INSERT INTO turns (conversation_id, index_in_conversation, role,"
                        " content, agent, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            conversation_id,
                            index,
                            turn.role,
                            turn.content,
                            turn.agent,
                            turn.created_at.isoformat(timespec="seconds"),
                        ),
                    )
                return
            except sqlite3.IntegrityError:
                if attempt == 2:
                    raise

    def update_summary(self, conversation_id: str, summary: str, through_turn: int) -> None:
        """Store the rolling summary and the index of the last turn it covers."""
        with self._store.connect() as connection:
            connection.execute(
                "UPDATE conversations SET summary = ?, summary_through_turn = ?"
                " WHERE conversation_id = ?",
                (summary, through_turn, conversation_id),
            )


class SqliteTicketRepository:
    """`TicketRepository` over the shared SQLite file."""

    def __init__(self, store: SqliteStore) -> None:
        """Bind to a store."""
        self._store = store

    def create_ticket(
        self,
        merchant_id: str,
        summary: str,
        priority: Priority,
        conversation_id: str,
        request_id: str,
        thread_id: str,
        reason: str = "",
    ) -> str:
        """Open a ticket, or return the one already opened for the same turn.

        Idempotency is keyed on `thread_id`, the server-generated id of the turn (it
        embeds the merchant-owned conversation), never on anything the client sends; the
        node that opens tickets can run twice without opening two.
        """
        with self._store.connect() as connection:
            existing = connection.execute(
                "SELECT ticket_id FROM tickets WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if existing is not None:
                return existing["ticket_id"]
            ticket_id = f"TCK-{secrets.token_hex(3).upper()}"
            connection.execute(
                "INSERT INTO tickets (ticket_id, merchant_id, conversation_id, request_id,"
                " thread_id, summary, priority, status, created_at, reason)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)",
                (
                    ticket_id,
                    merchant_id,
                    conversation_id,
                    request_id,
                    thread_id,
                    summary,
                    priority,
                    _now(),
                    reason,
                ),
            )
        return ticket_id

    def get_ticket(self, ticket_id: str, merchant_id: str | None = None) -> dict | None:
        """Return a ticket; when a merchant id is given, only that merchant's ticket."""
        query = "SELECT * FROM tickets WHERE ticket_id = ?"
        params: tuple = (ticket_id,)
        if merchant_id is not None:
            query += " AND merchant_id = ?"
            params = (ticket_id, merchant_id)
        with self._store.connect() as connection:
            row = connection.execute(query, params).fetchone()
        return dict(row) if row else None

    def for_conversation(self, conversation_id: str) -> list[dict]:
        """Return the tickets opened from one conversation, newest first."""
        with self._store.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tickets WHERE conversation_id = ? ORDER BY created_at DESC",
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_tickets(self, status: TicketStatus | None = None) -> list[dict]:
        """Return tickets, newest first, optionally filtered by status."""
        query = "SELECT * FROM tickets"
        params: tuple = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC"
        with self._store.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def claim(self, ticket_id: str, operator_key_id: str) -> dict:
        """Assign the ticket to an operator; conflict when another operator holds it."""
        with self._store.connect() as connection:
            row = connection.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if row is None:
                raise LookupError(ticket_id)
            if row["status"] == "resolved":
                raise ConflictError("already_resolved")
            if row["claimed_by"] not in (None, operator_key_id):
                raise ConflictError("already_claimed")
            connection.execute(
                "UPDATE tickets SET status = 'claimed', claimed_by = ? WHERE ticket_id = ?",
                (operator_key_id, ticket_id),
            )
        return self.get_ticket(ticket_id)

    def reply(
        self, ticket_id: str, operator_key_id: str, message: str, resolution: Resolution
    ) -> dict:
        """Store the operator's reply; conflict when the ticket is already resolved.

        `resolved` closes the ticket; `needs_merchant` keeps it open as `waiting_merchant`
        so the operator can answer again once the merchant replies.
        """
        with self._store.connect() as connection:
            row = connection.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if row is None:
                raise LookupError(ticket_id)
            if row["status"] == "resolved":
                raise ConflictError("already_resolved")
            if row["claimed_by"] not in (None, operator_key_id):
                raise ConflictError("already_claimed")
            status = "resolved" if resolution == "resolved" else "waiting_merchant"
            resolved_at = _now() if resolution == "resolved" else None
            connection.execute(
                "UPDATE tickets SET status = ?, claimed_by = ?, resolved_at = ?,"
                " operator_reply = ?, resolution = ? WHERE ticket_id = ?",
                (status, operator_key_id, resolved_at, message, resolution, ticket_id),
            )
        return self.get_ticket(ticket_id)
