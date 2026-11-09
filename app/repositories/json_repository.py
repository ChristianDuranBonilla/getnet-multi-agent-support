"""Repository backed by the JSON files under `data/`."""

import json
import secrets
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from app.repositories.protocols import Priority


class JsonSupportRepository:
    """In-memory repository loaded from `users.json`, `settlements.json` and `terminals.json`.

    Relative day offsets in the settlement file are converted to ISO dates at read
    time so that scenarios about "yesterday" stay valid on any day.
    """

    def __init__(self, data_dir: Path, today: date | None = None) -> None:
        """Load the JSON files.

        Args:
            data_dir: Directory containing the three JSON files.
            today: Reference date for relative offsets; defaults to the current UTC date.
        """
        self._today = today or datetime.now(UTC).date()
        self._users = self._load(data_dir / "users.json")
        self._settlements = self._load(data_dir / "settlements.json")
        self._terminals = self._load(data_dir / "terminals.json")
        self._tickets: dict[str, dict] = {}

    @staticmethod
    def _load(path: Path) -> dict:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        data.pop("_note", None)
        return data

    def _to_date(self, days_ago: int | None) -> str | None:
        if days_ago is None:
            return None
        return (self._today - timedelta(days=days_ago)).isoformat()

    def today(self) -> date:
        """Return the reference date used for relative dates."""
        return self._today

    def get_user(self, user_id: str) -> dict | None:
        """Return the merchant profile or `None`."""
        user = self._users.get(user_id)
        return dict(user, user_id=user_id) if user else None

    def get_settlements(self, user_id: str) -> list[dict] | None:
        """Return recent sales with ISO dates or `None`."""
        sales = self._settlements.get(user_id)
        if sales is None:
            return None
        result = []
        for sale in sales:
            item = {k: v for k, v in sale.items() if not k.endswith("days_ago")}
            item["sale_date"] = self._to_date(sale["days_ago"])
            item["expected_deposit_date"] = self._to_date(sale.get("expected_deposit_days_ago"))
            result.append(item)
        return result

    def get_terminal(self, user_id: str) -> dict | None:
        """Return the terminal status or `None`."""
        terminal = self._terminals.get(user_id)
        return dict(terminal) if terminal else None

    def create_ticket(self, user_id: str, summary: str, priority: Priority) -> str:
        """Open a ticket and return its id."""
        ticket_id = f"TCK-{secrets.token_hex(3).upper()}"
        self._tickets[ticket_id] = {
            "ticket_id": ticket_id,
            "user_id": user_id,
            "summary": summary,
            "priority": priority,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": "open",
        }
        return ticket_id

    def get_ticket(self, ticket_id: str) -> dict | None:
        """Return a ticket by id or `None`."""
        ticket = self._tickets.get(ticket_id)
        return dict(ticket) if ticket else None
