"""Repository backed by the JSON files under `data/`."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path


class JsonSupportRepository:
    """Merchant data loaded from `users.json`, `settlements.json` and `terminals.json`.

    Relative day offsets in the settlement file are converted to ISO dates at read
    time so that scenarios about "yesterday" stay valid on any day. Methods take the
    merchant id of the authenticated principal and never any other identifier.
    """

    def __init__(self, data_dir: Path, today: date | None = None) -> None:
        """Load the JSON files.

        Args:
            data_dir: Directory containing the three JSON files.
            today: Pinned reference date for relative offsets; `None` uses the current
                UTC date at every call, so a long-running process never drifts.
        """
        self._today = today
        self._users = self._load(data_dir / "users.json")
        self._settlements = self._load(data_dir / "settlements.json")
        self._terminals = self._load(data_dir / "terminals.json")

    @staticmethod
    def _load(path: Path) -> dict:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        data.pop("_note", None)
        return data

    def _to_date(self, days_ago: int | None) -> str | None:
        if days_ago is None:
            return None
        return (self.today() - timedelta(days=days_ago)).isoformat()

    def today(self) -> date:
        """Return the pinned reference date, or the current UTC date when none was pinned."""
        return self._today or datetime.now(UTC).date()

    def get_user(self, merchant_id: str) -> dict | None:
        """Return the merchant profile or `None`."""
        user = self._users.get(merchant_id)
        return dict(user, user_id=merchant_id) if user else None

    def get_settlements(self, merchant_id: str) -> list[dict] | None:
        """Return recent sales with ISO dates or `None`."""
        sales = self._settlements.get(merchant_id)
        if sales is None:
            return None
        result = []
        for sale in sales:
            item = {k: v for k, v in sale.items() if not k.endswith("days_ago")}
            item["sale_date"] = self._to_date(sale["days_ago"])
            item["expected_deposit_date"] = self._to_date(sale.get("expected_deposit_days_ago"))
            result.append(item)
        return result

    def get_terminal(self, merchant_id: str) -> dict | None:
        """Return the terminal status or `None`."""
        terminal = self._terminals.get(merchant_id)
        return dict(terminal) if terminal else None
