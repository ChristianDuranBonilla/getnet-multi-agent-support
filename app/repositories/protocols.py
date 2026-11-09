"""Interfaces that decouple the Support tools from the data source."""

from datetime import date
from typing import Literal, Protocol

Priority = Literal["low", "medium", "high"]


class SupportDataRepository(Protocol):
    """Read merchant data and open support tickets.

    Every read method returns `None` for an unknown merchant so that callers can
    answer politely without leaking whether other accounts exist.
    """

    def today(self) -> date:
        """Return the reference date used for relative dates."""
        ...

    def get_user(self, user_id: str) -> dict | None:
        """Return the merchant profile or `None`."""
        ...

    def get_settlements(self, user_id: str) -> list[dict] | None:
        """Return recent sales with ISO dates or `None`."""
        ...

    def get_terminal(self, user_id: str) -> dict | None:
        """Return the terminal status or `None`."""
        ...

    def create_ticket(self, user_id: str, summary: str, priority: Priority) -> str:
        """Open a ticket and return its id."""
        ...

    def get_ticket(self, ticket_id: str) -> dict | None:
        """Return a ticket by id or `None`."""
        ...
