"""Interfaces that decouple the agents from the data sources.

Three repositories exist: merchant data read by the Support tools, conversations that
hold the multi-turn memory, and tickets opened by the escalation path. Every method
that returns merchant-scoped data takes the merchant id of the authenticated principal
and answers `None` both for a missing record and for one owned by another merchant, so
that callers can never tell the two apart.
"""

from datetime import date
from typing import Literal, Protocol

from app.state import Conversation, Turn

Priority = Literal["low", "medium", "high"]
TicketStatus = Literal["open", "claimed", "waiting_merchant", "resolved"]
Resolution = Literal["resolved", "needs_merchant"]


class ConflictError(Exception):
    """A state transition that is not allowed, rendered as HTTP 409."""

    def __init__(self, code: str) -> None:
        """Store the stable error code."""
        super().__init__(code)
        self.code = code


class SupportDataRepository(Protocol):
    """Read merchant data."""

    def today(self) -> date:
        """Return the reference date used for relative dates."""
        ...

    def get_user(self, merchant_id: str) -> dict | None:
        """Return the merchant profile or `None`."""
        ...

    def get_settlements(self, merchant_id: str) -> list[dict] | None:
        """Return recent sales with ISO dates or `None`."""
        ...

    def get_terminal(self, merchant_id: str) -> dict | None:
        """Return the terminal status or `None`."""
        ...


class ConversationRepository(Protocol):
    """Persist merchant-owned conversations."""

    def create(self, merchant_id: str) -> str:
        """Create an empty conversation and return its id."""
        ...

    def get(self, conversation_id: str, merchant_id: str) -> Conversation | None:
        """Return the conversation when it exists and belongs to the merchant."""
        ...

    def append(self, conversation_id: str, turn: Turn) -> None:
        """Append one turn."""
        ...

    def update_summary(self, conversation_id: str, summary: str, through_turn: int) -> None:
        """Store the rolling summary and the index of the last turn it covers."""
        ...


class TicketRepository(Protocol):
    """Persist support tickets and their operator workflow."""

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
        """Open a ticket, or return the one already opened for the same turn (`thread_id`)."""
        ...

    def get_ticket(self, ticket_id: str, merchant_id: str | None = None) -> dict | None:
        """Return a ticket; when a merchant id is given, only that merchant's ticket."""
        ...

    def for_conversation(self, conversation_id: str) -> list[dict]:
        """Return the tickets opened from one conversation, newest first."""
        ...

    def list_tickets(self, status: TicketStatus | None = None) -> list[dict]:
        """Return tickets, newest first, optionally filtered by status."""
        ...

    def claim(self, ticket_id: str, operator_key_id: str) -> dict:
        """Assign the ticket to an operator; conflict when another operator holds it."""
        ...

    def reply(
        self, ticket_id: str, operator_key_id: str, message: str, resolution: Resolution
    ) -> dict:
        """Store the operator's reply; `resolved` closes the ticket, `needs_merchant` keeps it."""
        ...
