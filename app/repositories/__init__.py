"""Data access layer: merchant data, conversations and tickets."""

from functools import lru_cache

from app.config import get_settings
from app.repositories.json_repository import JsonSupportRepository
from app.repositories.protocols import (
    ConflictError,
    ConversationRepository,
    SupportDataRepository,
    TicketRepository,
)
from app.repositories.sqlite_repository import (
    SqliteConversationRepository,
    SqliteStore,
    SqliteTicketRepository,
)


@lru_cache
def get_repository() -> SupportDataRepository:
    """Return the process-wide merchant data repository."""
    return JsonSupportRepository(get_settings().data_dir)


@lru_cache
def get_store() -> SqliteStore:
    """Return the process-wide SQLite store under `state_dir`."""
    return SqliteStore(get_settings().state_dir / "app.sqlite")


@lru_cache
def get_conversation_repository() -> ConversationRepository:
    """Return the process-wide conversation repository."""
    return SqliteConversationRepository(get_store())


@lru_cache
def get_ticket_repository() -> TicketRepository:
    """Return the process-wide ticket repository."""
    return SqliteTicketRepository(get_store())


__all__ = [
    "ConflictError",
    "ConversationRepository",
    "JsonSupportRepository",
    "SqliteConversationRepository",
    "SqliteStore",
    "SqliteTicketRepository",
    "SupportDataRepository",
    "TicketRepository",
    "get_conversation_repository",
    "get_repository",
    "get_store",
    "get_ticket_repository",
]
