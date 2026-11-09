"""Data access layer for the Support agent."""

from functools import lru_cache

from app.config import get_settings
from app.repositories.json_repository import JsonSupportRepository
from app.repositories.protocols import SupportDataRepository


@lru_cache
def get_repository() -> SupportDataRepository:
    """Return the process-wide repository instance."""
    return JsonSupportRepository(get_settings().data_dir)


__all__ = ["JsonSupportRepository", "SupportDataRepository", "get_repository"]
