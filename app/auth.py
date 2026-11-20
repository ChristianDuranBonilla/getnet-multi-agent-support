"""Authentication: API keys, principals and per-key rate limiting.

The principal built here is the only source of the merchant identity in the whole
service. Agents, tools and repositories never read an identity from the request body
or from a model output; they read `AgentState["principal"]`.
"""

import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, Header, Request
from pydantic import BaseModel

from app.config import get_settings

OPERATOR_SCOPE = "operator"
CHAT_SCOPE = "chat"


class Principal(BaseModel):
    """Authenticated caller.

    Attributes:
        merchant_id: Merchant the credential belongs to; empty for operators.
        auth_method: How the caller was authenticated.
        scopes: Operations the credential allows.
        key_id: Short, non-secret identifier of the credential for logs and revocation.
    """

    merchant_id: str
    auth_method: Literal["api_key"]
    scopes: tuple[str, ...]
    key_id: str

    def has_scope(self, scope: str) -> bool:
        """Return whether the credential carries the scope."""
        return scope in self.scopes


class AuthError(Exception):
    """Authentication or authorization failure rendered as an HTTP error envelope."""

    def __init__(self, status: int, code: str, retry_after: int | None = None) -> None:
        """Store the HTTP status, the stable error code and an optional retry hint.

        Args:
            status: HTTP status code.
            code: Stable machine-readable error code.
            retry_after: Seconds the client should wait, for 429 responses.
        """
        super().__init__(code)
        self.status = status
        self.code = code
        self.retry_after = retry_after


def hash_key(key: str) -> str:
    """Return the SHA-256 hex digest under which a key is stored."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_id_of(key_hash: str) -> str:
    """Return the public identifier of a stored key: the first eight hex characters."""
    return key_hash[:8]


def generate_key(prefix: str = "gk") -> str:
    """Return a fresh plaintext API key; it carries no information about its owner."""
    return f"{prefix}_{secrets.token_hex(16)}"


class KeyStore:
    """Hashed key records from `data/api_keys.json`, reloaded when the file changes.

    Reloading on modification time keeps revocation effective without a restart while
    avoiding a disk read per request.
    """

    def __init__(self, path: Path) -> None:
        """Remember the file; nothing is read until the first lookup."""
        self._path = path
        self._mtime: float | None = None
        self._records: dict[str, dict] = {}
        self._lock = threading.Lock()

    def records(self) -> dict[str, dict]:
        """Return the mapping from key hash to record, refreshed when the file changed."""
        mtime = self._path.stat().st_mtime
        with self._lock:
            if mtime != self._mtime:
                with self._path.open(encoding="utf-8") as handle:
                    data = json.load(handle)
                data.pop("_note", None)
                self._records = data
                self._mtime = mtime
            return self._records


@lru_cache
def get_key_store() -> KeyStore:
    """Return the process-wide key store."""
    return KeyStore(get_settings().data_dir / "api_keys.json")


def load_api_keys() -> dict[str, dict]:
    """Return the current key records (`merchant_id`, `scopes`, `status`, `label`) by hash."""
    return get_key_store().records()


PLACEHOLDER_OPERATOR_KEYS = frozenset({"", "op_dev_change_me"})


def operator_key_configured() -> bool:
    """Return whether a real operator credential is set (placeholders never authenticate)."""
    return get_settings().operator_api_key not in PLACEHOLDER_OPERATOR_KEYS


def authenticate(api_key: str | None) -> Principal:
    """Resolve a presented API key to a principal.

    Args:
        api_key: Value of the `X-API-Key` header, if any.

    Returns:
        The principal the key belongs to.

    Raises:
        AuthError: 401 when the key is missing or unknown, 403 when it is revoked.
    """
    if not api_key:
        raise AuthError(401, "unauthorized")
    presented = hash_key(api_key)
    if operator_key_configured() and hmac.compare_digest(
        presented, hash_key(get_settings().operator_api_key)
    ):
        return Principal(
            merchant_id="",
            auth_method="api_key",
            scopes=(OPERATOR_SCOPE,),
            key_id=key_id_of(presented),
        )
    record = None
    for stored, item in load_api_keys().items():
        if hmac.compare_digest(presented, stored):
            record = item
    if record is None:
        raise AuthError(401, "unauthorized")
    if record.get("status") != "active":
        raise AuthError(403, "key_revoked")
    scopes = tuple(record.get("scopes", [CHAT_SCOPE]))
    merchant_id = record.get("merchant_id", "") if CHAT_SCOPE in scopes else ""
    if CHAT_SCOPE in scopes and not merchant_id:
        raise AuthError(401, "unauthorized")
    return Principal(
        merchant_id=merchant_id,
        auth_method="api_key",
        scopes=scopes,
        key_id=key_id_of(presented),
    )


class RateLimiter:
    """Token bucket per key, in process.

    Each key starts with `burst` tokens and earns `per_minute / 60` tokens per second up
    to `burst`. A request consumes one token; when none is available the caller is told
    how many seconds to wait. The clock is injectable so tests do not sleep.
    """

    def __init__(
        self, per_minute: float, burst: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        """Configure the bucket parameters.

        Args:
            per_minute: Sustained requests per minute.
            burst: Maximum requests served back to back.
            clock: Monotonic time source in seconds.
        """
        self._rate = per_minute / 60.0
        self._capacity = float(burst)
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str) -> int | None:
        """Consume one token for the key.

        Args:
            key: Bucket identifier, normally the principal's `key_id`.

        Returns:
            `None` when the request is allowed, otherwise the seconds to wait.
        """
        with self._lock:
            now = self._clock()
            tokens, updated = self._buckets.get(key, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - updated) * self._rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return None
            self._buckets[key] = (tokens, now)
            return max(1, math.ceil((1.0 - tokens) / self._rate))


@lru_cache
def get_rate_limiter() -> RateLimiter:
    """Return the process-wide rate limiter configured from settings."""
    settings = get_settings()
    return RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)


def authenticated_principal(
    request: Request, x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None
) -> Principal:
    """FastAPI dependency: any authenticated caller, after the rate limit.

    Failed authentications are throttled per client address so that keys cannot be
    guessed at line rate.

    Raises:
        AuthError: 401/403 from `authenticate`, 429 when the key exceeded its rate limit.
    """
    try:
        principal = authenticate(x_api_key)
    except AuthError as exc:
        client = request.client.host if request.client else "unknown"
        if exc.status == 401 and get_rate_limiter().acquire(f"auth-fail:{client}") is not None:
            raise AuthError(429, "rate_limited", retry_after=60) from exc
        raise
    retry_after = get_rate_limiter().acquire(principal.key_id)
    if retry_after is not None:
        raise AuthError(429, "rate_limited", retry_after=retry_after)
    request.state.principal = principal
    return principal


def merchant_principal(
    principal: Annotated[Principal, Depends(authenticated_principal)],
) -> Principal:
    """FastAPI dependency: an authenticated merchant with the `chat` scope.

    Raises:
        AuthError: 403 when the credential lacks the scope.
    """
    if not principal.has_scope(CHAT_SCOPE):
        raise AuthError(403, "forbidden")
    return principal


def operator_principal(
    principal: Annotated[Principal, Depends(authenticated_principal)],
) -> Principal:
    """FastAPI dependency: an authenticated operator.

    Raises:
        AuthError: 403 when the credential lacks the scope.
    """
    if not principal.has_scope(OPERATOR_SCOPE):
        raise AuthError(403, "forbidden")
    return principal
