"""Issue or revoke API keys in `data/api_keys.json`.

Usage:
    python scripts/issue_api_key.py issue <merchant_id> [--label TEXT]
    python scripts/issue_api_key.py issue --operator [--label TEXT]
    python scripts/issue_api_key.py revoke <key_id>

Merchant keys carry the `chat` scope and are bound to one merchant; operator keys carry
the `operator` scope and give each operator an identity of their own (`key_id`), which
is what the ticket claims and the audit log record. The plaintext key is printed once
and never stored; only its SHA-256 hash is kept.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth import generate_key, hash_key, key_id_of  # noqa: E402
from app.config import get_settings  # noqa: E402


def _store_path() -> Path:
    return get_settings().data_dir / "api_keys.json"


def _load() -> dict:
    with _store_path().open(encoding="utf-8") as handle:
        return json.load(handle)


def _save(store: dict) -> None:
    _store_path().write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")


def issue(merchant_id: str | None, label: str) -> str:
    """Create a key, store its hash and return the plaintext once.

    Args:
        merchant_id: Owner of a merchant key; `None` issues an operator key.
        label: Human-readable note stored with the hash.

    Returns:
        The plaintext key.
    """
    store = _load()
    key = generate_key("gk" if merchant_id else "op")
    store[hash_key(key)] = {
        "merchant_id": merchant_id or "",
        "scopes": ["chat"] if merchant_id else ["operator"],
        "status": "active",
        "label": label or (f"issued for {merchant_id}" if merchant_id else "operator"),
    }
    _save(store)
    return key


def revoke(key_id: str) -> bool:
    """Mark every key whose id matches as revoked; return whether one was found."""
    store = _load()
    found = False
    for stored, record in store.items():
        if stored != "_note" and key_id_of(stored) == key_id:
            record["status"] = "revoked"
            found = True
    if found:
        _save(store)
    return found


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    issue_parser = sub.add_parser("issue")
    issue_parser.add_argument("merchant_id", nargs="?")
    issue_parser.add_argument("--operator", action="store_true", help="Issue an operator key.")
    issue_parser.add_argument("--label", default="")
    revoke_parser = sub.add_parser("revoke")
    revoke_parser.add_argument("key_id")
    args = parser.parse_args(argv)
    if args.command == "issue":
        if bool(args.merchant_id) == args.operator:
            parser.error("give a merchant_id or --operator, not both")
        key = issue(None if args.operator else args.merchant_id, args.label)
        print(f"key_id={key_id_of(hash_key(key))}")
        print(key)
        return 0
    if revoke(args.key_id):
        print(f"revoked {args.key_id}")
        return 0
    print(f"no key with id {args.key_id}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
