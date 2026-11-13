"""HMAC-signed approval tokens (design B-5.4).

A token binds three things: an approval *scope* (``first_launch``,
``build_image`` or ``purge``), a *binding* dict describing exactly what
was approved (job name, spec hash, image, glob, ...), and an expiry.
The human side (webapp Gate 1/Gate 2, or a small CLI) issues tokens with
:func:`issue_token`; the action server verifies them with
:func:`verify_token` before running a gated tool. Changing any bound
value (for example editing the spec after approval) changes the binding
and invalidates the token.

Token format: ``<base64url(payload-json)>.<base64url(hmac-sha256)>``.
Only the first 8 characters of a token should ever be written to an
audit log (design B-5.4); this module never logs tokens itself.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

SCOPES = ("first_launch", "build_image", "purge")
DEFAULT_TTL_SECONDS = 600  # 10 minutes, per design B-5.4


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def canonical_binding(binding: dict[str, Any]) -> str:
    """Stable JSON form of a binding dict (used for signing/comparing)."""
    return json.dumps(binding, sort_keys=True, separators=(",", ":"), default=str)


def binding_hash(binding: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_binding(binding).encode()).hexdigest()


# --- Binding builders: issuer and verifier MUST use the same helpers. -----

def apply_binding(args: dict[str, Any]) -> dict[str, Any]:
    """Binding for the first-launch approval of apply_rayjob."""
    return {
        "tool": "apply_rayjob",
        "run_id": str(args.get("run_id", "")),
        "job_name": str(args.get("job_name", "")),
        "spec_hash": str(args.get("spec_hash", "")),
    }


def build_binding(args: dict[str, Any]) -> dict[str, Any]:
    """Binding for a build_image approval (every build is gated)."""
    return {
        "tool": "build_image",
        "base_image": str(args.get("base_image", "")),
        "requirements_path": str(args.get("requirements_path", "")),
    }


def purge_binding(run_id: str, attempt_id: str, glob: str) -> dict[str, Any]:
    """Binding for a purge_attempt_artifacts approval."""
    return {
        "tool": "purge_attempt_artifacts",
        "run_id": str(run_id),
        "attempt_id": str(attempt_id),
        "glob": str(glob),
    }


# --- Issue / verify --------------------------------------------------------

def issue_token(
    secret: str | bytes,
    scope: str,
    binding: dict[str, Any],
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    now: float | None = None,
) -> str:
    """Issue a signed approval token for (scope, binding)."""
    if scope not in SCOPES:
        raise ValueError(f"unknown approval scope: {scope!r}")
    issued_at = int(time.time() if now is None else now)
    payload = {
        "scope": scope,
        "binding": binding,
        "binding_hash": binding_hash(binding),
        "iat": issued_at,
        "exp": issued_at + int(ttl_seconds),
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    key = secret.encode() if isinstance(secret, str) else secret
    signature = hmac.new(key, payload_bytes, hashlib.sha256).digest()
    return f"{_b64encode(payload_bytes)}.{_b64encode(signature)}"


def verify_token(
    secret: str | bytes,
    token: str | None,
    scope: str,
    binding: dict[str, Any],
    now: float | None = None,
) -> bool:
    """Return True iff the token is well-formed, correctly signed, not
    expired, and bound to exactly this scope and binding."""
    if not token or not isinstance(token, str) or token.count(".") != 1:
        return False
    try:
        payload_b64, signature_b64 = token.split(".")
        payload_bytes = _b64decode(payload_b64)
        signature = _b64decode(signature_b64)
    except Exception:
        return False
    key = secret.encode() if isinstance(secret, str) else secret
    expected = hmac.new(key, payload_bytes, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        return False
    try:
        payload = json.loads(payload_bytes)
    except Exception:
        return False
    current = int(time.time() if now is None else now)
    if payload.get("exp", 0) <= current:
        return False
    if payload.get("scope") != scope:
        return False
    if canonical_binding(payload.get("binding", {})) != canonical_binding(binding):
        return False
    return payload.get("binding_hash") == binding_hash(binding)
