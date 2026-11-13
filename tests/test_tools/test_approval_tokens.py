"""Approval token tests: round trip, tamper, expiry, binding (B-5.4)."""
from __future__ import annotations

import base64
import json

import pytest

from trainpilot.tools import approval_tokens as tokens

SECRET = "unit-test-secret"
BINDING = {"tool": "apply_rayjob", "run_id": "r-1", "job_name": "tp-x-a01-abc", "spec_hash": "abc"}


def test_issue_and_verify_roundtrip():
    token = tokens.issue_token(SECRET, "first_launch", BINDING)
    assert tokens.verify_token(SECRET, token, "first_launch", BINDING)


def test_verify_rejects_wrong_secret():
    token = tokens.issue_token(SECRET, "first_launch", BINDING)
    assert not tokens.verify_token("other-secret", token, "first_launch", BINDING)


def test_verify_rejects_wrong_scope():
    token = tokens.issue_token(SECRET, "first_launch", BINDING)
    assert not tokens.verify_token(SECRET, token, "build_image", BINDING)


def test_verify_rejects_changed_binding():
    token = tokens.issue_token(SECRET, "first_launch", BINDING)
    changed = dict(BINDING, spec_hash="tampered")
    assert not tokens.verify_token(SECRET, token, "first_launch", changed)


def test_verify_rejects_expired_token():
    token = tokens.issue_token(SECRET, "first_launch", BINDING, ttl_seconds=60, now=1000)
    assert tokens.verify_token(SECRET, token, "first_launch", BINDING, now=1030)
    assert not tokens.verify_token(SECRET, token, "first_launch", BINDING, now=2000)


def test_verify_rejects_tampered_payload():
    token = tokens.issue_token(SECRET, "purge", {"tool": "purge_attempt_artifacts"})
    payload_b64, signature_b64 = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    payload["exp"] = payload["exp"] + 100000  # attacker extends expiry
    forged_b64 = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    forged = f"{forged_b64}.{signature_b64}"
    assert not tokens.verify_token(SECRET, forged, "purge", {"tool": "purge_attempt_artifacts"})


def test_verify_rejects_garbage():
    assert not tokens.verify_token(SECRET, None, "purge", {})
    assert not tokens.verify_token(SECRET, "", "purge", {})
    assert not tokens.verify_token(SECRET, "not-a-token", "purge", {})
    assert not tokens.verify_token(SECRET, "a.b.c", "purge", {})


def test_issue_rejects_unknown_scope():
    with pytest.raises(ValueError):
        tokens.issue_token(SECRET, "everything", {})


def test_binding_helpers_are_deterministic():
    args = {"run_id": "r-1", "job_name": "tp-x", "spec_hash": "h"}
    assert tokens.apply_binding(args) == tokens.apply_binding(dict(args))
    assert tokens.binding_hash({"a": 1, "b": 2}) == tokens.binding_hash({"b": 2, "a": 1})
