"""Purge tests: allowlist glob, approval binding, escape refusal (B-5.4)."""
from __future__ import annotations

from pathlib import Path

import pytest

from trainpilot.tools import approval_tokens
from trainpilot.tools.context import ToolContext
from trainpilot.tools.purge import purge_attempt_artifacts

SECRET = "purge-secret"
RUN = "r-1"
ATTEMPT = "attempt-01"
GLOB = "checkpoints/**"


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "attempt-01"
    ckpt = ws / "checkpoints" / "checkpoint-old"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"x" * 100)
    (ckpt / "config.json").write_text("{}")
    keep = ws / "notes"
    keep.mkdir()
    (keep / "keep.txt").write_text("do not delete")
    return ws


def _token(glob: str = GLOB) -> str:
    return approval_tokens.issue_token(
        SECRET, "purge", approval_tokens.purge_binding(RUN, ATTEMPT, glob)
    )


def test_purge_deletes_only_matching_files(workspace):
    result = purge_attempt_artifacts(
        workspace, GLOB, _token(), secret=SECRET, run_id=RUN, attempt_id=ATTEMPT
    )
    assert result.ok
    deleted = {entry["path"] for entry in result.data["deleted"]}
    assert deleted == {
        "checkpoints/checkpoint-old/model.safetensors",
        "checkpoints/checkpoint-old/config.json",
    }
    sizes = {entry["path"]: entry["size_bytes"] for entry in result.data["deleted"]}
    assert sizes["checkpoints/checkpoint-old/model.safetensors"] == 100
    assert result.data["total_bytes_freed"] == 102
    assert not (workspace / "checkpoints" / "checkpoint-old" / "model.safetensors").exists()
    # Files outside the glob are untouched.
    assert (workspace / "notes" / "keep.txt").read_text() == "do not delete"


def test_purge_requires_valid_token(workspace):
    result = purge_attempt_artifacts(
        workspace, GLOB, None, secret=SECRET, run_id=RUN, attempt_id=ATTEMPT
    )
    assert not result.ok
    assert result.error_code == "APPROVAL_REQUIRED"
    assert (workspace / "checkpoints" / "checkpoint-old" / "model.safetensors").exists()


def test_purge_token_bound_to_glob(workspace):
    other_glob_token = _token("checkpoints/*")
    result = purge_attempt_artifacts(
        workspace, GLOB, other_glob_token, secret=SECRET, run_id=RUN, attempt_id=ATTEMPT
    )
    assert not result.ok
    assert result.error_code == "APPROVAL_REQUIRED"


def test_purge_refuses_glob_outside_allowlist(workspace):
    token = _token("logs/**")
    result = purge_attempt_artifacts(
        workspace, "logs/**", token, secret=SECRET, run_id=RUN, attempt_id=ATTEMPT
    )
    assert not result.ok
    assert result.error_code == "PATH_NOT_ALLOWED"


def test_purge_refuses_traversal_glob(workspace):
    result = purge_attempt_artifacts(
        workspace, "../**", _token(), secret=SECRET, run_id=RUN, attempt_id=ATTEMPT
    )
    assert not result.ok
    assert result.error_code == "PATH_NOT_ALLOWED"


def test_purge_refuses_symlink_escape(workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret_file = outside / "secret.txt"
    secret_file.write_text("secret")
    (workspace / "checkpoints" / "escape.txt").symlink_to(secret_file)
    result = purge_attempt_artifacts(
        workspace, GLOB, _token(), secret=SECRET, run_id=RUN, attempt_id=ATTEMPT
    )
    assert not result.ok
    assert result.error_code == "PATH_NOT_ALLOWED"
    # Nothing at all was deleted, inside or outside.
    assert secret_file.read_text() == "secret"
    assert (workspace / "checkpoints" / "checkpoint-old" / "model.safetensors").exists()


def test_purge_empty_match_is_ok(workspace):
    result = purge_attempt_artifacts(
        workspace, "checkpoints/*.tmp", _token("checkpoints/*.tmp"),
        secret=SECRET, run_id=RUN, attempt_id=ATTEMPT,
    )
    # 'checkpoints/*.tmp' matches the allowlist entry 'checkpoints/*' and
    # simply finds nothing.
    assert result.ok
    assert result.data["deleted"] == []


def test_purge_respects_context_secret(workspace):
    ctx = ToolContext(secret="ctx-secret")
    token = approval_tokens.issue_token(
        "ctx-secret", "purge", approval_tokens.purge_binding(RUN, ATTEMPT, GLOB)
    )
    result = purge_attempt_artifacts(
        workspace, GLOB, token, run_id=RUN, attempt_id=ATTEMPT, context=ctx
    )
    assert result.ok
