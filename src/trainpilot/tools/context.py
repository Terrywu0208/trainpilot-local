"""Execution context shared by the validators and the MCP servers.

A ``ToolContext`` carries everything the stateless validator needs: where
the run workspaces live, which images are approved, the HMAC secret used
for approval tokens, the fixed resource limits (design B-5.1) and which
runs have already had a successful first launch (so only the first
``apply_rayjob`` of a run is approval-gated, design B-5.4).

Callers may pass a plain ``dict`` instead; :func:`coerce_context` converts
it, keeping only known fields. Pass a real ``ToolContext`` object when the
``applied_run_ids`` bookkeeping must persist across calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Placeholder digests: replace with the real values from
# `docker inspect <image>` when the images are pinned for a real cluster
# (design B-5.2). The validator compares the full `name@digest` reference,
# never the tag alone.
DEFAULT_APPROVED_IMAGES: dict[str, str] = {
    "rayproject/ray:2.41.0-py310-cpu": (
        "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    ),
    "python:3.11-slim-bookworm": (
        "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    ),
    "pytorch/pytorch:2.5.1-cpu": (
        "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    ),
}

# Development default only. Real deployments inject a secret through the
# context; the token issuer (webapp / human CLI) and the action server
# must share the same value.
DEFAULT_APPROVAL_SECRET = "trainpilot-dev-secret-change-me"

JOBS_NAMESPACE = "trainpilot-jobs"


@dataclass
class ToolContext:
    workspace_root: Path | None = None  # e.g. ~/trainpilot-local/workspaces
    approved_images: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_APPROVED_IMAGES)
    )
    secret: str = DEFAULT_APPROVAL_SECRET
    applied_run_ids: set[str] = field(default_factory=set)
    # Fixed limits (design B-5.1); the agent cannot change them.
    max_num_workers: int = 3
    max_cpu_per_pod: float = 2.0
    max_memory_per_pod: str = "3Gi"
    max_file_bytes: int = 1024 * 1024  # 1 MiB per written file (design B-4.8)
    # Globs the purge tool may ever be pointed at (design B-5.4 / B-4 note).
    purge_allowlist: tuple[str, ...] = ("checkpoints/**", "checkpoints/*")
    # Optional FakeClusterClient (or a real client) used by the MCP servers.
    cluster: Any = None


def coerce_context(context: Any) -> ToolContext:
    """Accept None / ToolContext / dict and return a ToolContext."""
    if context is None:
        return ToolContext()
    if isinstance(context, ToolContext):
        return context
    if isinstance(context, dict):
        known = {f for f in ToolContext.__dataclass_fields__}
        kwargs = {k: v for k, v in context.items() if k in known}
        ctx = ToolContext(**kwargs)
        if ctx.workspace_root is not None and not isinstance(ctx.workspace_root, Path):
            ctx.workspace_root = Path(ctx.workspace_root)
        return ctx
    raise TypeError(f"unsupported context type: {type(context)!r}")


def attempt_dir_name(attempt_id: str) -> str:
    """Normalize an attempt identifier to its on-disk directory name."""
    text = str(attempt_id)
    if text.startswith("attempt-"):
        return text
    if text.isdigit():
        return f"attempt-{int(text):02d}"
    return f"attempt-{text}"


def attempt_dir(ctx: ToolContext, run_id: str, attempt_id: str) -> Path:
    """workspaces/<run_id>/attempt-<n>/ for the given run and attempt."""
    if ctx.workspace_root is None:
        raise ValueError("context.workspace_root is not set")
    return Path(ctx.workspace_root) / str(run_id) / attempt_dir_name(attempt_id)
