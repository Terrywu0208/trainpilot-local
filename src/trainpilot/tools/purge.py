"""purge_attempt_artifacts: the reserved, approval-gated deletion tool.

This is the tool behind the proposal's second demo case (free disk space
by deleting old checkpoints). It is deliberately NOT part of the 13-tool
registry (design B-4): it may only ever delete files that

1. live under the given workspace directory (symlink escapes refused),
2. match a glob that is itself on the context allowlist
   (``checkpoints/**`` / ``checkpoints/*`` by default), and
3. are covered by a valid HMAC approval token bound to
   (run_id, attempt_id, glob) - design B-5.4.

Host-side logs (``logs/attempts/`` in design B-6) are outside any
workspace and therefore can never be reached by this tool.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from trainpilot.schemas import ToolResult
from trainpilot.tools import approval_tokens
from trainpilot.tools.context import DEFAULT_APPROVAL_SECRET, ToolContext, coerce_context
from trainpilot.tools.errors import (
    APPROVAL_REQUIRED,
    PATH_NOT_ALLOWED,
    VALIDATION_FAILED,
    err_result,
    ok_result,
)

TOOL_NAME = "purge_attempt_artifacts"


def purge_attempt_artifacts(
    workspace: str | Path,
    glob: str,
    approval_token: str | None,
    *,
    secret: str = DEFAULT_APPROVAL_SECRET,
    run_id: str = "",
    attempt_id: str = "",
    context: Any = None,
) -> ToolResult:
    """Delete files under ``workspace`` matching an allowlisted ``glob``.

    Returns a ToolResult whose data lists every deleted file with its
    size in bytes. Any refusal (bad token, glob not allowlisted, a match
    escaping the workspace) deletes nothing.
    """
    ctx = coerce_context(context) if context is not None else ToolContext()
    effective_secret = secret
    if context is not None:
        effective_secret = ctx.secret

    if not isinstance(glob, str) or not glob or glob.startswith("/") or ".." in Path(glob).parts:
        return err_result(TOOL_NAME, PATH_NOT_ALLOWED, f"illegal purge glob: {glob!r}")
    if not any(fnmatch.fnmatch(glob, allowed) for allowed in ctx.purge_allowlist):
        return err_result(
            TOOL_NAME, PATH_NOT_ALLOWED,
            f"glob {glob!r} is not on the purge allowlist {list(ctx.purge_allowlist)}",
        )

    binding = approval_tokens.purge_binding(run_id, attempt_id, glob)
    if not approval_tokens.verify_token(effective_secret, approval_token, "purge", binding):
        return err_result(
            TOOL_NAME, APPROVAL_REQUIRED,
            "purge requires a valid approval token (scope 'purge', bound to "
            "run_id + attempt_id + glob)",
        )

    root = Path(workspace)
    if not root.is_dir():
        return err_result(TOOL_NAME, VALIDATION_FAILED, f"workspace {str(root)!r} is not a directory")
    root = root.resolve()

    # Collect candidate files first; refuse the whole operation if any
    # match resolves outside the workspace (e.g. through a symlink).
    candidates: list[Path] = []
    for match in sorted(root.glob(glob)):
        if match.is_dir() and not match.is_symlink():
            candidates.extend(p for p in sorted(match.rglob("*")) if p.is_file())
        elif match.is_file() or match.is_symlink():
            candidates.append(match)
    resolved: list[Path] = []
    for candidate in candidates:
        target = candidate.resolve()
        if target != root and root not in target.parents:
            return err_result(
                TOOL_NAME, PATH_NOT_ALLOWED,
                f"match {str(candidate)!r} resolves outside the workspace; refusing the whole purge",
            )
        resolved.append(target)

    deleted: list[dict[str, Any]] = []
    total_bytes = 0
    seen: set[Path] = set()
    for target in resolved:
        if target in seen or not target.is_file():
            continue
        seen.add(target)
        size = target.stat().st_size
        target.unlink()
        total_bytes += size
        deleted.append({"path": target.relative_to(root).as_posix(), "size_bytes": size})

    return ok_result(
        TOOL_NAME,
        {
            "workspace": str(root),
            "glob": glob,
            "deleted": deleted,
            "deleted_count": len(deleted),
            "total_bytes_freed": total_bytes,
        },
    )
