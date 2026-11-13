"""Failure memory (design section A.5).

Entries live in two places:

- in the graph state (the working copy, outcomes updated as the run
  progresses: pending -> applied -> succeeded / failed), and
- in ``<run_dir>/failures.jsonl``, which is strictly append-only: the
  first write of an entry and every later outcome update are appended
  as new lines keyed by ``failure_id``; readers take the last line per
  ``failure_id``. The file is never rewritten or truncated.

At most ``MAX_INJECTED_FAILURES`` entries are injected into a prompt,
and fixes that already failed are marked ``[DO NOT REPEAT]``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .state import MAX_INJECTED_FAILURES

# Deterministic mapping from the agent-side Diagnosis to the closed code
# tables used by failure memory and the section D scorers.
_CODE_TABLE = {
    # (error_class, fix marker) -> (fault_family, root_cause_code, fix_code)
    ("CODE_BUG", "backend"): ("gpu_only_code", "NCCL_ON_CPU", "SET_BACKEND_GLOO"),
    ("OOM", "batch"): ("memory_limits", "OOMKILLED_137", "REDUCE_BATCH_INCREASE_ACCUM"),
    ("DEPENDENCY", "rebuild"): (
        "dependency_conflict", "DEP_VERSION_MISMATCH", "REBUILD_IMAGE_WITH_LOCKFILE"),
    ("DATA_SCHEMA", "sanitize"): (
        "data_or_training_error", "DATA_LABEL_INVALID", "SANITIZE_LABELS"),
    ("DATA_SCHEMA", "learning_rate"): (
        "data_or_training_error", "TRAIN_LOSS_NAN", "RESET_LEARNING_RATE"),
    ("TIMEOUT", ""): ("unknown", "TIMEOUT", "NONE"),
    ("UNKNOWN", ""): ("unknown", "UNKNOWN", "NONE"),
}


def classify_codes(diagnosis: dict[str, Any]) -> tuple[str, str, str]:
    """Map a Diagnosis dump to (fault_family, root_cause_code, fix_code)."""
    raw_class = diagnosis.get("error_class", "UNKNOWN")
    error_class = str(getattr(raw_class, "value", raw_class))
    fix = diagnosis.get("proposed_fix") or {}
    marker = ""
    if "backend" in fix:
        marker = "backend"
    elif "per_worker_batch_size" in fix or "gradient_accumulation_steps" in fix:
        marker = "batch"
    elif "rebuild_image_with_lockfile" in fix:
        marker = "rebuild"
    elif "sanitize_labels" in fix:
        marker = "sanitize"
    elif "learning_rate" in fix:
        marker = "learning_rate"
    hit = _CODE_TABLE.get((error_class, marker)) or _CODE_TABLE.get((error_class, ""))
    if hit is None:
        hit = ("unknown", error_class + "_UNMAPPED", "NONE")
    return hit


def make_entry(diagnosis: dict[str, Any], run_id: str, attempt: int,
               seq: int) -> dict[str, Any]:
    fault_family, root_cause_code, fix_code = classify_codes(diagnosis)
    return {
        "failure_id": f"F{attempt:02d}-{seq:02d}",
        "run_id": run_id,
        "attempt": attempt,
        "timestamp_epoch": time.time(),
        "fault_label": fault_family,
        "symptom": diagnosis.get("symptom", ""),
        "cause": diagnosis.get("root_cause", ""),
        "root_cause_code": root_cause_code,
        "error_class": str(getattr(
            diagnosis.get("error_class", "UNKNOWN"), "value",
            diagnosis.get("error_class", "UNKNOWN"))),
        "evidence": list(diagnosis.get("evidence", [])),
        "fix_code": fix_code,
        "fix_applied": None,
        "outcome": "pending",
    }


def append_entry(run_dir: Path | None, entry: dict[str, Any]) -> None:
    """Append one JSON line for ``entry`` to failures.jsonl (never rewrites)."""
    if run_dir is None:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "failures.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def load_entries(run_dir: Path | None) -> list[dict[str, Any]]:
    """Read failures.jsonl back, keeping the last line per failure_id."""
    if run_dir is None:
        return []
    path = run_dir / "failures.jsonl"
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        fid = entry.get("failure_id", "")
        if fid not in latest:
            order.append(fid)
        latest[fid] = entry
    return [latest[fid] for fid in order]


def select_for_prompt(entries: list[dict[str, Any]],
                      error_class: str | None = None,
                      limit: int = MAX_INJECTED_FAILURES) -> list[dict[str, Any]]:
    """Pick the entries to inject: most recent 3 plus same-class history,
    capped at ``limit`` entries total."""
    if not entries:
        return []
    recent = entries[-3:]
    picked: list[dict[str, Any]] = list(recent)
    if error_class:
        for entry in entries:
            if len(picked) >= limit:
                break
            if entry.get("error_class") == error_class and entry not in picked:
                picked.append(entry)
    return picked[:limit]


def format_for_prompt(entries: list[dict[str, Any]]) -> str:
    """Render entries for a prompt: four fields each, failed fixes flagged."""
    lines: list[str] = []
    for entry in entries:
        marker = " [DO NOT REPEAT]" if entry.get("outcome") == "failed" else ""
        lines.append(
            "- symptom: {symptom} | cause: {cause} | fix_applied: {fix} | "
            "outcome: {outcome}{marker}".format(
                symptom=entry.get("symptom", ""),
                cause=entry.get("cause", ""),
                fix=entry.get("fix_applied"),
                outcome=entry.get("outcome", "pending"),
                marker=marker,
            )
        )
    return "\n".join(lines)
