"""Task schema v2 (design D.1/D.6, steps 1-2).

`TaskSpec` is defined once in `trainpilot.schemas` and re-exported here so
lab50 tasks, benchmark data points and the safety suite all share one shape.
`validate_task` is the checker used by the benchmark generator and by the
runner when loading task files: it returns a list of human-readable errors
(empty list means the row is valid).
"""
from __future__ import annotations

from typing import Any

from trainpilot.schemas import TaskSpec

__all__ = ["TaskSpec", "validate_task", "row_to_taskspec"]

GRADING_TYPES = {"exact", "tool", "rubric", "unanswerable"}
DIFFICULTIES = {"easy", "hard", "edge", "unanswerable"}
SAFE_BEHAVIORS = {"refuse", "ask_confirmation", "answer_without_harm"}
SAFETY_CATEGORIES = {
    "prompt_injection", "memory_poisoning", "data_leakage",
    "harmful_request", "risky_action", "jailbreak", "harmless_control",
}


def row_to_taskspec(row: dict[str, Any]) -> TaskSpec:
    """Convert a JSONL row (lab task or benchmark data point) to TaskSpec.

    Data points nest the prompt under ``input.initial_request``; lab tasks
    carry a flat ``prompt``. Both are accepted.
    """
    data = dict(row)
    if not data.get("prompt"):
        inp = data.get("input") or {}
        data["prompt"] = inp.get("initial_request", "")
        follow = inp.get("follow_ups") or []
        if follow:
            data["prompt"] = data["prompt"] + "\n" + "\n".join(follow)
    known = set(TaskSpec.model_fields)
    return TaskSpec(**{k: v for k, v in data.items() if k in known})


def validate_task(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not row.get("id"):
        errors.append("missing id")
    prompt = row.get("prompt") or (row.get("input") or {}).get("initial_request")
    if not prompt:
        errors.append("missing prompt (prompt or input.initial_request)")
    grading = row.get("grading_type", "rubric")
    if grading not in GRADING_TYPES:
        errors.append(f"invalid grading_type: {grading!r}")
    difficulty = row.get("difficulty", "easy")
    if difficulty not in DIFFICULTIES:
        errors.append(f"invalid difficulty: {difficulty!r}")
    if not isinstance(row.get("expected_state", {}), dict):
        errors.append("expected_state must be an object")
    if grading == "exact" and not row.get("expected"):
        errors.append("exact tasks require an expected string")
    if grading == "tool" and not row.get("expected"):
        errors.append("tool tasks require an expected tool name")
    if grading == "unanswerable" and not (
        row.get("expected") or row.get("notes")
    ):
        errors.append("unanswerable tasks require expected or notes explaining why")
    if row.get("split") == "safety" or row.get("safety_category"):
        cat = row.get("safety_category")
        if cat not in SAFETY_CATEGORIES:
            errors.append(f"safety task has invalid safety_category: {cat!r}")
        if row.get("safe_behavior") not in SAFE_BEHAVIORS:
            errors.append("safety task requires safe_behavior in "
                          "refuse | ask_confirmation | answer_without_harm")
    return errors
