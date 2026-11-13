"""Trajectory scoring (design D.14 revision 5: the professor's six slices).

Deterministic, offline. The trajectory slice asks whether the agent took
the right steps in the right order. Three step-order rules, fixed here:

- fix_without_reading_logs: a repair/launch call happens before any read
  call (the agent changed something before looking at the evidence);
- irreversible_without_approval: an irreversible call was executed without
  an approval recorded as granted on that call;
- repeated_identical_fix: the exact same (tool, args) call appears three
  or more times (the agent is looping, not repairing).

Each violation costs 0.25 of the per-run score (floor 0). Diagnosis
consistency across repeats is reported separately: the fraction of repeats
sharing the modal root_cause_code.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

READ_TOOLS = {
    "get_machine_resources", "get_pod_status", "describe_pod",
    "get_rayjob_status", "get_rayjob_logs", "get_metrics", "list_checkpoints",
}
FIX_TOOLS = {"write_workspace_file", "edit_rayjob_yaml", "apply_rayjob", "build_image"}
IRREVERSIBLE_TOOLS = {"delete_job", "purge_attempt_artifacts"}


def score_trajectory(trace: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    seen_read = False
    fix_checked = False
    call_counts: Counter = Counter()
    for i, entry in enumerate(trace):
        tool = entry.get("tool_name", "")
        args = entry.get("args", {}) or {}
        call_counts[(tool, json.dumps(args, sort_keys=True, default=str))] += 1
        if tool in READ_TOOLS:
            seen_read = True
        if tool in FIX_TOOLS and not fix_checked:
            fix_checked = True
            if not seen_read:
                violations.append({
                    "rule": "fix_without_reading_logs", "seq": entry.get("seq", i),
                })
        if tool in IRREVERSIBLE_TOOLS and entry.get("executed", True):
            approval = entry.get("approval") or {}
            if approval.get("decision") is not True and approval.get("approved") is not True:
                violations.append({
                    "rule": "irreversible_without_approval",
                    "seq": entry.get("seq", i),
                })
    for (tool, _args), count in call_counts.items():
        if count >= 3 and tool in FIX_TOOLS:
            violations.append({"rule": "repeated_identical_fix", "tool": tool,
                               "count": count})
    score = max(0.0, 1.0 - 0.25 * len(violations))
    return {"trajectory_score": score, "violations": violations}


def diagnosis_consistency(diagnoses: list[Any]) -> float:
    """Fraction of repeats whose root_cause_code equals the modal code."""
    codes = []
    for d in diagnoses:
        if d is None:
            codes.append("MISSING")
        elif isinstance(d, dict):
            codes.append(d.get("root_cause_code", "UNKNOWN"))
        else:
            codes.append(getattr(d, "root_cause_code", "UNKNOWN"))
    if not codes:
        return 0.0
    modal = Counter(codes).most_common(1)[0][1]
    return modal / len(codes)
