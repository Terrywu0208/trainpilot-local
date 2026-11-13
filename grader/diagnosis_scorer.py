"""Diagnosis scoring (design D.4). Fully deterministic.

The agent's structured diagnosis record is compared against the scenario
ground truth using the closed code tables in `grader/code_tables.yaml`:

- fault_family: exact match against the ground-truth family.
- root_cause_code: 1.0 for an exact code match, 0.5 partial credit when the
  predicted code belongs to the same family as the truth (the rule is fixed
  here, never decided ad hoc), else 0.
- evidence: precision of the cited evidence against the ground-truth key
  evidence strings / log lines (substring match, case-insensitive).
- proposed_fix_code: 1.0 when it is in the scenario's acceptable_fixes.

A missing diagnosis is reported as `diagnosis_missing`, never raised.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_TABLES = REPO_ROOT / "grader" / "code_tables.yaml"


def load_code_tables(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else CODE_TABLES
    return yaml.safe_load(p.read_text())


def _get(diagnosis: Any, field: str) -> Any:
    if diagnosis is None:
        return None
    if isinstance(diagnosis, dict):
        return diagnosis.get(field)
    return getattr(diagnosis, field, None)


def score_diagnosis(diagnosis: Any, expected: dict[str, Any],
                    tables: dict[str, Any] | None = None) -> dict[str, Any]:
    tables = tables or load_code_tables()
    code_family = tables.get("root_cause_codes", {})
    if diagnosis is None:
        return {
            "diagnosis_missing": True,
            "family_correct": 0.0,
            "root_cause_score": 0.0,
            "evidence_precision": 0.0,
            "fix_acceptable": 0.0,
        }
    true_family = expected.get("fault_family", "unknown")
    true_code = expected.get("root_cause_code", "UNKNOWN")
    pred_family = _get(diagnosis, "fault_family") or "unknown"
    pred_code = _get(diagnosis, "root_cause_code") or "UNKNOWN"
    pred_fix = _get(diagnosis, "proposed_fix_code") or ""
    evidence = _get(diagnosis, "evidence") or []

    family_correct = 1.0 if pred_family == true_family else 0.0
    if pred_code == true_code:
        root_score = 1.0
    elif code_family.get(pred_code) == true_family and pred_code != "UNKNOWN":
        root_score = 0.5
    else:
        root_score = 0.0

    key_evidence = [str(e).lower() for e in expected.get("key_evidence", [])]
    if evidence and key_evidence:
        hits = sum(
            1 for item in evidence
            if any(k in str(item).lower() or str(item).lower() in k
                   for k in key_evidence)
        )
        precision = hits / len(evidence)
    elif evidence:
        precision = 0.0
    else:
        precision = 0.0

    acceptable = set(expected.get("acceptable_fixes", []))
    fix_ok = 1.0 if pred_fix in acceptable else 0.0
    return {
        "diagnosis_missing": False,
        "family_correct": family_correct,
        "root_cause_score": root_score,
        "evidence_precision": round(precision, 4),
        "fix_acceptable": fix_ok,
        "predicted_family": pred_family,
        "predicted_code": pred_code,
    }
