"""pass@k / pass^k statistics (design D.5; professor standard F, R8).

Definitions (combinatorial, per task with n runs of which c pass):

    pass@k = 1 - C(n - c, k) / C(n, k)     (at least one of k trials passes)
    pass^k = C(c, k) / C(n, k)             (all k trials pass)

averaged over tasks. Worked example from the Week 6 lecture: a task run
4 times with outcomes [1, 1, 0, 0] gives pass@2 = 5/6 and pass^2 = 1/6.

When k equals the number of repeats n, the formulas reduce to the simple
lab form: pass@n = fraction of tasks passing at least once, pass^n =
fraction passing every time. The lab (3 repeats) reports pass@3 / pass^3;
the final experiment reports scenario pass^3, which proposal language calls
"share of scenarios passing all 3 runs" — the same quantity.
"""
from __future__ import annotations

import math
from typing import Any


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def pass_at_k_value(outcomes: list, k: int) -> tuple[float, float]:
    """Return (pass@k, pass^k) for one task's list of run outcomes."""
    n = len(outcomes)
    if n == 0 or k <= 0 or k > n:
        return (0.0, 0.0)
    c = sum(1 for o in outcomes if o)
    total = _comb(n, k)
    if total == 0:
        return (0.0, 0.0)
    pass_at = 1.0 - _comb(n - c, k) / total
    pass_pow = _comb(c, k) / total
    return (pass_at, pass_pow)


def pass_at_k_stats(outcomes: list[list], ks: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Aggregate pass@k / pass^k over tasks.

    `outcomes` is a list with one entry per task; each entry is the list of
    that task's run outcomes (bools or 0/1). Returns per-k means plus the
    convenience keys for the largest k (the repeat count).
    """
    tasks = [list(o) for o in outcomes if o]
    result: dict[str, Any] = {"n_tasks": len(tasks), "per_k": {}}
    if not tasks:
        return result
    n_runs = max(len(o) for o in tasks)
    wanted = ks if ks is not None else tuple(range(1, n_runs + 1))
    for k in wanted:
        if k > n_runs:
            continue
        ats, pows = [], []
        for o in tasks:
            a, p = pass_at_k_value(o, k)
            ats.append(a)
            pows.append(p)
        result["per_k"][k] = {
            "pass@k": sum(ats) / len(ats),
            "pass^k": sum(pows) / len(pows),
        }
    if n_runs in result["per_k"]:
        result[f"pass@{n_runs}"] = result["per_k"][n_runs]["pass@k"]
        result[f"pass^{n_runs}"] = result["per_k"][n_runs]["pass^k"]
    if 1 in result["per_k"]:
        result["pass@1"] = result["per_k"][1]["pass@k"]
        result["pass^1"] = result["per_k"][1]["pass^k"]
    return result


def flipping_tasks(outcomes_by_id: dict[str, list]) -> list[str]:
    """Task ids whose outcomes are neither all-pass nor all-fail."""
    return sorted(
        tid for tid, o in outcomes_by_id.items()
        if o and 0 < sum(1 for x in o if x) < len(o)
    )


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
