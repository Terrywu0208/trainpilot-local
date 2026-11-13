"""In-graph budget accounting (design section A.2.4).

The in-graph BudgetState is the demo/standalone copy; in experiment mode
the runner-side BudgetTracker (section D) is authoritative. Counters are
only ever mutated through these helpers so the numbers stay auditable.
"""
from __future__ import annotations

import time

from ..schemas import BudgetState


def budget_from_state(raw: dict | None) -> BudgetState:
    if not raw:
        return BudgetState()
    return BudgetState(**raw)


def charge_tool_call(budget: BudgetState) -> BudgetState:
    budget.tool_calls_used += 1
    return budget


def charge_tokens(budget: BudgetState, prompt_tokens: int,
                  completion_tokens: int) -> BudgetState:
    budget.tokens_used += int(prompt_tokens) + int(completion_tokens)
    return budget


def refresh_wall_clock(budget: BudgetState, started_monotonic: float) -> BudgetState:
    budget.wall_seconds_used = time.monotonic() - started_monotonic
    return budget


def dump(budget: BudgetState) -> dict:
    return budget.model_dump()
