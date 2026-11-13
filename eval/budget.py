"""Runner-side budget enforcement (design D.2.4).

The counters live in the runner process, not inside the agent, so no system
under test can reset or rewrite them. Three caps: wall clock (1500 s),
tool calls (40) and total tokens (120,000). The 41st tool call is refused
with a BUDGET_EXCEEDED_TOOL_CALLS ToolResult; a run that exhausts its budget
is still graded (budget_exceeded is a fail, not an exclusion).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from trainpilot.schemas import ToolResult


@dataclass
class BudgetTracker:
    max_wall_seconds: float = 1500.0
    max_tool_calls: int = 40
    max_total_tokens: int = 120000
    tool_calls_used: int = 0
    tokens_used: int = 0
    _started: float = field(default_factory=time.monotonic)

    # -- wall clock ------------------------------------------------------
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def seconds_left(self) -> float:
        return max(0.0, self.max_wall_seconds - self.elapsed())

    # -- charging ---------------------------------------------------------
    def charge_tool_call(self, tool_name: str = "") -> Optional[ToolResult]:
        """Charge one tool call. Returns None while budget remains; on the
        first call past the cap returns the BUDGET_EXCEEDED ToolResult that
        the gateway hands back to the agent instead of executing."""
        self.tool_calls_used += 1
        if self.tool_calls_used > self.max_tool_calls:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                error_code="BUDGET_EXCEEDED_TOOL_CALLS",
                error_message=(
                    f"tool call budget exhausted "
                    f"({self.max_tool_calls} calls allowed)"
                ),
            )
        return None

    def charge_tokens(self, n: int) -> None:
        self.tokens_used += max(0, int(n))

    # -- status -----------------------------------------------------------
    def warning(self) -> Optional[str]:
        """An 80%-of-any-cap warning string, or None. Every system receives
        the same warnings (fairness, design D.2.4)."""
        ratios = [
            self.tool_calls_used / max(1, self.max_tool_calls),
            self.tokens_used / max(1, self.max_total_tokens),
            self.elapsed() / max(1.0, self.max_wall_seconds),
        ]
        if max(ratios) >= 0.8 and self.exceeded() is None:
            return "budget_warning: >=80% of a budget cap consumed"
        return None

    def exceeded(self) -> Optional[str]:
        if self.tool_calls_used > self.max_tool_calls:
            return "tool_calls"
        if self.tokens_used > self.max_total_tokens:
            return "tokens"
        if self.elapsed() > self.max_wall_seconds:
            return "wall_clock"
        return None

    def snapshot(self) -> dict:
        return {
            "max_wall_seconds": self.max_wall_seconds,
            "max_tool_calls": self.max_tool_calls,
            "max_total_tokens": self.max_total_tokens,
            "tool_calls_used": self.tool_calls_used,
            "tokens_used": self.tokens_used,
            "wall_seconds_used": round(self.elapsed(), 3),
            "exceeded": self.exceeded(),
        }
