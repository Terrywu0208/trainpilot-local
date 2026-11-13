"""Model routing constants (A section).

The default provider is `stub`: a deterministic, offline, zero-cost stand-in
used for tests, CI and the recorded demo. No code path in this repository
spends money unless a human explicitly sets TRAINPILOT_LLM_PROVIDER=openai
and supplies their own API key.
"""
from __future__ import annotations

import os

MODEL_MINI = "gpt-5-mini-2025-08-07"  # pinned snapshot for the planning agent
MODEL_NANO = "gpt-5-nano-2025-08-07"  # pinned snapshot for the low-effort variant

# Node -> (model, effort) routing, per architecture section A.
NODE_MODEL_ROUTING = {
    "intake_profiler": (MODEL_NANO, "low"),
    "planner": (MODEL_MINI, "medium"),
    "coder": (MODEL_MINI, "medium"),
    "critic": (MODEL_MINI, "high"),
    "final_reporter": (MODEL_NANO, "low"),
}


def llm_provider() -> str:
    """Return the active provider name. Defaults to the free stub."""
    return os.environ.get("TRAINPILOT_LLM_PROVIDER", "stub").lower()
