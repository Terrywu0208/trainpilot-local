"""Conditional-edge routing functions (pure, unit-testable).

The repair decision itself lives in the ``repair_router`` node, which
stores its verdict in ``state["route_decision"]``; the functions here
only read state and return the next node name for LangGraph.
"""
from __future__ import annotations

from typing import Any


def route_after_preflight(state: dict[str, Any]) -> str:
    return {
        "pass": "approval_gate",
        "fix": "coder",
        "fail": "critic",
    }.get(state.get("preflight_route", "fail"), "critic")


def route_after_approval(state: dict[str, Any]) -> str:
    if state.get("approval_route") == "approved":
        return "executor"
    return "final_reporter"


def route_after_log_collector(state: dict[str, Any]) -> str:
    if state.get("ray_job_status") == "succeeded":
        return "verifier"
    return "critic"


def route_after_repair_router(state: dict[str, Any]) -> str:
    return {
        "replan": "planner",
        "verify": "verifier",
        "give_up": "final_reporter",
    }.get(state.get("route_decision", "give_up"), "final_reporter")


def route_after_verifier(state: dict[str, Any]) -> str:
    verification = state.get("verification") or {}
    if verification.get("passed"):
        return "final_reporter"
    return "critic"
