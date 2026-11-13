"""Graph assembly and the public run entry point (docs/CONTRACTS.md).

``build_agent_graph`` compiles the 12-node LangGraph state machine with
an in-memory checkpointer. ``run_agent`` runs one request end to end,
collects the event stream, and (when ``run_dir`` is given) leaves the
run artifacts on disk: ``events.jsonl``, ``failures.jsonl``,
``attempts/attempt_<nn>/`` directories and the final report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ..llm import LLMClient
from ..schemas import ApprovalDecision, BudgetState, ProposedAction, VerificationResult
from . import nodes
from .context import AgentContext
from .routing import (
    route_after_approval,
    route_after_log_collector,
    route_after_preflight,
    route_after_repair_router,
    route_after_verifier,
)
from .state import MAX_ATTEMPTS, TrainPilotState

NODE_NAMES = [
    "intake_profiler", "planner", "coder", "preflight_validator",
    "approval_gate", "executor", "monitor_poller", "log_collector",
    "critic", "repair_router", "verifier", "final_reporter",
]


def _wrap(name: str, fn: Callable, ctx: AgentContext) -> Callable:
    def wrapped(state: dict) -> dict:
        ctx.emit("node_started", {
            "node": name, "attempt": int(state.get("attempt", 0))})
        updates = fn(state) or {}
        ctx.emit("node_finished", {
            "node": name,
            "status": updates.get("status", state.get("status", "")),
        })
        return updates

    return wrapped


def _build(ctx: AgentContext):
    graph = StateGraph(TrainPilotState)
    factories = {
        "intake_profiler": nodes.make_intake_profiler(ctx),
        "planner": nodes.make_planner(ctx),
        "coder": nodes.make_coder(ctx),
        "preflight_validator": nodes.make_preflight_validator(ctx),
        "approval_gate": nodes.make_approval_gate(ctx),
        "executor": nodes.make_executor_node(ctx),
        "monitor_poller": nodes.make_monitor_poller(ctx),
        "log_collector": nodes.make_log_collector(ctx),
        "critic": nodes.make_critic(ctx),
        "repair_router": nodes.make_repair_router(ctx),
        "verifier": nodes.make_verifier(ctx),
        "final_reporter": nodes.make_final_reporter(ctx),
    }
    for name in NODE_NAMES:
        graph.add_node(name, _wrap(name, factories[name], ctx))
    graph.add_edge(START, "intake_profiler")
    graph.add_edge("intake_profiler", "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "preflight_validator")
    graph.add_conditional_edges("preflight_validator", route_after_preflight, {
        "approval_gate": "approval_gate", "coder": "coder", "critic": "critic"})
    graph.add_conditional_edges("approval_gate", route_after_approval, {
        "executor": "executor", "final_reporter": "final_reporter"})
    graph.add_edge("executor", "monitor_poller")
    graph.add_edge("monitor_poller", "log_collector")
    graph.add_conditional_edges("log_collector", route_after_log_collector, {
        "verifier": "verifier", "critic": "critic"})
    graph.add_edge("critic", "repair_router")
    graph.add_conditional_edges("repair_router", route_after_repair_router, {
        "planner": "planner", "verifier": "verifier",
        "final_reporter": "final_reporter"})
    graph.add_conditional_edges("verifier", route_after_verifier, {
        "final_reporter": "final_reporter", "critic": "critic"})
    graph.add_edge("final_reporter", END)
    return graph.compile(checkpointer=InMemorySaver())


def build_agent_graph(llm_client, executor, approval_handler=None, run_dir=None):
    """Compile the 12-node agent graph (see docs/CONTRACTS.md)."""
    ctx = AgentContext(
        llm=llm_client,
        executor=executor,
        approval_handler=approval_handler,
        run_dir=Path(run_dir) if run_dir is not None else None,
    )
    return _build(ctx)


def _initial_state(ctx: AgentContext, request_text: str,
                    budget: Optional[BudgetState],
                    session_context: str) -> dict:
    return {
        "run_id": ctx.run_id,
        "user_request_raw": request_text,
        "session_context": session_context,
        "status": "intake",
        "request": None,
        "machine_profile": None,
        "spec": None,
        "spec_version": 0,
        "spec_hash": "",
        "plan_steps": [],
        "generated_files": {},
        "attempt": 0,
        "attempts_started": 0,
        "max_attempts": MAX_ATTEMPTS,
        "repair_cycles": 0,
        "route_decision": "",
        "give_up_reason": "",
        "outcome": "",
        "failure_memory": [],
        "failures_written": [],
        "last_diagnosis": None,
        "diagnosis_records": [],
        "pending_irreversible": None,
        "approval_first_launch": "pending",
        "approval_irreversible": {},
        "approval_notes": "",
        "launch_approved_attempt": -1,
        "approval_route": "",
        "budget": (budget or BudgetState()).model_dump(),
        "executor_result": None,
        "ray_job_name": None,
        "ray_job_status": None,
        "logs_ref": None,
        "logs_summary_sanitized": None,
        "logs_unavailable": False,
        "preflight_passed": False,
        "preflight_errors": [],
        "preflight_route": "",
        "preflight_fix_count": 0,
        "verification": None,
        "claimed_success": False,
        "success": False,
        "final_message": "",
        "error_fatal": None,
    }


def run_agent(request_text: str, *, executor, llm_client=None,
              approval_handler=None, run_dir=None, budget=None,
              session_context: str = "") -> dict:
    """Run one training request through the agent graph.

    Returns the contract dict: success, claimed_success, attempts,
    final_message, verification (dict), diagnosis_records, events and
    failures (the failure-memory entries first written during this run,
    with their final outcomes).
    """
    llm_client = llm_client or LLMClient()
    run_dir_path = Path(run_dir) if run_dir is not None else None
    ctx = AgentContext(
        llm=llm_client,
        executor=executor,
        approval_handler=approval_handler,
        run_dir=run_dir_path,
    )
    graph = _build(ctx)
    initial = _initial_state(ctx, request_text, budget, session_context)
    config = {
        "configurable": {"thread_id": ctx.run_id},
        # A 3-attempt repair loop legitimately takes ~30 node steps,
        # above LangGraph's default recursion limit of 25.
        "recursion_limit": 200,
    }
    try:
        final = graph.invoke(initial, config=config)
    except Exception as exc:  # orchestration bug: report, never hang
        final = dict(initial)
        final.update({
            "success": False,
            "claimed_success": False,
            "outcome": "gave_up",
            "give_up_reason": "orchestration_error",
            "error_fatal": f"{type(exc).__name__}: {exc}",
            "final_message": (
                "Run stopped by an orchestration error: "
                f"{type(exc).__name__}: {exc}"),
        })

    verification = final.get("verification")
    if not verification:
        verification = VerificationResult().model_dump()
        verification["passed"] = False
    failures_by_id = {
        entry.get("failure_id"): entry
        for entry in final.get("failure_memory", [])
    }
    failures = [
        failures_by_id.get(entry.get("failure_id"), entry)
        for entry in final.get("failures_written", [])
    ]
    result = {
        "run_id": ctx.run_id,
        "success": bool(final.get("success", False)),
        "claimed_success": bool(final.get("claimed_success", False)),
        "attempts": int(final.get("attempts_started", 0)),
        "outcome": final.get("outcome", ""),
        "final_message": final.get("final_message", ""),
        "verification": verification,
        "diagnosis_records": final.get("diagnosis_records", []),
        "events": ctx.events,
        "failures": failures,
        "spec": final.get("spec"),
        "give_up_reason": final.get("give_up_reason", ""),
        "error_fatal": final.get("error_fatal"),
    }
    if run_dir_path is not None:
        run_dir_path.mkdir(parents=True, exist_ok=True)
        events_path = run_dir_path / "events.jsonl"
        with events_path.open("w", encoding="utf-8") as handle:
            for event in ctx.events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        failures_path = run_dir_path / "failures.jsonl"
        if not failures_path.exists():
            failures_path.write_text("", encoding="utf-8")
    return result
