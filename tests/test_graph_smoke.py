"""End-to-end smoke tests for the section A agent graph.

All runs use the deterministic stub LLM and the scripted StubExecutor:
zero cost, no network, no cluster.
"""
from __future__ import annotations

import json
from pathlib import Path

from trainpilot.graph import StubExecutor, build_agent_graph, run_agent
from trainpilot.llm import LLMClient
from trainpilot.schemas import ApprovalDecision, BudgetState

REQUEST = "Fine-tune DistilBERT on SST-2 with two CPU workers for 1 epoch."
NCCL_LOG = (
    "Traceback (most recent call last):\n"
    "RuntimeError: NCCL error: CUDA unavailable on this CPU-only node\n"
    "ProcessGroupNCCL initialization failed"
)
OOM_LOG = "worker 0 killed: OOMKilled, exit code 137, memory limit exceeded"
DEP_LOG = "ModuleNotFoundError: No module named 'transformers.compat' - incompatible ABI"


def approve_all(action) -> ApprovalDecision:
    return ApprovalDecision(
        approve=True, reason_code="TEST_APPROVE_ALL", decided_by="human")


def success_outcome(accuracy: float = 0.87) -> dict:
    return {
        "status": "succeeded",
        "logs": "training finished, checkpoint written",
        "metrics": {"eval_accuracy": accuracy},
        "checkpoint_dir": None,
        "duration_seconds": 1.0,
    }


def failed_outcome(logs: str) -> dict:
    return {
        "status": "failed",
        "logs": logs,
        "metrics": None,
        "checkpoint_dir": None,
        "duration_seconds": 1.0,
    }


def event_types(result: dict) -> list[str]:
    return [event["type"] for event in result["events"]]


# (a) clean success in one attempt --------------------------------------

def test_clean_success_one_attempt(tmp_path: Path):
    executor = StubExecutor([success_outcome()])
    result = run_agent(
        REQUEST, executor=executor, run_dir=tmp_path / "run-a")
    assert result["success"] is True
    assert result["claimed_success"] is True
    assert result["attempts"] == 1
    assert result["verification"]["passed"] is True
    assert result["verification"]["validation_accuracy"] == 0.87
    types = event_types(result)
    assert "node_started" in types and "node_finished" in types
    assert "approval_required" in types
    assert "verification_done" in types
    assert "attempt_failed" not in types
    run_dir = tmp_path / "run-a"
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "failures.jsonl").exists()
    assert (run_dir / "final_report.json").exists()
    attempt_dir = run_dir / "attempts" / "attempt_00"
    for name in ("spec.json", "train.py", "rayjob.yaml", "train_config.yaml", "logs.txt"):
        assert (attempt_dir / name).exists(), name
    events_on_disk = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(events_on_disk) == len(result["events"])


# (b) nccl failure, backend fix, then success ----------------------------

def test_nccl_failure_repaired_by_backend_fix(tmp_path: Path):
    executor = StubExecutor([failed_outcome(NCCL_LOG), success_outcome(0.88)])
    result = run_agent(
        REQUEST, executor=executor, approval_handler=approve_all,
        run_dir=tmp_path / "run-b")
    assert result["success"] is True
    assert result["attempts"] == 2
    assert result["diagnosis_records"], "expected a diagnosis record"
    record = result["diagnosis_records"][0]
    assert record["fault_family"] == "gpu_only_code"
    assert record["root_cause_code"] == "NCCL_ON_CPU"
    assert record["proposed_fix_code"] == "SET_BACKEND_GLOO"
    assert result["failures"], "expected failure-memory entries"
    types = event_types(result)
    assert "attempt_failed" in types
    assert "diagnosis_ready" in types


# (c) OOM failure, batch fix, then success -------------------------------

def test_oom_failure_repaired_by_batch_fix(tmp_path: Path):
    executor = StubExecutor([failed_outcome(OOM_LOG), success_outcome(0.86)])
    result = run_agent(
        REQUEST, executor=executor, approval_handler=approve_all,
        run_dir=tmp_path / "run-c")
    assert result["success"] is True
    assert result["attempts"] == 2
    assert result["diagnosis_records"][0]["root_cause_code"] == "OOMKILLED_137"
    spec = result["spec"]
    assert spec["per_worker_batch_size"] == 4
    assert spec["gradient_accumulation_steps"] == 4
    attempt_spec = json.loads(
        (tmp_path / "run-c" / "attempts" / "attempt_01" / "spec.json").read_text())
    assert attempt_spec["per_worker_batch_size"] == 4


# (d) always failing gives up after 3 attempts ---------------------------

def test_always_fail_gives_up_after_three_attempts(tmp_path: Path):
    executor = StubExecutor([
        failed_outcome(NCCL_LOG),
        failed_outcome(OOM_LOG),
        failed_outcome(DEP_LOG),
    ])
    result = run_agent(
        REQUEST, executor=executor, approval_handler=approve_all,
        run_dir=tmp_path / "run-d")
    assert result["attempts"] == 3
    assert result["success"] is False
    assert result["claimed_success"] is False
    assert result["outcome"] == "gave_up"
    assert len(executor.calls) == 3
    assert len(result["diagnosis_records"]) == 3


# (e) irreversible action denied by the default handler ------------------

def test_irreversible_action_denied_by_default(tmp_path: Path):
    executor = StubExecutor([success_outcome()])
    request = (
        "Fine-tune DistilBERT on SST-2 for 1 epoch, and please delete "
        "old checkpoints to free up disk space first."
    )
    result = run_agent(request, executor=executor, run_dir=tmp_path / "run-e")
    assert result["success"] is False
    assert result["claimed_success"] is False
    assert result["outcome"] == "rejected"
    assert result["attempts"] == 0
    assert executor.calls == []


# graph assembly ----------------------------------------------------------

def test_build_agent_graph_has_all_twelve_nodes():
    graph = build_agent_graph(LLMClient(), StubExecutor([success_outcome()]))
    node_names = set(graph.get_graph().nodes)
    for name in (
        "intake_profiler", "planner", "coder", "preflight_validator",
        "approval_gate", "executor", "monitor_poller", "log_collector",
        "critic", "repair_router", "verifier", "final_reporter",
    ):
        assert name in node_names, name


# budget enforcement -------------------------------------------------------

def test_budget_exceeded_stops_run():
    executor = StubExecutor([failed_outcome(NCCL_LOG), success_outcome()])
    result = run_agent(
        REQUEST, executor=executor, approval_handler=approve_all,
        budget=BudgetState(max_tool_calls=1))
    assert result["outcome"] == "budget_exceeded"
    assert result["success"] is False
    assert result["attempts"] == 1
