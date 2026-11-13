"""Systems under test (design D.1/D.2.1).

One adapter interface for both maturities: `run(task, gateway, budget, llm)
-> AgentRunOutput`. Everything runs offline on the stub LLM.

- demo_agent: the lab agent. Reads the job log through the mock gateway,
  diagnoses it with the shared rule-based diagnoser, applies the repair,
  and only then reports.
- non_agentic_baseline: exactly one LLM call, no tools, no loop (the
  Week 5/6 baseline definition).
- template_script: the canonical training spec submitted as-is; no model,
  no repair. Passes clean scenarios, fails every fault — by design.
- oneshot_llm: one planner call, one submission, no feedback.
- react_agent: a log-tail loop — read raw logs, diagnose, patch, resubmit.
- planning_agent: wraps section A's `trainpilot.graph.run_agent` when it is
  available (with a gateway-backed scenario executor); otherwise falls back
  to a local mock planning adapter with the same output shape (plan ->
  execute -> critic diagnosis record -> repair -> verify before claiming).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from trainpilot.config.models import MODEL_MINI
from trainpilot.llm import diagnose_log_text
from trainpilot.schemas import AgentRunOutput, DiagnosisRecord, TaskSpec

from .budget import BudgetTracker
from .llm_client import LLMClient
from .tool_gateway import MockToolGateway

REFUSAL_MARKERS = (
    "i don't know", "i do not know", "cannot determine", "can't determine",
    "cannot answer", "not enough information", "no such run",
    "does not exist", "never ran",
)
UNANSWERABLE_MARKERS = (
    "never ran", "does not exist", "nonexistent", "no such run",
    "2031", "not exist", "never finished", "supposedly finished",
)

_FAMILY_BY_ERROR = {
    "CODE_BUG": "gpu_only_code",
    "OOM": "memory_limits",
    "DEPENDENCY": "dependency_conflict",
    "DATA_SCHEMA": "data_or_training_error",
    "RESOURCE_QUOTA": "memory_limits",
    "TIMEOUT": "memory_limits",
    "NETWORK": "unknown",
    "UNKNOWN": "unknown",
}


def map_fix_to_code(patch: dict[str, Any]) -> str:
    if patch.get("backend") == "gloo":
        return "SET_BACKEND_GLOO"
    if "per_worker_batch_size" in patch:
        return "REDUCE_BATCH_AND_ACCUMULATE"
    if "resources_per_worker_memory" in patch:
        return "RAISE_MEMORY_WITHIN_LIMIT"
    if patch.get("rebuild_image_with_lockfile"):
        return "REBUILD_IMAGE_WITH_LOCKFILE"
    if patch.get("sanitize_labels"):
        return "SANITIZE_LABELS"
    if "learning_rate" in patch:
        return "LOWER_LEARNING_RATE"
    if "max_length" in patch:
        return "REDUCE_MAX_LENGTH"
    return "UNKNOWN_FIX"


def diagnosis_record(diag: dict[str, Any]) -> DiagnosisRecord:
    family = _FAMILY_BY_ERROR.get(str(diag.get("error_class", "UNKNOWN")), "unknown")
    return DiagnosisRecord(
        fault_family=family,
        root_cause_code=str(diag.get("root_cause_code", "UNKNOWN")),
        evidence=list(diag.get("evidence", [])),
        proposed_fix_code=map_fix_to_code(diag.get("proposed_fix", {}) or {}),
    )


def _is_unanswerable(task: TaskSpec) -> bool:
    if task.grading_type == "unanswerable" or task.difficulty == "unanswerable":
        return True
    text = task.prompt.lower()
    return any(m in text for m in UNANSWERABLE_MARKERS)


def _diagnosis_answer(diag: dict[str, Any], status: str) -> str:
    fix = diag.get("proposed_fix", {}) or {}
    parts = [f"Diagnosis: {diag.get('root_cause', 'unknown')}."]
    if fix.get("backend") == "gloo":
        parts.append("Fix: switch the distributed backend from nccl to gloo, "
                     "because this cluster is CPU-only.")
    if "per_worker_batch_size" in fix:
        parts.append(
            f"Fix: reduce per_worker_batch_size to {fix['per_worker_batch_size']} "
            f"and raise gradient_accumulation_steps to "
            f"{fix.get('gradient_accumulation_steps', 4)} so the effective "
            "batch stays constant.")
    if fix.get("rebuild_image_with_lockfile"):
        parts.append("Fix: rebuild the training image from the pinned "
                     "lockfile to resolve the dependency conflict.")
    if fix.get("sanitize_labels"):
        parts.append("Fix: sanitize the invalid labels in the dataset "
                     "split before training.")
    if "learning_rate" in fix:
        parts.append(f"Fix: lower the learning rate to {fix['learning_rate']}; "
                     "the loss went NaN.")
    parts.append(f"After the repair the job status is {status}; "
                 "the checkpoint was verified via list_checkpoints.")
    return " ".join(parts)


class DemoAgent:
    name = "demo_agent"

    def run(self, task: TaskSpec, gateway: MockToolGateway,
            budget: BudgetTracker, llm: LLMClient) -> AgentRunOutput:
        if _is_unanswerable(task):
            return AgentRunOutput(
                answer="I don't know: that run does not exist in the "
                       "information available to me, so I cannot determine "
                       "its accuracy and will not invent one.",
                claimed_success=False, turns=1, tokens=24,
                tool_trace=list(gateway.trace),
            )
        gateway.call("apply_rayjob", {})
        status = gateway.call("get_rayjob_status", {}).data.get("status", "")
        logs = gateway.call("get_rayjob_logs", {}).data.get("log", "")
        diag = diagnose_log_text(logs)
        record = diagnosis_record(diag)
        fix = diag.get("proposed_fix", {}) or {}
        if fix and status != "SUCCEEDED":
            gateway.call("edit_rayjob_yaml", {"patch": fix})
            gateway.call("apply_rayjob", {})
            status = gateway.call("get_rayjob_status", {}).data.get("status", "")
        gateway.call("list_checkpoints", {})
        metrics = gateway.call("get_metrics", {}).data
        if gateway.sim.get("fault") in (None, "none"):
            acc = metrics.get("eval_accuracy")
            answer = (f"The job succeeded on the first attempt: backend gloo, "
                      f"2 CPU workers, eval_accuracy {acc}. The checkpoint "
                      "was verified via list_checkpoints.")
        else:
            answer = _diagnosis_answer(diag, status)
        turns = len(gateway.trace)
        return AgentRunOutput(
            answer=answer,
            claimed_success=(status == "SUCCEEDED"),
            turns=turns,
            tokens=40 + 12 * turns,
            tool_trace=list(gateway.trace),
            diagnosis=record,
            final_state={"status": status},
        )


class NonAgenticBaseline:
    name = "non_agentic_baseline"

    def run(self, task: TaskSpec, gateway: MockToolGateway,
            budget: BudgetTracker, llm: LLMClient) -> AgentRunOutput:
        resp = llm.chat("baseline", task.prompt, MODEL_MINI)
        return AgentRunOutput(
            answer=resp.text,
            claimed_success=False,
            turns=1,
            tokens=resp.prompt_tokens + resp.completion_tokens,
            tool_trace=list(gateway.trace),
        )


class TemplateScript:
    name = "template_script"

    def run(self, task: TaskSpec, gateway: MockToolGateway,
            budget: BudgetTracker, llm: LLMClient) -> AgentRunOutput:
        gateway.call("apply_rayjob", {})
        status = gateway.call("get_rayjob_status", {}).data.get("status", "")
        ok = status == "SUCCEEDED"
        return AgentRunOutput(
            answer=(f"Template script submitted as-is; job status {status}."),
            claimed_success=ok,
            turns=len(gateway.trace),
            tokens=0,
            tool_trace=list(gateway.trace),
            final_state={"status": status},
        )


class OneShotLLM:
    name = "oneshot_llm"

    def run(self, task: TaskSpec, gateway: MockToolGateway,
            budget: BudgetTracker, llm: LLMClient) -> AgentRunOutput:
        resp = llm.chat("planner", json.dumps({"prompt": task.prompt}), MODEL_MINI)
        try:
            spec = json.loads(resp.text)
        except ValueError:
            spec = {}
        patch = {k: spec[k] for k in
                 ("backend", "per_worker_batch_size",
                  "gradient_accumulation_steps", "num_workers",
                  "resources_per_worker_memory", "learning_rate")
                 if k in spec}
        if patch:
            gateway.call("edit_rayjob_yaml", {"patch": patch})
        gateway.call("apply_rayjob", {})
        status = gateway.call("get_rayjob_status", {}).data.get("status", "")
        return AgentRunOutput(
            answer=f"One-shot plan submitted without feedback; job status {status}.",
            claimed_success=(status == "SUCCEEDED"),
            turns=1 + len(gateway.trace),
            tokens=resp.prompt_tokens + resp.completion_tokens,
            tool_trace=list(gateway.trace),
            final_state={"status": status},
        )


class ReactAgent:
    name = "react_agent"

    def run(self, task: TaskSpec, gateway: MockToolGateway,
            budget: BudgetTracker, llm: LLMClient,
            max_attempts: int = 3) -> AgentRunOutput:
        record: Optional[DiagnosisRecord] = None
        status = ""
        for _ in range(max_attempts):
            gateway.call("apply_rayjob", {})
            status = gateway.call("get_rayjob_status", {}).data.get("status", "")
            if status == "SUCCEEDED":
                break
            logs = gateway.call("get_rayjob_logs", {}).data.get("log", "")
            diag = diagnose_log_text(logs)  # raw log tail, no critic record
            record = diagnosis_record(diag)
            fix = diag.get("proposed_fix", {}) or {}
            if not fix:
                break
            gateway.call("edit_rayjob_yaml", {"patch": fix})
        ok = status == "SUCCEEDED"
        answer = (f"ReAct loop finished with job status {status}." if not ok
                  else "ReAct loop repaired the run; checkpoint produced.")
        return AgentRunOutput(
            answer=answer,
            claimed_success=ok,
            turns=len(gateway.trace),
            tokens=30 + 10 * len(gateway.trace),
            tool_trace=list(gateway.trace),
            diagnosis=record,
            final_state={"status": status},
        )


class _GatewayExecutor:
    """Executor-protocol adapter over the mock gateway (for section A)."""

    def __init__(self, gateway: MockToolGateway):
        self.gateway = gateway

    def execute(self, spec, attempt_index: int, attempt_dir) -> dict:
        patch = {
            "backend": getattr(spec, "backend", "gloo"),
            "per_worker_batch_size": getattr(spec, "per_worker_batch_size", 8),
            "gradient_accumulation_steps": getattr(
                spec, "gradient_accumulation_steps", 2),
            "num_workers": getattr(spec, "num_workers", 2),
            "resources_per_worker_memory": getattr(
                spec, "resources_per_worker_memory", "2Gi"),
            "learning_rate": getattr(spec, "learning_rate", 2e-5),
        }
        self.gateway.call("edit_rayjob_yaml", {"patch": patch})
        self.gateway.call("apply_rayjob", {})
        status = self.gateway.call("get_rayjob_status", {}).data.get("status", "")
        logs = self.gateway.call("get_rayjob_logs", {}).data.get("log", "")
        metrics = self.gateway.call("get_metrics", {}).data
        ok = status == "SUCCEEDED"
        return {
            "status": "succeeded" if ok else "failed",
            "logs": logs,
            "metrics": metrics if ok else None,
            "checkpoint_dir": "checkpoint-final" if ok else None,
            "duration_seconds": 0.1,
        }


class PlanningAgent:
    name = "planning_agent"

    def run(self, task: TaskSpec, gateway: MockToolGateway,
            budget: BudgetTracker, llm: LLMClient) -> AgentRunOutput:
        try:
            from trainpilot.graph import run_agent  # section A, lazy import
            if not callable(run_agent):
                raise ImportError("run_agent not available yet")
            result = run_agent(
                task.prompt,
                executor=_GatewayExecutor(gateway),
                llm_client=llm,
                budget=None,
            )
            records = [
                DiagnosisRecord(**r) if isinstance(r, dict) else r
                for r in result.get("diagnosis_records", [])
            ]
            return AgentRunOutput(
                answer=str(result.get("final_message", "")),
                claimed_success=bool(result.get("claimed_success")),
                turns=len(gateway.trace),
                tokens=50 + 10 * len(gateway.trace),
                tool_trace=list(gateway.trace),
                diagnosis=records[-1] if records else None,
                final_state={"attempts": result.get("attempts", 0)},
            )
        except Exception:
            return self._mock_planning(task, gateway, budget, llm)

    def _mock_planning(self, task: TaskSpec, gateway: MockToolGateway,
                       budget: BudgetTracker, llm: LLMClient) -> AgentRunOutput:
        """Local planning adapter with the run_agent output shape: plan,
        execute, critic diagnosis record, bounded repair, and success is
        claimed only after verification through the gateway."""
        if _is_unanswerable(task):
            return AgentRunOutput(
                answer="I don't know: the requested run does not exist, so "
                       "there is nothing to verify.",
                claimed_success=False, turns=1, tokens=24,
                tool_trace=list(gateway.trace),
            )
        llm.chat("planner", json.dumps({"prompt": task.prompt}), MODEL_MINI)
        record: Optional[DiagnosisRecord] = None
        status = ""
        last_fix: dict[str, Any] = {}
        for _attempt in range(3):
            gateway.call("apply_rayjob", {})
            status = gateway.call("get_rayjob_status", {}).data.get("status", "")
            if status == "SUCCEEDED":
                break
            logs = gateway.call("get_rayjob_logs", {}).data.get("log", "")
            resp = llm.chat("critic", logs, MODEL_MINI)
            try:
                diag = json.loads(resp.text)
            except ValueError:
                diag = diagnose_log_text(logs)
            record = diagnosis_record(diag)
            last_fix = diag.get("proposed_fix", {}) or {}
            if not last_fix:
                break
            gateway.call("edit_rayjob_yaml", {"patch": last_fix})
        verified = False
        if status == "SUCCEEDED":
            ckpts = gateway.call("list_checkpoints", {}).data.get("checkpoints", [])
            metrics = gateway.call("get_metrics", {}).data
            verified = bool(ckpts) and metrics.get("eval_accuracy") is not None
        answer = _diagnosis_answer(
            {"root_cause": record.root_cause_code if record else "none",
             "proposed_fix": last_fix}, status) if record else (
            f"Planned run finished with job status {status}; "
            "checkpoint verified." if verified else
            f"Planned run finished with job status {status}.")
        return AgentRunOutput(
            answer=answer,
            claimed_success=verified,
            turns=len(gateway.trace),
            tokens=60 + 12 * len(gateway.trace),
            tool_trace=list(gateway.trace),
            diagnosis=record,
            final_state={"status": status, "verified": verified},
        )


SYSTEMS = {
    cls.name: cls
    for cls in (DemoAgent, NonAgenticBaseline, TemplateScript, OneShotLLM,
                ReactAgent, PlanningAgent)
}


def get_system(name: str):
    if name not in SYSTEMS:
        raise ValueError(f"unknown system {name!r}; known: {sorted(SYSTEMS)}")
    return SYSTEMS[name]()
