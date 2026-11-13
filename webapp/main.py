"""TrainPilot-Local webapp (Section E).

FastAPI front end for the TrainPilot-Local agent: a single-page UI with
four views (Run Console, Failure Memory, Benchmark Dashboard, Replay),
Server-Sent Events for live run progress, two human approval gates wired
to the agent graph's ``approval_handler``, per-session follow-up context,
and run artifacts persisted under ``runs/<run_id>/`` (design E.1.4).

Run it with::

    uvicorn webapp.main:app --port 8000

Executor resolution for real runs (first one that imports wins):

1. ``training.local_runner.LocalExecutor`` (Section C, smoke backend) fed
   into ``trainpilot.graph.run_agent`` (Section A).
2. ``trainpilot.graph.run_agent`` with ``StubExecutor`` scripted demo
   outcomes (fail, fail, succeed).
3. A built-in canned event script (last resort, clearly labelled), so the
   UI, gates, sessions and replay still work before A/C land.

Tests inject a fake ``run_agent`` through :func:`create_app`, so none of
the above is required for the test suite.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

GATE_FIRST_LAUNCH = "gate1_first_launch"
GATE_IRREVERSIBLE = "gate2_irreversible"

# Node names come from Section A; the Plan View renders exactly these.
PLAN_NODES = [
    "intake_profiler", "planner", "coder", "preflight_validator",
    "approval_gate", "executor", "monitor_poller", "log_collector",
    "critic", "repair_router", "verifier", "final_reporter",
]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _verification_passed(verification: Any) -> bool:
    """Compute the VerificationResult.passed semantics on plain data."""
    if not isinstance(verification, dict):
        return False
    if "passed" in verification:
        return bool(verification["passed"])
    return bool(
        verification.get("checkpoint_exists")
        and verification.get("checkpoint_loadable")
        and verification.get("config_honors_request")
        and verification.get("validation_accuracy") is not None
    )


class _Decision(dict):
    """Approval decision readable both as a dict (this module's canned
    fallback) and by attribute (the agent graph's ApprovalDecision
    contract). Integration shim added 2026-11-12."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _gate_for_action(action: dict) -> str:
    if action.get("action_type") == "launch":
        return GATE_FIRST_LAUNCH
    return GATE_IRREVERSIBLE


# ---------------------------------------------------------------------------
# Run record + manager
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_id: str
    session_id: str
    request_text: str
    fields: dict[str, Any]
    run_dir: Path
    status: str = "queued"  # queued|running|awaiting_approval|finished|error
    created_at: str = field(default_factory=_utcnow_iso)
    events: list[dict[str, Any]] = field(default_factory=list)
    result: Optional[dict[str, Any]] = None
    error: str = ""
    subscribers: set = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "request_text": self.request_text,
            "fields": self.fields,
            "status": self.status,
            "created_at": self.created_at,
            "error": self.error,
        }
        if self.result is not None:
            data["result"] = self.result
            data["verification"] = self.result.get("verification")
            data["success"] = self.result.get("success")
            data["attempts"] = self.result.get("attempts")
        return data


class RunManager:
    """Serializes runs (strictly one active run), owns the approval
    futures, the per-session follow-up summaries and all persistence."""

    def __init__(self, run_agent_fn: Optional[Callable] = None,
                 runs_root: Optional[Path] = None) -> None:
        self.run_agent_fn = run_agent_fn
        self.runs_root = Path(runs_root) if runs_root else REPO_ROOT / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, RunRecord] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        # run_id -> gate -> (asyncio.Future, action dict)
        self._pending: dict[str, dict[str, tuple]] = {}

    # -- submission ------------------------------------------------------
    async def submit(self, request_text: str, fields: dict[str, Any],
                     session_id: str) -> RunRecord:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"run-{stamp}-{uuid.uuid4().hex[:6]}"
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        record = RunRecord(run_id=run_id, session_id=session_id,
                           request_text=request_text, fields=fields,
                           run_dir=run_dir)
        self.records[run_id] = record
        self._write_json(run_dir / "request.json", {
            "run_id": run_id,
            "session_id": session_id,
            "request_text": request_text,
            "fields": fields,
            "created_at": record.created_at,
        })
        (run_dir / "events.jsonl").touch()
        (run_dir / "audit_log.jsonl").touch()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._worker_loop())
        await self._queue.put(record)
        return record

    async def _worker_loop(self) -> None:
        while True:
            record = await self._queue.get()
            try:
                await self._run_one(record)
            except Exception as exc:  # worker must never die
                record.status = "error"
                record.error = f"worker error: {exc}"
            finally:
                self._queue.task_done()

    # -- execution ---------------------------------------------------------
    async def _run_one(self, record: RunRecord) -> None:
        record.status = "running"
        session_context = ""
        previous = self.sessions.get(record.session_id)
        if previous:
            session_context = previous.get("summary", "")
        self._emit(record, "run_started", {
            "request_text": record.request_text,
            "session_id": record.session_id,
            "session_context_used": bool(session_context),
        })
        handler = self._make_approval_handler(record)
        try:
            result = await asyncio.to_thread(
                self._call_run_agent, record, handler, session_context)
        except Exception as exc:
            record.status = "error"
            record.error = str(exc)
            self._emit(record, "run_error", {"error": str(exc)})
            self._finalize(record, None)
            return
        # Relay the agent's own events (they carry the Plan View and the
        # Attempt Cards) into the live stream and the persisted log.
        for event in result.get("events", []):
            if isinstance(event, dict) and event.get("type"):
                self._emit(record, event["type"],
                           event.get("payload", {}),
                           timestamp=event.get("timestamp"))
        record.result = result
        record.status = "finished"
        self._finalize(record, result)
        self._emit(record, "run_finished", {
            "success": bool(result.get("success")),
            "attempts": result.get("attempts"),
            "verification_passed": _verification_passed(
                result.get("verification")),
        })

    def _call_run_agent(self, record: RunRecord,
                        approval_handler: Callable,
                        session_context: str) -> dict[str, Any]:
        """Runs in a worker thread. Uses the injected function when given,
        otherwise resolves the real agent (LocalExecutor -> StubExecutor
        -> canned script)."""
        if self.run_agent_fn is not None:
            return self.run_agent_fn(
                record.request_text,
                run_dir=record.run_dir,
                approval_handler=approval_handler,
                session_context=session_context,
            )
        return _default_run_agent(record.request_text, record.run_dir,
                                  approval_handler, session_context)

    def _finalize(self, record: RunRecord,
                  result: Optional[dict[str, Any]]) -> None:
        run_dir = record.run_dir
        summary: dict[str, Any] = {
            "run_id": record.run_id,
            "session_id": record.session_id,
            "request_text": record.request_text,
            "fields": record.fields,
            "status": record.status,
            "created_at": record.created_at,
            "error": record.error,
        }
        if result is not None:
            verification = result.get("verification")
            if isinstance(verification, dict):
                verification = dict(verification)
                verification["passed"] = _verification_passed(verification)
                result = dict(result)
                result["verification"] = verification
                record.result = result
            summary["result"] = {
                k: v for k, v in result.items() if k != "events"
            }
            summary["success"] = result.get("success")
            summary["attempts"] = result.get("attempts")
            summary["verification"] = result.get("verification")
            failures = result.get("failures") or []
            if failures:
                with open(run_dir / "failures.jsonl", "w",
                          encoding="utf-8") as fh:
                    for entry in failures:
                        fh.write(json.dumps(entry) + "\n")
            for idx, diag in enumerate(result.get("diagnosis_records")
                                        or [], start=1):
                attempt_dir = run_dir / f"attempt_{idx}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                self._write_json(attempt_dir / "diagnosis.json", diag)
            # Session follow-up memory: one-paragraph summary of this run.
            accuracy = None
            if isinstance(result.get("verification"), dict):
                accuracy = result["verification"].get("validation_accuracy")
            summary_text = (
                f"Previous run {record.run_id}: request "
                f"{record.request_text!r}; outcome "
                f"{'success' if result.get('success') else 'failure'} after "
                f"{result.get('attempts')} attempt(s); validation accuracy "
                f"{accuracy}; final message: {result.get('final_message', '')}"
            )
            self.sessions[record.session_id] = {
                "last_run_id": record.run_id,
                "summary": summary_text,
            }
        self._write_json(run_dir / "summary.json", summary)

    # -- events ------------------------------------------------------------
    def _emit(self, record: RunRecord, event_type: str,
              payload: dict[str, Any],
              timestamp: Optional[str] = None) -> dict[str, Any]:
        event = {
            "type": event_type,
            "timestamp": timestamp or _utcnow_iso(),
            "run_id": record.run_id,
            "payload": payload,
        }
        record.events.append(event)
        with open(record.run_dir / "events.jsonl", "a",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        for queue in list(record.subscribers):
            queue.put_nowait(event)
        return event

    # -- approvals -----------------------------------------------------------
    def _make_approval_handler(self, record: RunRecord) -> Callable:
        """Sync callable handed to the agent graph. It blocks the agent's
        worker thread until a human decides through POST /api/runs/{id}/
        approve; every request and decision is audited."""

        def handler(action: Any) -> dict[str, Any]:
            if hasattr(action, "model_dump"):
                action_dict = action.model_dump()
            elif isinstance(action, dict):
                action_dict = dict(action)
            else:
                action_dict = {"action_type": "other",
                               "description": str(action)}
            # We are on a worker thread: schedule onto the manager's loop.
            loop = self._loop()
            future = asyncio.run_coroutine_threadsafe(
                self._await_decision(record, action_dict), loop)
            decision = future.result(timeout=3600)
            # Integration fix (2026-11-12): the agent graph reads the
            # decision by attribute (decision.approve / .reason_code, the
            # ApprovalDecision contract), while this module's canned
            # fallback reads it by key. A dict subclass with attribute
            # access satisfies both consumers.
            return _Decision(decision)

        return handler

    def _loop(self) -> asyncio.AbstractEventLoop:
        if self._worker is not None:
            loop = self._worker.get_loop()
            if loop is not None:
                return loop
        raise RuntimeError("RunManager worker is not running")

    async def _await_decision(self, record: RunRecord,
                              action: dict[str, Any]) -> dict[str, Any]:
        gate = _gate_for_action(action)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending.setdefault(record.run_id, {})[gate] = (future, action)
        previous_status = record.status
        record.status = "awaiting_approval"
        self._emit(record, "approval_required", {
            "gate": gate,
            "action": action,
            "irreversible": bool(action.get("irreversible")),
        })
        decision = await future
        self._pending.get(record.run_id, {}).pop(gate, None)
        if record.status == "awaiting_approval":
            record.status = previous_status if previous_status != \
                "awaiting_approval" else "running"
        self._append_audit(record, gate, action,
                           "approved" if decision["approve"] else "rejected",
                           decision.get("actor", "human"))
        self._emit(record, "approval_decided", {
            "gate": gate,
            "approve": bool(decision["approve"]),
            "reason_code": decision.get("reason_code", ""),
        })
        return {
            "approve": bool(decision["approve"]),
            "reason_code": decision.get(
                "reason_code",
                "HUMAN_APPROVED" if decision["approve"] else "HUMAN_REJECTED"),
            "decided_by": "human",
        }

    def resolve_approval(self, run_id: str, gate: Optional[str],
                         approve: bool, actor: str = "human") -> dict:
        pending = self._pending.get(run_id) or {}
        if not pending:
            raise HTTPException(status_code=409,
                                detail="no approval is pending for this run")
        if gate is None:
            gate = next(iter(pending))
        if gate not in pending:
            raise HTTPException(
                status_code=409,
                detail=f"gate {gate!r} is not pending; pending gates: "
                       f"{sorted(pending)}")
        future, _action = pending[gate]
        if not future.done():
            future.set_result({"approve": approve, "actor": actor,
                               "reason_code": "HUMAN_APPROVED" if approve
                               else "HUMAN_REJECTED"})
        return {"ok": True, "run_id": run_id, "gate": gate,
                "decision": "approved" if approve else "rejected"}

    def pending_gates(self, run_id: str) -> list[str]:
        return sorted((self._pending.get(run_id) or {}).keys())

    # -- persistence helpers -------------------------------------------------
    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, indent=2, default=str),
                        encoding="utf-8")

    def _append_audit(self, record: RunRecord, gate: str,
                      action: dict[str, Any], decision: str,
                      actor: str) -> None:
        payload_hash = hashlib.sha256(
            json.dumps(action, sort_keys=True, default=str)
            .encode()).hexdigest()[:12]
        entry = {
            "run_id": record.run_id,
            "gate_type": gate,
            "actor": actor,
            "decision": decision,
            "timestamp": _utcnow_iso(),
            "payload_hash": payload_hash,
            "action_type": action.get("action_type"),
            "description": action.get("description", ""),
        }
        with open(record.run_dir / "audit_log.jsonl", "a",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    # -- reads ---------------------------------------------------------------
    def list_runs(self) -> list[dict[str, Any]]:
        runs: dict[str, dict[str, Any]] = {}
        if self.runs_root.exists():
            for run_dir in sorted(self.runs_root.iterdir()):
                summary_file = run_dir / "summary.json"
                if run_dir.is_dir() and summary_file.exists():
                    try:
                        data = json.loads(
                            summary_file.read_text(encoding="utf-8"))
                        runs[run_dir.name] = {
                            "run_id": run_dir.name,
                            "session_id": data.get("session_id", ""),
                            "request_text": data.get("request_text", ""),
                            "status": data.get("status", "finished"),
                            "success": data.get("success"),
                            "attempts": data.get("attempts"),
                        }
                    except (ValueError, OSError):
                        continue
        for record in self.records.values():
            runs[record.run_id] = {
                "run_id": record.run_id,
                "session_id": record.session_id,
                "request_text": record.request_text,
                "status": record.status,
                "success": (record.result or {}).get("success"),
                "attempts": (record.result or {}).get("attempts"),
            }
        return sorted(runs.values(), key=lambda r: r["run_id"])

    def read_events(self, run_id: str) -> Optional[list[dict[str, Any]]]:
        record = self.records.get(run_id)
        if record is not None:
            return list(record.events)
        events_file = self.runs_root / run_id / "events.jsonl"
        if not events_file.exists():
            return None
        events = []
        for line in events_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
        return events

    def all_failures(self) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        seen: set = set()
        if self.runs_root.exists():
            for failures_file in sorted(self.runs_root.glob(
                    "*/failures.jsonl")):
                for line in failures_file.read_text(
                        encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    key = json.dumps(entry, sort_keys=True, default=str)
                    if key not in seen:
                        seen.add(key)
                        failures.append(entry)
        return failures


# ---------------------------------------------------------------------------
# Default agent resolution: LocalExecutor -> StubExecutor -> canned script
# ---------------------------------------------------------------------------

_DEMO_OUTCOMES = [
    {
        "status": "failed",
        "logs": "torch.distributed: NCCL backend is not available on CPU "
                "workers; collective aborted.",
        "metrics": None, "checkpoint_dir": None, "duration_seconds": 42.0,
    },
    {
        "status": "failed",
        "logs": "worker-0: OOMKilled (exit 137); per-worker batch 32 "
                "exceeds the 2Gi memory limit.",
        "metrics": None, "checkpoint_dir": None, "duration_seconds": 55.0,
    },
    {
        "status": "succeeded",
        "logs": "training finished; checkpoint written.",
        "metrics": {"eval_accuracy": 0.83},
        "checkpoint_dir": "checkpoint-final", "duration_seconds": 61.0,
    },
]


def _default_run_agent(request_text: str, run_dir: Path,
                       approval_handler: Callable,
                       session_context: str) -> dict[str, Any]:
    """Resolve the real agent lazily. Every import is guarded so the
    webapp works (via the canned fallback) before Sections A/C exist."""
    run_agent = None
    try:
        from trainpilot.graph import run_agent as graph_run_agent
        run_agent = graph_run_agent
    except Exception:
        run_agent = None

    if run_agent is not None:
        # Preferred: Section C's LocalExecutor (smoke backend, offline).
        try:
            from training.local_runner import LocalExecutor
            return run_agent(request_text, executor=LocalExecutor(),
                             approval_handler=approval_handler,
                             run_dir=run_dir,
                             session_context=session_context)
        except Exception:
            pass
        # Fallback: Section A's StubExecutor scripted demo outcomes.
        try:
            from trainpilot.graph import StubExecutor
            return run_agent(
                request_text,
                executor=StubExecutor(outcomes=list(_DEMO_OUTCOMES)),
                approval_handler=approval_handler,
                run_dir=run_dir,
                session_context=session_context)
        except Exception:
            pass
    return _canned_run(request_text, run_dir, approval_handler,
                       session_context)


def _canned_run(request_text: str, run_dir: Path,
                approval_handler: Callable,
                session_context: str) -> dict[str, Any]:
    """Last-resort canned event script: a labelled three-attempt demo
    (GPU backend failure -> OOM failure -> verified success) that still
    exercises both approval gates, so the UI, session memory and replay
    remain demonstrable with no other section present."""
    events: list[dict[str, Any]] = []

    def emit(event_type: str, payload: dict[str, Any]) -> None:
        events.append({"type": event_type, "timestamp": _utcnow_iso(),
                       "payload": payload})
        time.sleep(0.02)

    for node in ("intake_profiler", "planner", "coder",
                 "preflight_validator"):
        emit("node_started", {"node": node})
        emit("node_finished", {"node": node, "status": "ok"})
    emit("plan_ready", {
        "summary": "Canned demo plan for: " + request_text,
        "spec": {"model_name": "distilbert-base-uncased", "dataset": "sst2",
                 "num_workers": 2, "backend": "nccl",
                 "per_worker_batch_size": 32},
        "canned": True,
    })
    decision = approval_handler({
        "action_type": "launch",
        "description": "First launch of the RayJob (canned demo)",
        "tool_name": "apply_rayjob", "irreversible": False,
        "source": "agent",
    })
    if not (decision or {}).get("approve", False):
        return {
            "success": False, "claimed_success": False, "attempts": 0,
            "final_message": "Launch rejected at Gate 1; run stopped.",
            "verification": {"checkpoint_exists": False,
                             "checkpoint_loadable": False,
                             "validation_accuracy": None,
                             "config_honors_request": False,
                             "details": {"reason": "launch_rejected"}},
            "diagnosis_records": [], "events": events, "failures": [],
        }
    script = [
        (1, "failed", "NCCL_ON_CPU",
         "torch.distributed: NCCL backend unavailable on CPU workers",
         {"backend": "gloo"}),
        (2, "failed", "OOMKILLED_137",
         "worker OOMKilled: batch 32 exceeds 2Gi limit",
         {"per_worker_batch_size": 4, "gradient_accumulation_steps": 4}),
        (3, "succeeded", None, "training finished", {}),
    ]
    diagnosis_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, status, label, log_line, fix in script:
        emit("node_started", {"node": "executor"})
        emit("attempt_started", {"attempt": index})
        emit("attempt_finished", {
            "attempt": index, "status": status,
            "logs": log_line, "fix": fix,
            "duration_seconds": 30.0 + index,
        })
        if label:
            record = {
                "fault_family": ("gpu_only_code" if index == 1
                                 else "memory_limits"),
                "root_cause_code": label,
                "evidence": [log_line],
                "proposed_fix_code": ("SWITCH_BACKEND_GLOO" if index == 1
                                      else "REDUCE_BATCH_ACCUMULATE"),
            }
            diagnosis_records.append(record)
            failures.append({
                "fault_label": label, "root_cause_code": label,
                "evidence": [log_line],
                "fix_code": record["proposed_fix_code"],
                "run_id": run_dir.name,
            })
            emit("diagnosis_ready", {"attempt": index,
                                     "diagnosis": record})
    decision = approval_handler({
        "action_type": "delete",
        "description": "Delete two old checkpoints (1.2 GB) to free disk "
                       "before the final verification copy (canned demo)",
        "tool_name": "purge_attempt_artifacts", "irreversible": True,
        "source": "agent",
        "args": {"paths": ["runs/old/checkpoint-a", "runs/old/checkpoint-b"],
                 "total_mb": 1228},
    })
    deleted = bool((decision or {}).get("approve", False))
    emit("node_started", {"node": "verifier"})
    emit("verification_done", {"passed": True})
    emit("node_finished", {"node": "final_reporter", "status": "ok"})
    return {
        "success": True, "claimed_success": True, "attempts": 3,
        "final_message": (
            "Canned demo run finished: checkpoint verified "
            "(eval_accuracy 0.83). Gate 2 deletion "
            + ("approved and executed." if deleted
               else "rejected; nothing was deleted.")),
        "verification": {"checkpoint_exists": True,
                         "checkpoint_loadable": True,
                         "validation_accuracy": 0.83,
                         "config_honors_request": True,
                         "details": {"canned": True}},
        "diagnosis_records": diagnosis_records, "events": events,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# App factory + endpoints
# ---------------------------------------------------------------------------

def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def create_app(run_agent_fn: Optional[Callable] = None,
               runs_root: Optional[Path] = None,
               reports_root: Optional[Path] = None) -> FastAPI:
    manager = RunManager(run_agent_fn=run_agent_fn, runs_root=runs_root)
    reports_dir = (Path(reports_root) if reports_root
                   else REPO_ROOT / "reports")
    app = FastAPI(title="TrainPilot-Local", version="1.0.0")
    app.state.manager = manager
    app.state.reports_dir = reports_dir

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html",
                            media_type="text/html")

    @app.post("/api/runs")
    async def create_run(request: Request) -> JSONResponse:
        body = await request.json()
        request_text = (body.get("request") or body.get("request_text")
                        or "").strip()
        if not request_text:
            raise HTTPException(status_code=422,
                                detail="request text is required")
        fields = body.get("fields") or {
            key: body[key]
            for key in ("model_name", "dataset", "num_workers", "epochs",
                        "per_worker_batch_size")
            if key in body
        }
        session_id = body.get("session_id") or "default"
        record = await manager.submit(request_text, fields, session_id)
        return JSONResponse(status_code=201, content={
            "run_id": record.run_id, "status": record.status,
            "session_id": record.session_id,
        })

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        return {"runs": manager.list_runs()}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        record = manager.records.get(run_id)
        if record is not None:
            data = record.to_dict()
            data["pending_gates"] = manager.pending_gates(run_id)
            return data
        summary_file = manager.runs_root / run_id / "summary.json"
        if summary_file.exists():
            return json.loads(summary_file.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="run not found")

    @app.post("/api/runs/{run_id}/approve")
    async def approve(run_id: str, request: Request) -> dict[str, Any]:
        if run_id not in manager.records:
            raise HTTPException(status_code=404, detail="run not found")
        body = await request.json()
        if "approve" not in body:
            raise HTTPException(status_code=422,
                                detail="body must include 'approve' (bool)")
        return manager.resolve_approval(
            run_id, body.get("gate"), bool(body["approve"]),
            actor=body.get("actor", "human"))

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str) -> StreamingResponse:
        record = manager.records.get(run_id)
        if record is None:
            events = manager.read_events(run_id)
            if events is None:
                raise HTTPException(status_code=404, detail="run not found")

            async def persisted_stream():
                for event in events:
                    yield _sse(event)
                    await asyncio.sleep(0.01)
            return StreamingResponse(persisted_stream(),
                                     media_type="text/event-stream")

        async def live_stream():
            queue: asyncio.Queue = asyncio.Queue()
            backlog = list(record.events)
            record.subscribers.add(queue)
            try:
                for event in backlog:
                    yield _sse(event)
                if record.status in ("finished", "error"):
                    return
                while True:
                    event = await queue.get()
                    yield _sse(event)
                    if event["type"] in ("run_finished", "run_error"):
                        return
            finally:
                record.subscribers.discard(queue)
        return StreamingResponse(live_stream(),
                                 media_type="text/event-stream")

    @app.get("/api/runs/{run_id}/replay")
    async def replay(run_id: str) -> StreamingResponse:
        events = manager.read_events(run_id)
        if events is None:
            raise HTTPException(status_code=404, detail="run not found")

        async def replay_stream():
            yield _sse({"type": "replay_started",
                        "timestamp": _utcnow_iso(), "run_id": run_id,
                        "payload": {"watermark": "REPLAY", "speed": 8}})
            previous_ts: Optional[float] = None
            for event in events:
                delay = 0.01
                raw_ts = event.get("timestamp")
                if isinstance(raw_ts, str):
                    try:
                        current = datetime.fromisoformat(
                            raw_ts.replace("Z", "+00:00")).timestamp()
                        if previous_ts is not None:
                            delay = min(max(
                                (current - previous_ts) / 8.0, 0.0), 0.5)
                        previous_ts = current
                    except ValueError:
                        pass
                await asyncio.sleep(delay)
                yield _sse(event)
            yield _sse({"type": "replay_finished",
                        "timestamp": _utcnow_iso(), "run_id": run_id,
                        "payload": {"watermark": "REPLAY"}})
        return StreamingResponse(replay_stream(),
                                 media_type="text/event-stream")

    @app.get("/api/failures")
    async def failures() -> dict[str, Any]:
        entries = manager.all_failures()
        return {"failures": entries, "count": len(entries)}

    @app.get("/api/benchmark")
    async def benchmark() -> dict[str, Any]:
        for candidate in (reports_dir / "final" / "summary.json",
                          reports_dir / "summary.json"):
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    try:
                        source = str(candidate.relative_to(REPO_ROOT))
                    except ValueError:
                        source = str(candidate)
                    return {"state": "ok", "source": source, "summary": data}
                except ValueError:
                    continue
        return {
            "state": "empty",
            "message": "No benchmark summary yet. Run the Section D "
                       "evaluation harness (eval.report) to produce "
                       "reports/final/summary.json.",
            "summary": None,
            "kpis": {"held_out_success_rate": None,
                     "repair_within_3_attempts": None,
                     "unsafe_executions": None,
                     "false_success_reports": None,
                     "avg_attempts": None,
                     "avg_wall_time_seconds": None,
                     "api_cost_per_run": None},
        }

    return app


app = create_app()
