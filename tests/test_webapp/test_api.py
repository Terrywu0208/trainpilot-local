"""Section E webapp tests: FastAPI TestClient with injected fake agents.

No real server, no network, no Sections A/C required. The fakes follow
the run_agent contract from docs/CONTRACTS.md exactly.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webapp.main import create_app


# ---------------------------------------------------------------------------
# Fake agents
# ---------------------------------------------------------------------------

def _events(*specs):
    return [{"type": t, "timestamp": "2026-11-20T10:00:0%d+00:00" % i,
             "payload": p} for i, (t, p) in enumerate(specs)]


def fake_success_agent(request_text, *, run_dir, approval_handler,
                       session_context=""):
    # No approval gates in this fake; the gated fake below covers them.
    return {
        "success": True, "claimed_success": True, "attempts": 1,
        "final_message": "done",
        "verification": {"checkpoint_exists": True,
                         "checkpoint_loadable": True,
                         "validation_accuracy": 0.9,
                         "config_honors_request": True, "details": {}},
        "diagnosis_records": [],
        "events": _events(
            ("node_started", {"node": "planner"}),
            ("attempt_finished", {"attempt": 1, "status": "succeeded",
                                  "logs": "ok", "fix": {}}),
        ),
        "failures": [],
        "_session_context_seen": session_context,
    }


def fake_gated_agent(request_text, *, run_dir, approval_handler,
                     session_context=""):
    first = approval_handler({
        "action_type": "launch", "description": "First launch",
        "tool_name": "apply_rayjob", "irreversible": False,
        "source": "agent",
    })
    second = approval_handler({
        "action_type": "delete",
        "description": "Delete old checkpoints",
        "tool_name": "purge_attempt_artifacts", "irreversible": True,
        "source": "agent",
        "args": {"paths": ["runs/old/a"], "total_mb": 10},
    })
    return {
        "success": True, "claimed_success": True, "attempts": 3,
        "final_message": f"gate1={first['approve']} gate2={second['approve']}",
        "verification": {"checkpoint_exists": True,
                         "checkpoint_loadable": True,
                         "validation_accuracy": 0.83,
                         "config_honors_request": True, "details": {}},
        "diagnosis_records": [
            {"fault_family": "gpu_only_code",
             "root_cause_code": "NCCL_ON_CPU",
             "evidence": ["log line"], "proposed_fix_code": "GLOO"},
        ],
        "events": _events(
            ("attempt_finished", {"attempt": 1, "status": "failed",
                                  "logs": "nccl", "fix": {"backend": "gloo"}}),
        ),
        "failures": [
            {"fault_label": "NCCL_ON_CPU", "root_cause_code": "NCCL_ON_CPU",
             "evidence": ["log line"], "fix_code": "GLOO",
             "run_id": Path(run_dir).name},
        ],
    }


def fake_unverified_agent(request_text, *, run_dir, approval_handler,
                          session_context=""):
    return {
        "success": False, "claimed_success": False, "attempts": 3,
        "final_message": "could not verify",
        "verification": {"checkpoint_exists": True,
                         "checkpoint_loadable": False,
                         "validation_accuracy": None,
                         "config_honors_request": True, "details": {}},
        "diagnosis_records": [],
        "events": _events(
            ("attempt_finished", {"attempt": 3, "status": "failed",
                                  "logs": "boom", "fix": {}}),
        ),
        "failures": [],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def runs_root(tmp_path):
    return tmp_path / "runs"


def wait_for_status(client: TestClient, run_id: str, statuses: set,
                    timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in statuses:
            return last
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} never reached {statuses}: {last}")


def start_run(client: TestClient, **overrides) -> str:
    body = {"request": "Fine-tune DistilBERT on SST-2, two CPU workers",
            "session_id": "s-test"}
    body.update(overrides)
    resp = client.post("/api/runs", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["run_id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_index_serves_four_views(runs_root, tmp_path):
    app = create_app(run_agent_fn=fake_success_agent, runs_root=runs_root,
                     reports_root=tmp_path / "reports")
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.text
        for view in ("Run Console", "Failure Memory",
                     "Benchmark Dashboard", "Replay", "REPLAY",
                     "UNVERIFIED", "I understand this cannot be undone"):
            assert view in html


def test_create_run_streams_fake_events(runs_root, tmp_path):
    app = create_app(run_agent_fn=fake_success_agent, runs_root=runs_root,
                     reports_root=tmp_path / "reports")
    with TestClient(app) as client:
        run_id = start_run(client)
        wait_for_status(client, run_id, {"finished"})
        with client.stream("GET", f"/api/runs/{run_id}/events") as resp:
            assert resp.status_code == 200
            body = "".join(chunk for chunk in resp.iter_text())
        assert "run_started" in body
        assert "attempt_finished" in body
        assert "run_finished" in body
        # Persisted artifacts per E.1.4
        run_dir = runs_root / run_id
        assert (run_dir / "request.json").exists()
        assert (run_dir / "events.jsonl").exists()
        assert (run_dir / "summary.json").exists()
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["success"] is True
        assert summary["verification"]["passed"] is True


def test_approval_flow_blocks_and_releases(runs_root, tmp_path):
    app = create_app(run_agent_fn=fake_gated_agent, runs_root=runs_root,
                     reports_root=tmp_path / "reports")
    with TestClient(app) as client:
        run_id = start_run(client)
        # Gate 1: the run must block until we approve.
        state = wait_for_status(client, run_id, {"awaiting_approval"})
        assert state["pending_gates"] == ["gate1_first_launch"]
        resp = client.post(f"/api/runs/{run_id}/approve",
                           json={"gate": "gate1_first_launch",
                                 "approve": True})
        assert resp.status_code == 200
        # Gate 2: reject the irreversible deletion.
        state = wait_for_status(client, run_id, {"awaiting_approval"})
        assert state["pending_gates"] == ["gate2_irreversible"]
        resp = client.post(f"/api/runs/{run_id}/approve",
                           json={"gate": "gate2_irreversible",
                                 "approve": False})
        assert resp.status_code == 200
        final = wait_for_status(client, run_id, {"finished"})
        assert "gate1=True" in final["result"]["final_message"]
        assert "gate2=False" in final["result"]["final_message"]
        # Audit log records both decisions.
        audit = (runs_root / run_id / "audit_log.jsonl").read_text()
        entries = [json.loads(line) for line in audit.splitlines() if line]
        assert [e["decision"] for e in entries] == ["approved", "rejected"]
        assert entries[0]["gate_type"] == "gate1_first_launch"
        assert entries[1]["gate_type"] == "gate2_irreversible"
        assert all(e["actor"] == "human" for e in entries)
        # Failure memory endpoint sees the recorded failure.
        failures = client.get("/api/failures").json()
        assert failures["count"] == 1
        assert failures["failures"][0]["fault_label"] == "NCCL_ON_CPU"


def test_approve_without_pending_gate_is_conflict(runs_root, tmp_path):
    app = create_app(run_agent_fn=fake_success_agent, runs_root=runs_root,
                     reports_root=tmp_path / "reports")
    with TestClient(app) as client:
        run_id = start_run(client)
        wait_for_status(client, run_id, {"finished"})
        resp = client.post(f"/api/runs/{run_id}/approve",
                           json={"gate": "gate1_first_launch",
                                 "approve": True})
        assert resp.status_code == 409


def test_failed_verification_is_reported_not_passed(runs_root, tmp_path):
    app = create_app(run_agent_fn=fake_unverified_agent, runs_root=runs_root,
                     reports_root=tmp_path / "reports")
    with TestClient(app) as client:
        run_id = start_run(client)
        final = wait_for_status(client, run_id, {"finished"})
        assert final["success"] is False
        verification = final["verification"]
        assert verification["passed"] is False  # UNVERIFIED, never "success"
        assert verification["checkpoint_loadable"] is False
        summary = json.loads(
            (runs_root / run_id / "summary.json").read_text())
        assert summary["verification"]["passed"] is False


def test_session_followup_passes_previous_summary(runs_root, tmp_path):
    captured = []

    def agent(request_text, *, run_dir, approval_handler,
              session_context=""):
        captured.append(session_context)
        return fake_success_agent(request_text, run_dir=run_dir,
                                  approval_handler=approval_handler,
                                  session_context=session_context)

    app = create_app(run_agent_fn=agent, runs_root=runs_root,
                     reports_root=tmp_path / "reports")
    with TestClient(app) as client:
        first = start_run(client, session_id="sess-42")
        wait_for_status(client, first, {"finished"})
        second = start_run(client, session_id="sess-42",
                           request="Same settings, but epochs = 3")
        wait_for_status(client, second, {"finished"})
        other = start_run(client, session_id="someone-else")
        wait_for_status(client, other, {"finished"})
    assert captured[0] == ""  # first run in a session has no context
    assert first in captured[1]  # follow-up carries the previous summary
    assert "Previous run" in captured[1]
    assert captured[2] == ""  # a different session gets nothing


def test_replay_emits_watermark_and_events(runs_root, tmp_path):
    app = create_app(run_agent_fn=fake_success_agent, runs_root=runs_root,
                     reports_root=tmp_path / "reports")
    with TestClient(app) as client:
        run_id = start_run(client)
        wait_for_status(client, run_id, {"finished"})
        with client.stream("GET", f"/api/runs/{run_id}/replay") as resp:
            assert resp.status_code == 200
            body = "".join(chunk for chunk in resp.iter_text())
        assert "replay_started" in body
        assert "REPLAY" in body
        assert "attempt_finished" in body
        assert "replay_finished" in body


def test_benchmark_empty_state_is_wellformed(runs_root, tmp_path):
    app = create_app(run_agent_fn=fake_success_agent, runs_root=runs_root,
                     reports_root=tmp_path / "no-reports-here")
    with TestClient(app) as client:
        resp = client.get("/api/benchmark")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "empty"
        assert data["summary"] is None
        assert "held_out_success_rate" in data["kpis"]
        assert data["kpis"]["unsafe_executions"] is None


def test_benchmark_reads_summary_when_present(runs_root, tmp_path):
    reports = tmp_path / "reports"
    (reports / "final").mkdir(parents=True)
    (reports / "final" / "summary.json").write_text(json.dumps({
        "held_out_success_rate": 0.8, "unsafe_executions": 0,
        "per_family": {"memory_limits": 0.75},
    }))
    app = create_app(run_agent_fn=fake_success_agent, runs_root=runs_root,
                     reports_root=reports)
    with TestClient(app) as client:
        data = client.get("/api/benchmark").json()
        assert data["state"] == "ok"
        assert data["summary"]["held_out_success_rate"] == 0.8


def test_runs_are_serialized_one_at_a_time(runs_root, tmp_path):
    order = []
    lock = threading.Lock()

    def slow_agent(request_text, *, run_dir, approval_handler,
                   session_context=""):
        with lock:
            order.append(("start", request_text))
        time.sleep(0.4)
        with lock:
            order.append(("end", request_text))
        return fake_success_agent(request_text, run_dir=run_dir,
                                  approval_handler=approval_handler,
                                  session_context=session_context)

    app = create_app(run_agent_fn=slow_agent, runs_root=runs_root,
                     reports_root=tmp_path / "reports")
    with TestClient(app) as client:
        first = start_run(client, request="first request")
        second = start_run(client, request="second request")
        wait_for_status(client, first, {"finished"})
        wait_for_status(client, second, {"finished"})
    starts = [i for i, item in enumerate(order) if item[0] == "start"]
    ends = [i for i, item in enumerate(order) if item[0] == "end"]
    assert order[0] == ("start", "first request")
    assert ends[0] < starts[1]  # second run starts only after first ends



def test_canned_fallback_exercises_both_gates(runs_root, tmp_path):
    """The built-in last-resort demo (used when Sections A/C are absent)
    still goes through Gate 1 and Gate 2 and finishes verified."""
    from webapp.main import _canned_run

    def canned(request_text, *, run_dir, approval_handler,
               session_context=""):
        return _canned_run(request_text, run_dir, approval_handler,
                           session_context)

    app = create_app(run_agent_fn=canned, runs_root=runs_root,
                     reports_root=tmp_path / "reports")
    with TestClient(app) as client:
        run_id = start_run(client)
        state = wait_for_status(client, run_id, {"awaiting_approval"})
        assert state["pending_gates"] == ["gate1_first_launch"]
        client.post(f"/api/runs/{run_id}/approve",
                    json={"gate": "gate1_first_launch", "approve": True})
        state = wait_for_status(client, run_id, {"awaiting_approval"})
        assert state["pending_gates"] == ["gate2_irreversible"]
        client.post(f"/api/runs/{run_id}/approve",
                    json={"gate": "gate2_irreversible", "approve": False})
        final = wait_for_status(client, run_id, {"finished"})
        assert final["success"] is True
        assert final["attempts"] == 3
        assert final["verification"]["passed"] is True
        assert "nothing was deleted" in final["result"]["final_message"]
