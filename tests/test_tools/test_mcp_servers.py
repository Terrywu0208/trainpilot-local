"""Server round-trip tests on the fake cluster: edit -> apply (gated) ->
status -> logs -> checkpoints -> stop -> delete."""
from __future__ import annotations

from pathlib import Path

import pytest

from trainpilot.mcp import action_server, read_server
from trainpilot.mcp.fake_cluster import FakeClusterClient
from trainpilot.tools import approval_tokens
from trainpilot.tools.context import ToolContext

SECRET = "server-secret"
RUN = "r-20261120-a3f9"
ATTEMPT = "attempt-01"


@pytest.fixture()
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace_root=tmp_path / "workspaces",
        secret=SECRET,
        cluster=FakeClusterClient(),
    )


def _launch(ctx: ToolContext) -> tuple[str, str]:
    """Edit the spec and apply with a first-launch token. Returns (job, hash)."""
    edit = action_server.call_tool(
        "edit_rayjob_yaml",
        {"run_id": RUN, "attempt_id": ATTEMPT,
         "patch": {"num_workers": 2, "memory_per_worker": "2Gi"}},
        ctx,
    )
    assert edit.ok, edit.error_message
    spec_hash = edit.data["spec_hash"]
    job_name = edit.data["job_name"]
    apply_args = {"job_name": job_name, "run_id": RUN, "attempt_id": ATTEMPT,
                  "spec_hash": spec_hash}
    token = approval_tokens.issue_token(
        SECRET, "first_launch", approval_tokens.apply_binding(apply_args)
    )
    applied = action_server.call_tool(
        "apply_rayjob", dict(apply_args, approval_token=token), ctx
    )
    assert applied.ok, applied.error_message
    assert applied.data["created"] is True
    return job_name, spec_hash


def test_full_round_trip(ctx):
    job_name, _ = _launch(ctx)

    status = read_server.call_tool("get_rayjob_status", {"job_name": job_name}, ctx)
    assert status.ok
    assert status.data["job_status"] == "RUNNING"
    assert status.data["ray_cluster_name"]

    pods = read_server.call_tool("get_pod_status", {"job_name": job_name}, ctx)
    assert pods.ok and pods.data["total"] >= 3
    roles = {p["role"] for p in pods.data["pods"]}
    assert roles == {"submitter", "head", "worker"}

    one_pod = pods.data["pods"][0]["name"]
    described = read_server.call_tool("describe_pod", {"pod_name": one_pod}, ctx)
    assert described.ok and described.data["phase"] == "Running"

    logs = read_server.call_tool("get_rayjob_logs", {"job_name": job_name}, ctx)
    assert logs.ok and logs.data["sources"]
    containers = {s["container"] for s in logs.data["sources"]}
    assert "submitter" in containers

    metrics = read_server.call_tool("get_metrics", {"job_name": job_name}, ctx)
    assert metrics.ok and metrics.data["pods"]

    resources = read_server.call_tool("get_machine_resources", {}, ctx)
    assert resources.ok and resources.data["host"]["cpu_count"] >= 1
    assert len(resources.data["nodes"]) == 3

    ctx.cluster.finish_job(job_name, success=True)
    status = read_server.call_tool("get_rayjob_status", {"job_name": job_name}, ctx)
    assert status.data["job_status"] == "SUCCEEDED"

    stopped = action_server.call_tool(
        "stop_job", {"job_name": job_name, "reason": "checking idempotency"}, ctx
    )
    assert stopped.ok and stopped.data["already_terminal"] is True

    deleted = action_server.call_tool("delete_job", {"job_name": job_name}, ctx)
    assert deleted.ok and deleted.data["deleted"] is True
    gone = read_server.call_tool("get_rayjob_status", {"job_name": job_name}, ctx)
    assert not gone.ok and gone.error_code == "NOT_FOUND"
    deleted_again = action_server.call_tool("delete_job", {"job_name": job_name}, ctx)
    assert deleted_again.ok and deleted_again.data["deleted"] is False


def test_first_launch_gate_and_idempotent_reapply(ctx):
    edit = action_server.call_tool(
        "edit_rayjob_yaml",
        {"run_id": RUN, "attempt_id": ATTEMPT, "patch": {"num_workers": 2}},
        ctx,
    )
    job_name, spec_hash = edit.data["job_name"], edit.data["spec_hash"]
    apply_args = {"job_name": job_name, "run_id": RUN, "attempt_id": ATTEMPT,
                  "spec_hash": spec_hash}

    refused = action_server.call_tool("apply_rayjob", apply_args, ctx)
    assert not refused.ok and refused.error_code == "APPROVAL_REQUIRED"
    assert read_server.call_tool(
        "get_rayjob_status", {"job_name": job_name}, ctx
    ).error_code == "NOT_FOUND"

    token = approval_tokens.issue_token(
        SECRET, "first_launch", approval_tokens.apply_binding(apply_args)
    )
    applied = action_server.call_tool(
        "apply_rayjob", dict(apply_args, approval_token=token), ctx
    )
    assert applied.ok and applied.data["created"] is True

    # Same name + same hash, no token: idempotent re-apply of a launched run.
    again = action_server.call_tool("apply_rayjob", apply_args, ctx)
    assert again.ok and again.data["created"] is False


def test_apply_conflict_on_changed_spec(ctx):
    job_name, _ = _launch(ctx)
    # Re-editing the spec changes the validated rayjob.yaml/spec.json on
    # disk. Re-applying under the OLD job name must conflict rather than
    # overwrite (design B-4.10: the server trusts the file, not the
    # caller's claimed hash).
    edit = action_server.call_tool(
        "edit_rayjob_yaml",
        {"run_id": RUN, "attempt_id": ATTEMPT, "patch": {"num_workers": 3}},
        ctx,
    )
    assert edit.ok and edit.data["job_name"] != job_name
    changed = action_server.call_tool(
        "apply_rayjob",
        {"job_name": job_name, "run_id": RUN, "attempt_id": ATTEMPT,
         "spec_hash": edit.data["spec_hash"]},
        ctx,
    )
    assert not changed.ok and changed.error_code == "CONFLICT"


def test_write_workspace_file_round_trip(ctx):
    args = {"run_id": RUN, "attempt_id": ATTEMPT, "relative_path": "train.py",
            "content": "print('train')\n"}
    written = action_server.call_tool("write_workspace_file", args, ctx)
    assert written.ok and written.data["bytes_written"] == len("print('train')\n")
    on_disk = ctx.workspace_root / RUN / "attempt-01" / "train.py"
    assert on_disk.read_text() == "print('train')\n"

    conflict = action_server.call_tool("write_workspace_file", args, ctx)
    assert not conflict.ok and conflict.error_code == "CONFLICT"

    overwritten = action_server.call_tool(
        "write_workspace_file", dict(args, content="print('v2')\n", overwrite=True), ctx
    )
    assert overwritten.ok
    assert on_disk.read_text() == "print('v2')\n"


def test_list_checkpoints_with_complete_checkpoint(ctx):
    ckpt = ctx.workspace_root / RUN / "attempt-01" / "checkpoints" / "checkpoint-final"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"weights")
    (ckpt / "config.json").write_text("{}")
    (ckpt / "metrics.json").write_text('{"eval_accuracy": 0.9}')

    result = read_server.call_tool(
        "list_checkpoints", {"run_id": RUN, "attempt_id": ATTEMPT}, ctx
    )
    assert result.ok
    assert len(result.data["checkpoints"]) == 1
    entry = result.data["checkpoints"][0]
    assert entry["name"] == "checkpoint-final"
    assert entry["is_complete"] is True
    assert entry["size_bytes"] > 0


def test_list_checkpoints_empty_is_ok(ctx):
    result = read_server.call_tool(
        "list_checkpoints", {"run_id": RUN, "attempt_id": ATTEMPT}, ctx
    )
    assert result.ok and result.data["checkpoints"] == []


def test_servers_reject_each_others_tools(ctx):
    assert not read_server.call_tool("delete_job", {"job_name": "tp-x"}, ctx).ok
    assert not action_server.call_tool("get_rayjob_status", {"job_name": "tp-x"}, ctx).ok
    assert not action_server.call_tool("exec_command", {"command": "ls"}, ctx).ok


def test_tools_listings_match_registry_split():
    assert {t["name"] for t in read_server.TOOLS} == {
        "get_machine_resources", "get_pod_status", "describe_pod",
        "get_rayjob_status", "get_rayjob_logs", "get_metrics", "list_checkpoints",
    }
    assert {t["name"] for t in action_server.TOOLS} == {
        "write_workspace_file", "edit_rayjob_yaml", "apply_rayjob",
        "build_image", "stop_job", "delete_job",
    }
