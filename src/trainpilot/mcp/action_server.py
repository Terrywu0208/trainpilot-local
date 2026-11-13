"""Action MCP server: the 6 write tools, on the fake cluster client.

In production this module serves the same tools over FastMCP (stdio in
development, streamable-http on 127.0.0.1:8766 for the demo) with the
``trainpilot-writer`` ServiceAccount token - RayJob CRUD only, explicitly
no pods/exec (design B-1.5, B-3). Every call passes through
``validate_tool_call`` first (allowlist, path/image/limit checks and the
approval gates), so a gated or invalid call never reaches the cluster.

There is no docker/kind/kubectl in this environment: image builds are
simulated by the fake cluster client and reported as such.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from trainpilot.schemas import ToolResult
from trainpilot.tools.context import attempt_dir, attempt_dir_name, coerce_context
from trainpilot.tools.errors import CONFLICT, NOT_FOUND, err_result, ok_result
from trainpilot.tools.registry import TOOL_REGISTRY
from trainpilot.tools.validators import resolve_workspace_path, validate_tool_call
from trainpilot.mcp.fake_cluster import TERMINAL_JOB_STATUSES, FakeClusterClient, resolve_client

TOOLS: list[dict] = [spec for spec in TOOL_REGISTRY.values() if spec["server"] == "action"]

DEFAULT_SPEC: dict[str, Any] = {
    "entrypoint": "python train.py --config config.yaml",
    "num_workers": 2,
    "cpu_per_worker": "1",
    "memory_per_worker": "2Gi",
    "env_vars": {"HF_HUB_OFFLINE": "1"},
    "image": "rayproject/ray:2.41.0-py310-cpu",
}


def _spec_hash(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _job_name(run_id: str, attempt_id: str, spec_hash: str) -> str:
    middle = re.sub(r"[^a-z0-9]", "", str(run_id).lower())[:38]
    digits = re.sub(r"\D", "", str(attempt_id))
    attempt_num = f"{int(digits):02d}" if digits else "01"
    return f"tp-{middle}-a{attempt_num}-{spec_hash[:6]}"[:53]


def _render_rayjob_yaml(spec: dict[str, Any], job_name: str, run_id: str, attempt_id: str, spec_hash: str) -> str:
    resources = {
        "requests": {"cpu": "500m", "memory": "512Mi"},
        "limits": {"cpu": str(spec["cpu_per_worker"]), "memory": str(spec["memory_per_worker"])},
    }
    container = {"name": "ray-worker", "image": spec["image"], "resources": resources}
    document = {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": job_name,
            "namespace": "trainpilot-jobs",
            "labels": {
                "app.kubernetes.io/part-of": "trainpilot",
                "trainpilot.io/run-id": str(run_id),
                "trainpilot.io/attempt": str(attempt_id),
                "trainpilot.io/spec-hash": spec_hash,
            },
            "annotations": {"trainpilot.io/created-by": "trainpilot-agent"},
        },
        "spec": {
            "entrypoint": spec["entrypoint"],
            "ttlSecondsAfterFinished": 300,
            "shutdownAfterJobFinishes": True,
            "runtimeEnv": {
                "working_dir": f"/trainpilot/workspaces/{run_id}/{attempt_dir_name(attempt_id)}",
                "env_vars": spec["env_vars"],
            },
            "rayClusterSpec": {
                "rayVersion": "2.41.0",
                "headGroupSpec": {
                    "rayStartParams": {"dashboard-host": "0.0.0.0"},
                    "template": {"spec": {"containers": [dict(container, name="ray-head")]}},
                },
                "workerGroupSpecs": [
                    {
                        "groupName": "workers",
                        "replicas": spec["num_workers"],
                        "minReplicas": spec["num_workers"],
                        "maxReplicas": spec["num_workers"],
                        "rayStartParams": {},
                        "template": {"spec": {"containers": [container]}},
                    }
                ],
            },
        },
    }
    return yaml.safe_dump(document, sort_keys=False)


# --------------------------------------------------------------------------
# Tool implementations (validation has already passed at this point)
# --------------------------------------------------------------------------

def _write_workspace_file(args: dict[str, Any], ctx: Any) -> ToolResult:
    name = "write_workspace_file"
    resolved, err = resolve_workspace_path(ctx, args["run_id"], args["attempt_id"], args["relative_path"])
    assert resolved is not None and err is None
    if resolved.exists() and not args.get("overwrite", False):
        return err_result(name, CONFLICT, f"file {args['relative_path']!r} already exists and overwrite is false")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    data = args["content"].encode("utf-8")
    resolved.write_bytes(data)
    base = attempt_dir(ctx, args["run_id"], args["attempt_id"])
    return ok_result(name, {
        "path": f"{args['run_id']}/{attempt_dir_name(args['attempt_id'])}/{args['relative_path']}",
        "bytes_written": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "attempt_dir": str(base),
    })


def _edit_rayjob_yaml(args: dict[str, Any], ctx: Any) -> ToolResult:
    name = "edit_rayjob_yaml"
    base = attempt_dir(ctx, args["run_id"], args["attempt_id"])
    base.mkdir(parents=True, exist_ok=True)
    spec = dict(DEFAULT_SPEC)
    spec_file = base / "spec.json"
    if spec_file.exists():
        stored = json.loads(spec_file.read_text())
        if isinstance(stored.get("spec"), dict):
            spec.update(stored["spec"])
    for key, value in args["patch"].items():
        if value is not None:
            spec[key] = value
    spec_hash = _spec_hash(spec)
    job_name = _job_name(args["run_id"], args["attempt_id"], spec_hash)
    rendered = _render_rayjob_yaml(spec, job_name, args["run_id"], args["attempt_id"], spec_hash)
    (base / "rayjob.yaml").write_text(rendered)
    spec_file.write_text(json.dumps(
        {"spec": spec, "spec_hash": spec_hash, "job_name": job_name}, indent=2, sort_keys=True
    ))
    return ok_result(name, {
        "rayjob_yaml_path": str(base / "rayjob.yaml"),
        "rendered_yaml": rendered,
        "spec_hash": spec_hash,
        "job_name": job_name,
        "validation": {"passed": True, "warnings": []},
    })


def _apply_rayjob(args: dict[str, Any], ctx: Any, client: FakeClusterClient) -> ToolResult:
    name = "apply_rayjob"
    base = attempt_dir(ctx, args["run_id"], args["attempt_id"])
    spec_hash = str(args.get("spec_hash", ""))
    num_workers = 2
    spec_file = base / "spec.json"
    if spec_file.exists():
        stored = json.loads(spec_file.read_text())
        spec_hash = stored.get("spec_hash", spec_hash)
        if isinstance(stored.get("spec"), dict):
            num_workers = int(stored["spec"].get("num_workers", 2))
    existing = client.get_rayjob(args["job_name"])
    if existing is not None:
        if existing["spec_hash"] == spec_hash:
            return ok_result(name, {
                "job_name": existing["job_name"], "uid": existing["uid"],
                "resource_version": existing["resource_version"], "created": False,
                "ray_cluster_name": existing["ray_cluster_name"],
            })
        return err_result(
            name, CONFLICT,
            "a RayJob with this name but a different spec_hash already exists; delete it first",
        )
    job = client.create_rayjob(
        args["job_name"], run_id=args["run_id"], attempt_id=args["attempt_id"],
        spec_hash=spec_hash, num_workers=num_workers,
    )
    ctx.applied_run_ids.add(args["run_id"])
    return ok_result(name, {
        "job_name": job["job_name"], "uid": job["uid"],
        "resource_version": job["resource_version"], "created": True,
        "ray_cluster_name": job["ray_cluster_name"],
    })


def _build_image(args: dict[str, Any], ctx: Any, client: FakeClusterClient) -> ToolResult:
    name = "build_image"
    resolved, _ = resolve_workspace_path(ctx, args["run_id"], args["attempt_id"], args["requirements_path"])
    content = ""
    if resolved is not None and resolved.exists():
        content = resolved.read_text(errors="replace")
    cache_key = "|".join([args["base_image"], args["requirements_path"], hashlib.sha256(content.encode()).hexdigest()])
    cached_digest = client.build_image_cached(cache_key)
    cached = cached_digest is not None
    digest = cached_digest or client.record_built_image(cache_key, cache_key)
    digits = re.sub(r"\D", "", str(args["attempt_id"]))
    attempt_num = f"{int(digits):02d}" if digits else "01"
    image_ref = f"trainpilot/worker:{args['run_id']}-a{attempt_num}-{digest[7:13]}"
    return ok_result(name, {
        "image_ref": image_ref,
        "image_digest": digest,
        "build_seconds": 0.0,
        "kind_loaded": False,
        "cached": cached,
        "simulated": True,
        "note": "no docker/kind in this environment; the build is simulated by the fake cluster client",
    })


def _stop_job(args: dict[str, Any], client: FakeClusterClient) -> ToolResult:
    name = "stop_job"
    job = client.get_rayjob(args["job_name"])
    if job is None:
        return err_result(name, NOT_FOUND, f"rayjob {args['job_name']!r} not found")
    if job["job_status"] in TERMINAL_JOB_STATUSES:
        return ok_result(name, {
            "job_name": args["job_name"], "previous_status": job["job_status"],
            "stopped": False, "already_terminal": True,
        })
    previous = client.suspend_rayjob(args["job_name"])
    return ok_result(name, {
        "job_name": args["job_name"], "previous_status": previous, "stopped": True,
    })


def _delete_job(args: dict[str, Any], client: FakeClusterClient) -> ToolResult:
    name = "delete_job"
    existed = client.delete_rayjob(args["job_name"])
    return ok_result(name, {
        "job_name": args["job_name"], "deleted": existed,
        "pods_remaining": 0, "cleanup_complete": True,
    })


def call_tool(name: str, args: dict[str, Any], context: Any = None) -> ToolResult:
    """Run one action tool. Validation always runs first; never raises."""
    ctx = coerce_context(context)
    spec = TOOL_REGISTRY.get(name)
    if spec is None or spec["server"] != "action":
        return err_result(str(name), "VALIDATION_FAILED", f"{name!r} is not an action-server tool")
    check = validate_tool_call(name, args, ctx)
    if not check.ok:
        return check
    client: FakeClusterClient = resolve_client(ctx)
    if name == "write_workspace_file":
        return _write_workspace_file(args, ctx)
    if name == "edit_rayjob_yaml":
        return _edit_rayjob_yaml(args, ctx)
    if name == "apply_rayjob":
        return _apply_rayjob(args, ctx, client)
    if name == "build_image":
        return _build_image(args, ctx, client)
    if name == "stop_job":
        return _stop_job(args, client)
    if name == "delete_job":
        return _delete_job(args, client)
    return err_result(name, "VALIDATION_FAILED", f"unhandled action tool: {name!r}")


def build_fastmcp_server(client: FakeClusterClient | None = None, **_: Any):
    """Build a FastMCP server for these tools if the `mcp` package exists.

    Optional: tests and the rest of the project never require it.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the 'mcp' package is not installed; use call_tool() instead"
        ) from exc
    server = FastMCP("trainpilot-action", host="127.0.0.1", port=8766)
    bound_client = client or FakeClusterClient()

    def _make(tool_name: str):
        def _tool(**kwargs: Any) -> dict:
            result = call_tool(tool_name, kwargs, {"cluster": bound_client})
            return result.model_dump()

        return _tool

    for spec in TOOLS:
        server.add_tool(_make(spec["name"]), name=spec["name"], description=spec["description"])
    return server
