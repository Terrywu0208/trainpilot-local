"""Read MCP server: the 7 read-only tools, on the fake cluster client.

In production this module serves the same tools over FastMCP (stdio in
development, streamable-http on 127.0.0.1:8765 for the demo) with the
``trainpilot-reader`` ServiceAccount token - get/list/watch only, and no
writer credentials anywhere in its environment (design B-3). Here the
tools run on ``FakeClusterClient`` so they are testable without
Kubernetes; ``call_tool`` is the transport-independent entry point.

Nothing in this module can mutate cluster state.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainpilot.schemas import ToolResult
from trainpilot.tools.context import attempt_dir, coerce_context
from trainpilot.tools.errors import NOT_FOUND, err_result, ok_result
from trainpilot.tools.registry import TOOL_REGISTRY
from trainpilot.tools.validators import validate_tool_call
from trainpilot.mcp.fake_cluster import FakeClusterClient, resolve_client

TOOLS: list[dict] = [spec for spec in TOOL_REGISTRY.values() if spec["server"] == "read"]

_MAX_LOG_CONTENT_BYTES = 512 * 1024  # design B-4.5


def _pod_brief(pod: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": pod["name"],
        "phase": pod["phase"],
        "role": pod["role"],
        "node_name": pod["node_name"],
        "restarts": pod["restarts"],
        "ready": pod["ready"],
        "created_at": pod["created_at"],
    }


def _list_checkpoints(args: dict[str, Any], ctx: Any) -> ToolResult:
    tool_name = "list_checkpoints"
    base = attempt_dir(ctx, args["run_id"], args["attempt_id"]) / "checkpoints"
    if not base.is_dir():
        return ok_result(tool_name, {
            "run_id": args["run_id"], "attempt_id": args["attempt_id"],
            "checkpoints": [], "total_size_bytes": 0,
        })
    checkpoints: list[dict[str, Any]] = []
    total = 0
    dirs = sorted(p for p in base.iterdir() if p.is_dir())
    if not dirs and any(p.is_file() for p in base.iterdir()):
        dirs = [base]
    file_count = 0
    truncated = False
    for ckpt in dirs:
        size = 0
        complete_files = {"model.safetensors": 0, "config.json": 0, "metrics.json": 0}
        for path in sorted(ckpt.rglob("*")):
            if path.is_file():
                file_count += 1
                if file_count > 500:
                    truncated = True
                    break
                size += path.stat().st_size
                if path.name in complete_files and path.stat().st_size > 0:
                    complete_files[path.name] = 1
        total += size
        checkpoints.append({
            "name": ckpt.name,
            "path": ckpt.relative_to(base.parent).as_posix() if ckpt != base else "checkpoints",
            "size_bytes": size,
            "modified_at": datetime.fromtimestamp(ckpt.stat().st_mtime, timezone.utc).isoformat(),
            "is_complete": all(complete_files.values()),
        })
    data: dict[str, Any] = {
        "run_id": args["run_id"],
        "attempt_id": args["attempt_id"],
        "checkpoints": checkpoints,
        "total_size_bytes": total,
    }
    if truncated:
        data["truncated"] = True
    return ok_result(tool_name, data)


def call_tool(name: str, args: dict[str, Any], context: Any = None) -> ToolResult:
    """Run one read tool. Always returns a ToolResult, never raises."""
    ctx = coerce_context(context)
    spec = TOOL_REGISTRY.get(name)
    if spec is None or spec["server"] != "read":
        return err_result(str(name), "VALIDATION_FAILED", f"{name!r} is not a read-server tool")
    check = validate_tool_call(name, args, ctx)
    if not check.ok:
        return check
    client: FakeClusterClient = resolve_client(ctx)

    if name == "get_machine_resources":
        return ok_result(name, client.machine_resources())
    if name == "get_pod_status":
        pods = [_pod_brief(p) for p in client.list_pods(args["job_name"])]
        return ok_result(name, {"job_name": args["job_name"], "pods": pods, "total": len(pods)})
    if name == "describe_pod":
        pod = client.get_pod(args["pod_name"])
        if pod is None:
            return err_result(name, NOT_FOUND, f"pod {args['pod_name']!r} not found")
        return ok_result(name, {
            "pod_name": pod["name"],
            "phase": pod["phase"],
            "conditions": pod["conditions"],
            "events": pod["events"][-20:],
            "container_statuses": pod["container_statuses"],
            "resource_requests": pod["resource_requests"],
            "resource_limits": pod["resource_limits"],
        })
    if name == "get_rayjob_status":
        job = client.get_rayjob(args["job_name"])
        if job is None:
            return err_result(name, NOT_FOUND, f"rayjob {args['job_name']!r} not found")
        return ok_result(name, {
            "job_name": job["job_name"],
            "job_status": job["job_status"],
            "job_deployment_status": job["job_deployment_status"],
            "ray_cluster_name": job["ray_cluster_name"],
            "start_time": job["start_time"],
            "end_time": job["end_time"],
            "message": job["message"],
            "failed_attempts": job["failed_attempts"],
            "spec_hash": job["spec_hash"],
        })
    if name == "get_rayjob_logs":
        if client.get_rayjob(args["job_name"]) is None:
            return err_result(name, NOT_FOUND, f"rayjob {args['job_name']!r} not found")
        sources = client.get_logs(
            args["job_name"],
            container=args.get("container"),
            tail_lines=args.get("tail_lines", 500),
        )
        total_bytes = 0
        for source in sources:
            encoded = source["content"].encode()
            if len(encoded) > _MAX_LOG_CONTENT_BYTES:
                source["content"] = encoded[-_MAX_LOG_CONTENT_BYTES:].decode(errors="replace")
                source["truncated"] = True
            total_bytes += len(source["content"].encode())
        return ok_result(name, {
            "job_name": args["job_name"],
            "sources": sources,
            "total_bytes": total_bytes,
            "stored_paths": [],
        })
    if name == "get_metrics":
        metrics = client.get_metrics(args["job_name"])
        if metrics is None:
            return err_result(name, NOT_FOUND, f"rayjob {args['job_name']!r} not found")
        return ok_result(name, metrics)
    if name == "list_checkpoints":
        return _list_checkpoints(args, ctx)
    return err_result(name, "VALIDATION_FAILED", f"unhandled read tool: {name!r}")


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
    server = FastMCP("trainpilot-read", host="127.0.0.1", port=8765)
    bound_client = client or FakeClusterClient()

    def _make(tool_name: str):
        def _tool(**kwargs: Any) -> dict:
            result = call_tool(tool_name, kwargs, {"cluster": bound_client})
            return result.model_dump()

        return _tool

    for spec in TOOLS:
        server.add_tool(_make(spec["name"]), name=spec["name"], description=spec["description"])
    return server
