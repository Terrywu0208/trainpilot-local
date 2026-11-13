"""Tool registry: the single source of truth for the tool surface.

Exactly 13 tools (design B-4): 7 read tools served by the read MCP server
and 6 action tools served by the action MCP server. The registry is a
read-only constant - the agent cannot register new tools at runtime, which
is itself part of the sandbox (design B-3.3).

The reserved deletion tool ``purge_attempt_artifacts`` is deliberately NOT
in this registry: it lives in ``trainpilot.tools.purge`` and is only wired
up for the approval-gated disk-cleanup case (design B-4, "deliberately
absent tools").
"""
from __future__ import annotations

# Each spec: name, kind (read|action), server, risk_tier, approval_required
# (False, or the approval scope string), timeout_seconds, description,
# dangerous_args (arguments the validator must scrutinize, design B-5).
TOOL_REGISTRY: dict[str, dict] = {
    # ---------------------------------------------------------------- read
    "get_machine_resources": {
        "name": "get_machine_resources",
        "kind": "read",
        "server": "read",
        "risk_tier": "read",
        "approval_required": False,
        "timeout_seconds": 5,
        "description": (
            "Report host CPU/memory/disk and per-node capacity, allocatable "
            "resources and the trainpilot-jobs quota usage. Takes no inputs."
        ),
        "dangerous_args": [],
    },
    "get_pod_status": {
        "name": "get_pod_status",
        "kind": "read",
        "server": "read",
        "risk_tier": "read",
        "approval_required": False,
        "timeout_seconds": 10,
        "description": (
            "List pods belonging to a RayJob (phase, role, node, restarts). "
            "An empty list is a valid result when the pods were cleaned up."
        ),
        "dangerous_args": [],
    },
    "describe_pod": {
        "name": "describe_pod",
        "kind": "read",
        "server": "read",
        "risk_tier": "read",
        "approval_required": False,
        "timeout_seconds": 10,
        "description": (
            "Pod conditions, recent events and container states with the raw "
            "Kubernetes reason strings (OOMKilled, ImagePullBackOff, ...) "
            "preserved for diagnosis."
        ),
        "dangerous_args": [],
    },
    "get_rayjob_status": {
        "name": "get_rayjob_status",
        "kind": "read",
        "server": "read",
        "risk_tier": "read",
        "approval_required": False,
        "timeout_seconds": 10,
        "description": (
            "RayJob status: job_status, deployment status, RayCluster name, "
            "start/end times and spec hash. NOT_FOUND means it was deleted."
        ),
        "dangerous_args": [],
    },
    "get_rayjob_logs": {
        "name": "get_rayjob_logs",
        "kind": "read",
        "server": "read",
        "risk_tier": "read",
        "approval_required": False,
        "timeout_seconds": 15,
        "description": (
            "Tail the submitter/head/worker logs of a RayJob (tail kept, "
            "content capped at 512 KiB per call, full logs stored on disk)."
        ),
        "dangerous_args": [],
    },
    "get_metrics": {
        "name": "get_metrics",
        "kind": "read",
        "server": "read",
        "risk_tier": "read",
        "approval_required": False,
        "timeout_seconds": 10,
        "description": (
            "Per-pod CPU/memory usage and Ray object-store metrics for a "
            "RayJob over a time window. Reports source='unavailable' "
            "instead of inventing numbers."
        ),
        "dangerous_args": [],
    },
    "list_checkpoints": {
        "name": "list_checkpoints",
        "kind": "read",
        "server": "read",
        "risk_tier": "read",
        "approval_required": False,
        "timeout_seconds": 10,
        "description": (
            "List checkpoints written back to the run workspace for one "
            "attempt, with sizes and a completeness flag."
        ),
        "dangerous_args": [],
    },
    # -------------------------------------------------------------- action
    "write_workspace_file": {
        "name": "write_workspace_file",
        "kind": "action",
        "server": "action",
        "risk_tier": "write",
        "approval_required": False,
        "timeout_seconds": 5,
        "description": (
            "Write a text file inside the attempt workspace directory. "
            "Paths cannot escape the attempt directory; existing files "
            "require overwrite=true."
        ),
        "dangerous_args": ["relative_path", "content"],
    },
    "edit_rayjob_yaml": {
        "name": "edit_rayjob_yaml",
        "kind": "action",
        "server": "action",
        "risk_tier": "write",
        "approval_required": False,
        "timeout_seconds": 5,
        "description": (
            "Apply a structured patch (entrypoint, workers, resources, env, "
            "image) to the attempt spec and re-render the RayJob YAML. "
            "Fields outside the patch schema cannot be changed at all."
        ),
        "dangerous_args": [
            "patch.entrypoint",
            "patch.image",
            "patch.env_vars",
            "patch.num_workers",
            "patch.cpu_per_worker",
            "patch.memory_per_worker",
        ],
    },
    "apply_rayjob": {
        "name": "apply_rayjob",
        "kind": "action",
        "server": "action",
        "risk_tier": "write",
        "approval_required": "first_launch",
        "timeout_seconds": 30,
        "description": (
            "Submit the validated rayjob.yaml of an attempt to the cluster. "
            "The first launch of a run requires an approval token; retries "
            "of the same run do not. Same name + same spec hash is "
            "idempotent; same name + different hash is a CONFLICT."
        ),
        "dangerous_args": ["job_name"],
    },
    "build_image": {
        "name": "build_image",
        "kind": "action",
        "server": "action",
        "risk_tier": "build",
        "approval_required": "build_image",
        "timeout_seconds": 600,
        "description": (
            "Build the worker image from an approved base image and the "
            "fixed Dockerfile template, then load it into the Kind cluster. "
            "Every build requires an approval token."
        ),
        "dangerous_args": ["base_image", "requirements_path"],
    },
    "stop_job": {
        "name": "stop_job",
        "kind": "action",
        "server": "action",
        "risk_tier": "write",
        "approval_required": False,
        "timeout_seconds": 30,
        "description": (
            "Suspend a RayJob while keeping the object and its logs for "
            "inspection. Idempotent on already-terminal jobs."
        ),
        "dangerous_args": ["job_name"],
    },
    "delete_job": {
        "name": "delete_job",
        "kind": "action",
        "server": "action",
        "risk_tier": "destructive",
        "approval_required": False,
        "timeout_seconds": 60,
        "description": (
            "Delete a RayJob (and, via the operator, its RayCluster and "
            "pods). Never deletes files on the host; disk cleanup is the "
            "separate approval-gated purge tool."
        ),
        "dangerous_args": ["job_name"],
    },
}

READ_TOOL_NAMES = [n for n, s in TOOL_REGISTRY.items() if s["kind"] == "read"]
ACTION_TOOL_NAMES = [n for n, s in TOOL_REGISTRY.items() if s["kind"] == "action"]
