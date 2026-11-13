"""Layer 1 of the three-layer sandbox: tool-call validation (design B-5.2).

``validate_tool_call`` is a pure check: it performs no I/O against a
cluster and mutates nothing except reading file existence for path
checks. It enforces, in order:

1. **Allowlist** - the tool must be one of the 13 in ``TOOL_REGISTRY``.
2. **Name hygiene** - ``job_name`` / ``run_id`` / ``attempt_id`` must match
   the conservative DNS-1123 subset from design B-4; a ``namespace``
   argument other than the fixed jobs namespace is rejected (B-1.3).
3. **No general shell** - the ``entrypoint`` field (the only input that
   resembles a command) must be an allowlisted Python invocation shape
   with no shell metacharacters or dangerous tokens (design B-5.5), and
   env values / image references pass character checks.
4. **Path allowlist** - file paths must resolve inside the attempt
   workspace directory; no ``..``, no absolute paths, no symlink escape,
   extension allowlist for written files.
5. **Approved images** - image references must match the approved list by
   name and digest (never by tag alone); ``latest`` is always rejected.
6. **Resource limits** - worker count / CPU / memory from a structured
   patch must fit the fixed caps in the context (design B-5.1).
7. **Approval gates** - first-launch ``apply_rayjob`` and every
   ``build_image`` require a valid HMAC approval token (design B-5.4).

Kubernetes-level enforcement (ResourceQuota, Pod Security, RBAC in
``infra/k8s/``) is layers 2 and 3, so the sandbox still holds even if a
bug slips through here.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from trainpilot.schemas import ToolResult
from trainpilot.tools import approval_tokens
from trainpilot.tools.context import JOBS_NAMESPACE, ToolContext, attempt_dir, coerce_context
from trainpilot.tools.errors import (
    APPROVAL_REQUIRED,
    IMAGE_NOT_APPROVED,
    PATH_NOT_ALLOWED,
    RESOURCE_LIMIT_EXCEEDED,
    VALIDATION_FAILED,
    err_result,
    ok_result,
)
from trainpilot.tools.registry import TOOL_REGISTRY

# DNS-1123 subdomain conservative subset, max 53 chars (design B-4).
NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]{0,51}[a-z0-9])?$")
# Pod names are DNS subdomains and may contain dots.
POD_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$")
ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

ALLOWED_FILE_EXTENSIONS = {".py", ".yaml", ".yml", ".txt", ".md", ".json", ".toml", ".cfg", ".sh"}
PATCHABLE_FIELDS = {"entrypoint", "num_workers", "cpu_per_worker", "memory_per_worker", "env_vars", "image"}
ALLOWED_DOCKERFILE_TEMPLATES = {"ray-train-cpu"}

SHELL_METACHARACTERS = set(";&|$`\\<>\n\r\x00")
DANGEROUS_ENTRYPOINT_TOKENS = {
    "curl", "wget", "bash", "sh", "sudo", "kubectl", "docker", "rm",
    "chmod", "chown", "nc", "ssh", "scp", "eval", "exec",
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def parse_cpu(value: Any) -> float | None:
    """Parse a Kubernetes CPU quantity ('500m', '2', 1.5) into cores."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        if text.endswith("m"):
            return float(text[:-1]) / 1000.0
        return float(text)
    except ValueError:
        return None


def parse_memory(value: Any) -> int | None:
    """Parse a Kubernetes memory quantity ('512Mi', '3Gi', '1G') to bytes."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    units = {
        "Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
        "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4,
    }
    for suffix in sorted(units, key=len, reverse=True):
        if text.endswith(suffix):
            try:
                return int(float(text[: -len(suffix)]) * units[suffix])
            except ValueError:
                return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_entrypoint(entrypoint: Any) -> str | None:
    """Return an error message if the entrypoint is not an allowlisted
    Python invocation, else None (design B-5.5)."""
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        return "entrypoint must be a non-empty string"
    if any(ch in SHELL_METACHARACTERS for ch in entrypoint):
        return "entrypoint contains shell metacharacters"
    if "$(" in entrypoint or "`" in entrypoint:
        return "entrypoint contains command substitution"
    # Only plain flags/values after the program part: letters, digits and
    # a small punctuation set (paths, --flag=value, dots, dashes).
    if not re.fullmatch(r"[A-Za-z0-9_./:=+\- ]+", entrypoint):
        return "entrypoint contains characters outside the allowlist"
    tokens = entrypoint.split()
    if tokens[0] not in ("python", "python3"):
        return "entrypoint must start with python/python3"
    rest = tokens[1:]
    if not rest:
        return "entrypoint must name a script or module"
    if rest[0] == "-m":
        if len(rest) < 2 or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", rest[1]):
            return "entrypoint module after -m is not a plain module name"
    elif rest[0] != "train.py":
        return "entrypoint script must be train.py (or use python -m <module>)"
    lowered = {t.lower().strip("-") for t in tokens}
    bad = lowered & DANGEROUS_ENTRYPOINT_TOKENS
    if bad:
        return f"entrypoint contains forbidden token(s): {sorted(bad)}"
    if "-c" in tokens:
        return "entrypoint must not use python -c (inline code)"
    return None


def check_image_approved(image: Any, ctx: ToolContext) -> str | None:
    """Return an error message if the image reference is not approved."""
    if not isinstance(image, str) or not image.strip():
        return "image must be a non-empty string"
    if any(ch in SHELL_METACHARACTERS or ch.isspace() for ch in image):
        return "image reference contains illegal characters"
    if image.endswith(":latest") or image == "latest":
        return "the 'latest' tag is not allowed; pin a version and digest"
    if "@" in image:
        name, _, digest = image.partition("@")
        expected = ctx.approved_images.get(name)
        if expected is None:
            return f"image {name!r} is not on the approved list"
        if digest != expected:
            return f"image digest {digest!r} does not match the approved digest for {name!r}"
        return None
    if image not in ctx.approved_images:
        return f"image {image!r} is not on the approved list"
    return None


def resolve_workspace_path(
    ctx: ToolContext, run_id: str, attempt_id: str, relative_path: Any
) -> tuple[Path | None, str | None]:
    """Resolve a workspace-relative path, refusing escapes.

    Returns (resolved_path, None) on success or (None, error_message).
    """
    if not isinstance(relative_path, str) or not relative_path:
        return None, "relative_path must be a non-empty string"
    rel = Path(relative_path)
    if rel.is_absolute():
        return None, "absolute paths are not allowed"
    if any(part == ".." for part in rel.parts):
        return None, "path must not contain '..'"
    try:
        base = attempt_dir(ctx, run_id, attempt_id).resolve()
    except ValueError as exc:
        return None, str(exc)
    resolved = (base / rel).resolve()
    if resolved != base and base not in resolved.parents:
        return None, "path escapes the attempt workspace (symlink or traversal)"
    return resolved, None


def _check_names(args: dict, fields: tuple[str, ...]) -> str | None:
    for field_name in fields:
        value = args.get(field_name)
        if value is None:
            return f"missing required argument: {field_name}"
        if not isinstance(value, str) or not NAME_RE.match(value):
            return f"{field_name} {value!r} is not a valid DNS-1123 name (max 53 chars)"
    return None


def _check_namespace(args: dict) -> str | None:
    namespace = args.get("namespace")
    if namespace is not None and namespace != JOBS_NAMESPACE:
        return f"namespace is fixed to {JOBS_NAMESPACE!r}; got {namespace!r}"
    return None


def _run_already_applied(ctx: ToolContext, run_id: str) -> bool:
    if run_id in ctx.applied_run_ids:
        return True
    cluster = ctx.cluster
    applied = getattr(cluster, "applied_runs", None)
    return bool(applied) and run_id in applied


# --------------------------------------------------------------------------
# Per-tool validators. Each returns None when the call is acceptable, or a
# ToolResult describing the rejection.
# --------------------------------------------------------------------------

def _validate_read(tool_name: str, args: dict, ctx: ToolContext) -> ToolResult | None:
    err = _check_namespace(args)
    if err:
        return err_result(tool_name, VALIDATION_FAILED, err)
    if tool_name in ("get_pod_status", "get_rayjob_status", "get_rayjob_logs", "get_metrics"):
        err = _check_names(args, ("job_name",))
        if err:
            return err_result(tool_name, VALIDATION_FAILED, err)
    if tool_name == "describe_pod":
        pod_name = args.get("pod_name")
        if not isinstance(pod_name, str) or not POD_NAME_RE.match(pod_name):
            return err_result(tool_name, VALIDATION_FAILED, f"pod_name {pod_name!r} is not a valid pod name")
    if tool_name == "get_rayjob_logs":
        tail = args.get("tail_lines", 500)
        if not _is_int(tail) or not (1 <= tail <= 2000):
            return err_result(tool_name, VALIDATION_FAILED, "tail_lines must be an int in [1, 2000]")
        since = args.get("since_seconds")
        if since is not None and (not _is_int(since) or not (1 <= since <= 3600)):
            return err_result(tool_name, VALIDATION_FAILED, "since_seconds must be an int in [1, 3600]")
        container = args.get("container")
        if container is not None and container not in ("submitter", "head", "worker"):
            return err_result(tool_name, VALIDATION_FAILED, f"unknown container selector: {container!r}")
    if tool_name == "get_metrics":
        window = args.get("window_seconds", 300)
        if not _is_int(window) or not (1 <= window <= 1800):
            return err_result(tool_name, VALIDATION_FAILED, "window_seconds must be an int in [1, 1800]")
    if tool_name == "list_checkpoints":
        err = _check_names(args, ("run_id", "attempt_id"))
        if err:
            return err_result(tool_name, VALIDATION_FAILED, err)
        if ctx.workspace_root is None:
            return err_result(tool_name, VALIDATION_FAILED, "context.workspace_root is not set")
    return None


def _validate_write_workspace_file(tool_name: str, args: dict, ctx: ToolContext) -> ToolResult | None:
    err = _check_names(args, ("run_id", "attempt_id"))
    if err:
        return err_result(tool_name, VALIDATION_FAILED, err)
    resolved, err = resolve_workspace_path(ctx, args["run_id"], args["attempt_id"], args.get("relative_path"))
    if err:
        return err_result(tool_name, PATH_NOT_ALLOWED, err)
    assert resolved is not None
    if resolved.suffix not in ALLOWED_FILE_EXTENSIONS:
        return err_result(
            tool_name, PATH_NOT_ALLOWED,
            f"file extension {resolved.suffix!r} is not in the allowlist {sorted(ALLOWED_FILE_EXTENSIONS)}",
        )
    content = args.get("content")
    if not isinstance(content, str):
        return err_result(tool_name, VALIDATION_FAILED, "content must be a string")
    if "\x00" in content:
        return err_result(tool_name, VALIDATION_FAILED, "content must not contain NUL bytes")
    if len(content.encode("utf-8")) > ctx.max_file_bytes:
        return err_result(
            tool_name, VALIDATION_FAILED,
            f"content exceeds the {ctx.max_file_bytes}-byte per-file limit",
        )
    return None


def _validate_edit_rayjob_yaml(tool_name: str, args: dict, ctx: ToolContext) -> ToolResult | None:
    err = _check_names(args, ("run_id", "attempt_id"))
    if err:
        return err_result(tool_name, VALIDATION_FAILED, err)
    patch = args.get("patch")
    if isinstance(patch, str):
        return err_result(
            tool_name, VALIDATION_FAILED,
            "patch must be a structured object, not a YAML string; raw YAML edits are not accepted",
        )
    if not isinstance(patch, dict) or not patch:
        return err_result(tool_name, VALIDATION_FAILED, "patch must be a non-empty object")
    unknown = set(patch) - PATCHABLE_FIELDS
    if unknown:
        return err_result(
            tool_name, VALIDATION_FAILED,
            f"patch field(s) not in the allowlist: {sorted(unknown)}; "
            f"only {sorted(PATCHABLE_FIELDS)} may be changed",
        )
    if patch.get("entrypoint") is not None:
        err = validate_entrypoint(patch["entrypoint"])
        if err:
            return err_result(tool_name, VALIDATION_FAILED, err)
    if patch.get("num_workers") is not None:
        workers = patch["num_workers"]
        if not _is_int(workers) or workers < 1:
            return err_result(tool_name, VALIDATION_FAILED, "num_workers must be a positive int")
        if workers > ctx.max_num_workers:
            return err_result(
                tool_name, RESOURCE_LIMIT_EXCEEDED,
                f"num_workers {workers} exceeds the cap of {ctx.max_num_workers}",
            )
    if patch.get("cpu_per_worker") is not None:
        cpu = parse_cpu(patch["cpu_per_worker"])
        if cpu is None:
            return err_result(tool_name, VALIDATION_FAILED, f"unparseable cpu quantity: {patch['cpu_per_worker']!r}")
        if cpu > ctx.max_cpu_per_pod:
            return err_result(
                tool_name, RESOURCE_LIMIT_EXCEEDED,
                f"cpu_per_worker {cpu} exceeds the per-pod cap of {ctx.max_cpu_per_pod}",
            )
    if patch.get("memory_per_worker") is not None:
        memory = parse_memory(patch["memory_per_worker"])
        if memory is None:
            return err_result(tool_name, VALIDATION_FAILED, f"unparseable memory quantity: {patch['memory_per_worker']!r}")
        cap = parse_memory(ctx.max_memory_per_pod) or 0
        if memory > cap:
            return err_result(
                tool_name, RESOURCE_LIMIT_EXCEEDED,
                f"memory_per_worker {patch['memory_per_worker']!r} exceeds the per-pod cap of {ctx.max_memory_per_pod}",
            )
    if patch.get("env_vars") is not None:
        env_vars = patch["env_vars"]
        if not isinstance(env_vars, dict):
            return err_result(tool_name, VALIDATION_FAILED, "env_vars must be an object")
        for key, value in env_vars.items():
            if not isinstance(key, str) or not ENV_KEY_RE.match(key):
                return err_result(tool_name, VALIDATION_FAILED, f"env var name {key!r} must match ^[A-Z_][A-Z0-9_]*$")
            if not isinstance(value, str) or len(value) > 512 or any(c in SHELL_METACHARACTERS for c in value):
                return err_result(tool_name, VALIDATION_FAILED, f"env var {key!r} has an illegal value")
    if patch.get("image") is not None:
        err = check_image_approved(patch["image"], ctx)
        if err:
            return err_result(tool_name, IMAGE_NOT_APPROVED, err)
    return None


def _validate_apply_rayjob(tool_name: str, args: dict, ctx: ToolContext) -> ToolResult | None:
    err = _check_names(args, ("job_name", "run_id", "attempt_id"))
    if err:
        return err_result(tool_name, VALIDATION_FAILED, err)
    if not _run_already_applied(ctx, args["run_id"]):
        token = args.get("approval_token")
        if not approval_tokens.verify_token(
            ctx.secret, token, "first_launch", approval_tokens.apply_binding(args)
        ):
            return err_result(
                tool_name, APPROVAL_REQUIRED,
                "first launch of this run requires a valid approval token "
                "(scope 'first_launch', bound to run_id + job_name + spec_hash)",
            )
    return None


def _validate_build_image(tool_name: str, args: dict, ctx: ToolContext) -> ToolResult | None:
    err = _check_names(args, ("run_id", "attempt_id"))
    if err:
        return err_result(tool_name, VALIDATION_FAILED, err)
    err = check_image_approved(args.get("base_image"), ctx)
    if err:
        return err_result(tool_name, IMAGE_NOT_APPROVED, err)
    template = args.get("dockerfile_template", "ray-train-cpu")
    if template not in ALLOWED_DOCKERFILE_TEMPLATES:
        return err_result(
            tool_name, VALIDATION_FAILED,
            f"dockerfile_template must be one of {sorted(ALLOWED_DOCKERFILE_TEMPLATES)}; got {template!r}",
        )
    resolved, err = resolve_workspace_path(ctx, args["run_id"], args["attempt_id"], args.get("requirements_path"))
    if err:
        return err_result(tool_name, PATH_NOT_ALLOWED, err)
    assert resolved is not None
    if resolved.suffix != ".txt":
        return err_result(tool_name, PATH_NOT_ALLOWED, "requirements_path must point at a .txt file")
    token = args.get("approval_token")
    if not approval_tokens.verify_token(
        ctx.secret, token, "build_image", approval_tokens.build_binding(args)
    ):
        return err_result(
            tool_name, APPROVAL_REQUIRED,
            "every image build requires a valid approval token "
            "(scope 'build_image', bound to base_image + requirements_path)",
        )
    return None


def _validate_stop_delete(tool_name: str, args: dict, ctx: ToolContext) -> ToolResult | None:
    err = _check_names(args, ("job_name",))
    if err:
        return err_result(tool_name, VALIDATION_FAILED, err)
    if tool_name == "stop_job":
        reason = args.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return err_result(tool_name, VALIDATION_FAILED, "stop_job requires a non-empty reason (written to the audit log)")
    return None


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def validate_tool_call(tool_name: str, args: dict[str, Any], context: Any) -> ToolResult:
    """Validate one tool call against the sandbox rules.

    Returns a ToolResult with ok=True when the call may proceed, or
    ok=False with one of the design B-4 error codes when it may not.
    """
    ctx = coerce_context(context)
    if not isinstance(args, dict):
        return err_result(str(tool_name), VALIDATION_FAILED, "args must be an object")
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return err_result(
            str(tool_name), VALIDATION_FAILED,
            f"tool {tool_name!r} is not in the allowlist (the 13 registered tools)",
        )
    if spec["kind"] == "read":
        rejection = _validate_read(tool_name, args, ctx)
    elif tool_name == "write_workspace_file":
        rejection = _validate_write_workspace_file(tool_name, args, ctx)
    elif tool_name == "edit_rayjob_yaml":
        rejection = _validate_edit_rayjob_yaml(tool_name, args, ctx)
    elif tool_name == "apply_rayjob":
        rejection = _validate_apply_rayjob(tool_name, args, ctx)
    elif tool_name == "build_image":
        rejection = _validate_build_image(tool_name, args, ctx)
    else:  # stop_job, delete_job
        rejection = _validate_stop_delete(tool_name, args, ctx)
    if rejection is not None:
        return rejection
    return ok_result(tool_name, {"validated": True})
