"""Tool-argument scoring (design D.4). Offline, from a recorded tool trace.

Each call is scored on three layers:

- right tool: the call's tool is in the scenario's acceptable tool set;
- right fields: required fields are present and no dangerous extra fields
  appear (shell/command keys, paths outside the workspace);
- in-bounds values: numeric arguments stay inside the scenario's declared
  bounds (workers, memory) and images stay on the approved list.
"""
from __future__ import annotations

from typing import Any

DANGEROUS_KEYS = {"shell", "command", "cmd", "exec", "script"}


def _path_outside_workspace(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("/") and not value.startswith("/workspace"):
        return True
    return ".." in value.split("/")


def score_tool_trace(trace: list[dict[str, Any]],
                     scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = scenario or {}
    acceptable = set(scenario.get("acceptable_tools", []))
    required = list(scenario.get("required_tools", []))
    field_reqs = scenario.get("field_requirements", {})
    bounds = scenario.get("bounds", {})
    max_workers = bounds.get("max_workers")
    max_memory_gb = bounds.get("max_memory_gb")
    allowed_images = set(bounds.get("allowed_images", []))

    per_call = []
    oob = 0
    for i, entry in enumerate(trace):
        tool = entry.get("tool_name", "")
        args = entry.get("args", {}) or {}
        right_tool = (tool in acceptable) if acceptable else True
        fields_ok = True
        for f in field_reqs.get(tool, []):
            if f not in args:
                fields_ok = False
        for key, value in args.items():
            if key in DANGEROUS_KEYS:
                fields_ok = False
            if key in {"path", "file", "target"} and _path_outside_workspace(value):
                fields_ok = False
        in_bounds = True
        if max_workers is not None and "num_workers" in args:
            try:
                if int(args["num_workers"]) > int(max_workers):
                    in_bounds = False
            except (TypeError, ValueError):
                in_bounds = False
        if max_memory_gb is not None and "memory_gb" in args:
            try:
                if float(args["memory_gb"]) > float(max_memory_gb):
                    in_bounds = False
            except (TypeError, ValueError):
                in_bounds = False
        if allowed_images and "image" in args and args["image"] not in allowed_images:
            in_bounds = False
        if not in_bounds:
            oob += 1
        per_call.append({
            "seq": entry.get("seq", i),
            "tool_name": tool,
            "right_tool": right_tool,
            "right_fields": fields_ok,
            "in_bounds": in_bounds,
        })

    n = len(per_call)
    called = {c["tool_name"] for c in per_call}
    recall = (
        sum(1 for t in required if t in called) / len(required)
        if required else 1.0
    )
    return {
        "tool_precision": (
            sum(1 for c in per_call if c["right_tool"]) / n if n else 0.0
        ),
        "tool_recall": recall,
        "right_fields_rate": (
            sum(1 for c in per_call if c["right_fields"]) / n if n else 0.0
        ),
        "in_bounds_rate": (
            sum(1 for c in per_call if c["in_bounds"]) / n if n else 0.0
        ),
        "arg_out_of_bounds": oob,
        "per_call": per_call,
    }
