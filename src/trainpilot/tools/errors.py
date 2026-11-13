"""Unified error codes and ToolResult helpers for the tool layer.

Error codes are the closed set from design section B-4. Every tool call
returns a ``trainpilot.schemas.ToolResult``; the MCP servers never leak a
raw Python exception to a caller (tracebacks go to the server log only).
"""
from __future__ import annotations

from trainpilot.schemas import ToolResult

# --- Closed error-code set (design B-4) -----------------------------------
NOT_FOUND = "NOT_FOUND"
VALIDATION_FAILED = "VALIDATION_FAILED"
APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
PATH_NOT_ALLOWED = "PATH_NOT_ALLOWED"
IMAGE_NOT_APPROVED = "IMAGE_NOT_APPROVED"
CLUSTER_UNREACHABLE = "CLUSTER_UNREACHABLE"
K8S_API_ERROR = "K8S_API_ERROR"
TIMEOUT = "TIMEOUT"
CLEANUP_INCOMPLETE = "CLEANUP_INCOMPLETE"
CONFLICT = "CONFLICT"

ERROR_CODES = {
    NOT_FOUND,
    VALIDATION_FAILED,
    APPROVAL_REQUIRED,
    RESOURCE_LIMIT_EXCEEDED,
    PATH_NOT_ALLOWED,
    IMAGE_NOT_APPROVED,
    CLUSTER_UNREACHABLE,
    K8S_API_ERROR,
    TIMEOUT,
    CLEANUP_INCOMPLETE,
    CONFLICT,
}


def ok_result(tool_name: str, data: dict | None = None) -> ToolResult:
    return ToolResult(ok=True, tool_name=tool_name, data=data or {})


def err_result(tool_name: str, error_code: str, error_message: str) -> ToolResult:
    assert error_code in ERROR_CODES, f"unknown error code: {error_code}"
    return ToolResult(
        ok=False,
        tool_name=tool_name,
        data={},
        error_code=error_code,
        error_message=error_message,
    )
