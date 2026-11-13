"""Registry completeness: exactly the 13 tools of design B-4."""
from __future__ import annotations

from trainpilot.tools.registry import TOOL_REGISTRY

EXPECTED_READ = {
    "get_machine_resources",
    "get_pod_status",
    "describe_pod",
    "get_rayjob_status",
    "get_rayjob_logs",
    "get_metrics",
    "list_checkpoints",
}
EXPECTED_ACTION = {
    "write_workspace_file",
    "edit_rayjob_yaml",
    "apply_rayjob",
    "build_image",
    "stop_job",
    "delete_job",
}


def test_registry_has_exactly_13_tools():
    assert len(TOOL_REGISTRY) == 13
    assert set(TOOL_REGISTRY) == EXPECTED_READ | EXPECTED_ACTION


def test_registry_kinds_are_correct():
    for name in EXPECTED_READ:
        assert TOOL_REGISTRY[name]["kind"] == "read", name
        assert TOOL_REGISTRY[name]["server"] == "read", name
    for name in EXPECTED_ACTION:
        assert TOOL_REGISTRY[name]["kind"] == "action", name
        assert TOOL_REGISTRY[name]["server"] == "action", name


def test_registry_specs_are_complete():
    for name, spec in TOOL_REGISTRY.items():
        assert spec["name"] == name
        assert isinstance(spec["description"], str) and spec["description"]
        assert isinstance(spec["dangerous_args"], list)
        assert isinstance(spec["timeout_seconds"], int) and spec["timeout_seconds"] > 0
        assert spec["risk_tier"] in ("read", "write", "destructive", "build")


def test_registry_approval_gates():
    assert TOOL_REGISTRY["apply_rayjob"]["approval_required"] == "first_launch"
    assert TOOL_REGISTRY["build_image"]["approval_required"] == "build_image"
    for name in EXPECTED_READ:
        assert TOOL_REGISTRY[name]["approval_required"] is False


def test_purge_is_not_in_the_registry():
    # Deliberate absence (design B-4): purge lives in trainpilot.tools.purge.
    assert "purge_attempt_artifacts" not in TOOL_REGISTRY
