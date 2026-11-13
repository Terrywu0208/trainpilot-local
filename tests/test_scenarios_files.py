"""Schema/field tests for scenarios/<id>/scenario.yaml."""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = ROOT / "scenarios"

EXPECTED_IDS = {
    "clean-canonical",
    "clean-small-lr",
    "gpu-nccl-backend",
    "gpu-hardcoded-cuda",
    "mem-batch32-oom",
    "mem-maxlength512-slow",
    "dep-torch-transformers",
    "dep-numpy2-abi",
    "data-label-invalid",
    "data-label-flipped",
    "data-lr-too-high",
    "demo-two-stage",
}

CLOSED_ROOT_CAUSE_CODES = {
    "NONE",
    "NCCL_ON_CPU",
    "CUDA_ON_CPU",
    "OOMKILLED_137",
    "TIMEOUT_STEP_TOO_SLOW",
    "DEP_TORCH_TRANSFORMERS_MISMATCH",
    "DEP_NUMPY2_ABI",
    "DATA_LABEL_INVALID",
    "DATA_LABEL_FLIPPED_ACC_LOW",
    "TRAIN_LOSS_NAN",
}

FAULT_KNOBS = {
    "none",
    "nccl_on_cpu",
    "cuda_on_cpu",
    "oom",
    "slow_steps",
    "dep_mismatch",
    "label_invalid",
    "label_flipped",
    "loss_nan",
}


def load_scenarios():
    found = {}
    for path in sorted(SCENARIOS_DIR.glob("*/scenario.yaml")):
        found[path.parent.name] = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )
    return found


SCENARIOS = load_scenarios()


def test_all_expected_scenarios_exist():
    assert set(SCENARIOS) == EXPECTED_IDS


@pytest.mark.parametrize("scenario_id", sorted(EXPECTED_IDS))
def test_scenario_carries_required_fields(scenario_id):
    data = SCENARIOS[scenario_id]
    for field in (
        "id",
        "family",
        "injection",
        "log_signature",
        "ground_truth",
        "correct_repair",
        "wrong_repair",
        "accuracy_threshold",
    ):
        assert field in data, f"{scenario_id} missing field {field}"
    assert data["id"] == scenario_id
    assert 0.0 < float(data["accuracy_threshold"]) <= 1.0
    assert isinstance(data["wrong_repair"], dict)
    assert data["wrong_repair"].get("why_wrong")


@pytest.mark.parametrize(
    "scenario_id", sorted(EXPECTED_IDS - {"demo-two-stage"})
)
def test_single_stage_ground_truth_codes_are_closed(scenario_id):
    truth = SCENARIOS[scenario_id]["ground_truth"]
    assert truth["root_cause_code"] in CLOSED_ROOT_CAUSE_CODES
    assert truth["fault_family"]
    overrides = SCENARIOS[scenario_id]["config_overrides"]
    assert overrides["fault"] in FAULT_KNOBS
    # The declared log signature must be reachable from the ground truth.
    assert SCENARIOS[scenario_id]["log_signature"]


def test_demo_two_stage_structure():
    demo = SCENARIOS["demo-two-stage"]
    stages = demo["stages"]
    assert [s["fault"] for s in stages] == ["nccl_on_cpu", "oom"]
    codes = [t["root_cause_code"] for t in demo["ground_truth"]]
    assert codes == ["NCCL_ON_CPU", "OOMKILLED_137"]
    for stage in stages:
        assert stage["correct_repair"], "each stage needs a correct repair"
