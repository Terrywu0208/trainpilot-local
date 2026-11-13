"""Benchmark generator invariants (design D.6/D.14): frozen counts,
difficulty buckets, safety composition, lab50 view, manifest hashes."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from eval.schema import row_to_taskspec, validate_task

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark"


def _load(name):
    return [json.loads(line) for line in
            (BENCH / name).read_text().splitlines() if line.strip()]


def test_datapoints_counts_and_buckets():
    rows = _load("datapoints.jsonl")
    assert len(rows) == 200
    assert Counter(r["difficulty"] for r in rows) == {
        "easy": 100, "hard": 60, "edge": 20, "unanswerable": 20}
    assert Counter(r["split"] for r in rows) == {"dev": 50, "heldout": 150}
    assert len({r["scenario_id"] for r in rows}) == 20
    for row in rows:
        assert validate_task(row) == [], row["id"]
        assert isinstance(row["expected_state"], dict) and row["expected_state"]
        assert row["expected"] or row["grading_type"] == "rubric"
        spec = row_to_taskspec(row)
        assert spec.id == row["id"]


def test_safety_suite_composition():
    rows = _load("safety_suite.jsonl")
    cats = Counter(r["safety_category"] for r in rows)
    assert len(rows) == 66
    for cat in ("prompt_injection", "memory_poisoning", "data_leakage",
                "harmful_request", "risky_action", "jailbreak"):
        assert cats[cat] == 10, cat
    assert cats["harmless_control"] == 6
    for row in rows:
        assert validate_task(row) == [], row["id"]
        if row["safety_category"] == "risky_action":
            assert row["safe_behavior"] == "ask_confirmation"
        elif row["safety_category"] == "harmless_control":
            assert row["safe_behavior"] == "answer_without_harm"
        else:
            assert row["safe_behavior"] == "refuse"


def test_lab50_composition():
    rows = _load("tasks_lab50.jsonl")
    assert len(rows) == 50
    assert {r["grading_type"] for r in rows} == {
        "exact", "tool", "rubric", "unanswerable"}
    assert sum(1 for r in rows if r["difficulty"] == "edge") >= 5
    assert sum(1 for r in rows if r["difficulty"] == "unanswerable") >= 4
    for row in rows:
        # The professor's tasks.jsonl field shape.
        for field in ("id", "prompt", "expected", "rubric", "notes"):
            assert field in row, (row["id"], field)
        assert validate_task(row) == [], row["id"]


def test_manifest_hashes_verify():
    manifest = json.loads((BENCH / "manifest.json").read_text())
    assert manifest["version"] == "benchmark-freeze-v1"
    assert manifest["counts"]["datapoints"] == 200
    assert manifest["counts"]["safety_prompts"] == 60
    assert manifest["counts"]["harmless_controls"] == 6
    assert manifest["counts"]["lab50"] == 50
    for rel, digest in manifest["files"].items():
        actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert actual == digest, rel
    thresholds = ROOT / manifest["grader_thresholds"]["path"]
    assert hashlib.sha256(thresholds.read_bytes()).hexdigest() == \
        manifest["grader_thresholds"]["sha256"]
