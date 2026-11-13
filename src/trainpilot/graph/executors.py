"""Executors: the boundary between the agent graph and the training run.

An executor is any object with
``execute(spec, attempt_index, attempt_dir) -> dict`` returning the
protocol dict from docs/CONTRACTS.md (status, logs, metrics,
checkpoint_dir, duration_seconds). ``StubExecutor`` is the scripted
implementation used by tests and the offline demo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import TrainingSpec

_DEFAULT_SUCCESS_METRICS = {"eval_accuracy": 0.85}
_EXHAUSTED_LOG = "StubExecutor: no scripted outcome left for this attempt"


class StubExecutor:
    """Scripted executor: each call pops the next outcome dict.

    Outcome dicts use the executor protocol shape. On a succeeded
    outcome the stub materializes a minimal fake checkpoint directory
    (config.json, model.safetensors, metrics.json) so the verifier has
    something real to check; the directory defaults to
    ``<attempt_dir>/checkpoint-final`` unless the outcome names one.
    Calling execute with no outcomes left returns a failed outcome
    instead of raising, so a run degrades into a responsible give-up.
    """

    def __init__(self, outcomes: list[dict]):
        self._outcomes: list[dict[str, Any]] = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def execute(self, spec: TrainingSpec, attempt_index: int,
                attempt_dir: Path) -> dict:
        attempt_dir = Path(attempt_dir)
        self.calls.append({
            "attempt_index": attempt_index,
            "spec_hash": spec.spec_hash(),
            "attempt_dir": str(attempt_dir),
        })
        if self._outcomes:
            outcome = dict(self._outcomes.pop(0))
        else:
            outcome = {
                "status": "failed",
                "logs": _EXHAUSTED_LOG,
                "metrics": None,
                "checkpoint_dir": None,
                "duration_seconds": 0.0,
            }
        outcome.setdefault("logs", "")
        outcome.setdefault("metrics", None)
        outcome.setdefault("checkpoint_dir", None)
        outcome.setdefault("duration_seconds", 0.0)
        if outcome.get("status") == "succeeded":
            metrics = outcome.get("metrics") or dict(_DEFAULT_SUCCESS_METRICS)
            outcome["metrics"] = metrics
            checkpoint_dir = self._materialize_checkpoint(
                outcome.get("checkpoint_dir"), attempt_dir, metrics)
            outcome["checkpoint_dir"] = str(checkpoint_dir)
        return outcome

    @staticmethod
    def _materialize_checkpoint(checkpoint_dir: Any, attempt_dir: Path,
                                metrics: dict) -> Path:
        target = Path(checkpoint_dir) if checkpoint_dir else attempt_dir / "checkpoint-final"
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps({"model_type": "distilbert", "architectures": ["DistilBertForSequenceClassification"]}),
            encoding="utf-8",
        )
        (target / "model.safetensors").write_bytes(b"stub-checkpoint-weights")
        (target / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return target
