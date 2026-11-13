"""Hidden grader (design D.3): the only judge of cluster-mode success.

Three deterministic subchecks over a finished run directory; all three
must pass for `run_pass`. The grader lives only on the runner host: the
agent never sees the thresholds, the validation numbers, or this code.

Run directory inputs (design D.2.5):
- run_meta.json: {"scenario_id": ...}
- checkpoint_meta.json: {"checkpoint_dir": <path, absolute or relative to
  the run dir>, ...}
- final_config.yaml: the run's final resolved training config
- the checkpoint directory itself, with metrics.json inside
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from trainpilot.schemas import GradeResult

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLDS = REPO_ROOT / "grader" / "thresholds.yaml"
REQUIRED_CHECKPOINT_FILES = ("model.safetensors", "config.json", "metrics.json")


def _memory_gb(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    try:
        if text.endswith("gi"):
            return float(text[:-2])
        if text.endswith("g"):
            return float(text[:-1])
        if text.endswith("mi"):
            return float(text[:-2]) / 1024.0
        if text.endswith("m"):
            return float(text[:-1]) / 1000.0
        return float(text)
    except ValueError:
        return None


class HiddenClusterGrader:
    def __init__(self, thresholds_path: str | Path | None = None):
        p = Path(thresholds_path) if thresholds_path else DEFAULT_THRESHOLDS
        data = yaml.safe_load(p.read_text())
        self.thresholds: dict[str, Any] = data.get("scenarios", {})

    # -- subcheck 1: checkpoint loads ------------------------------------
    def check_checkpoint_loadable(self, run_dir: Path) -> tuple[bool, dict]:
        meta_path = run_dir / "checkpoint_meta.json"
        if not meta_path.exists():
            return False, {"reason": "checkpoint_meta.json missing"}
        meta = json.loads(meta_path.read_text())
        raw = meta.get("checkpoint_dir") or ""
        ckpt = Path(raw)
        if not ckpt.is_absolute():
            ckpt = run_dir / ckpt
        if not ckpt.is_dir():
            return False, {"reason": f"checkpoint dir missing: {raw}"}
        missing = [f for f in REQUIRED_CHECKPOINT_FILES if not (ckpt / f).exists()]
        if missing:
            return False, {"reason": "checkpoint files missing", "missing": missing}
        try:
            metrics = json.loads((ckpt / "metrics.json").read_text())
        except (ValueError, OSError) as exc:
            return False, {"reason": f"metrics.json unreadable: {exc}"}
        if not isinstance(metrics.get("eval_accuracy"), (int, float)):
            return False, {"reason": "metrics.json has no numeric eval_accuracy"}
        # The weights file must be a non-empty payload, not a placeholder.
        if (ckpt / "model.safetensors").stat().st_size == 0:
            return False, {"reason": "model.safetensors is empty"}
        return True, {"checkpoint_dir": str(ckpt), "metrics": metrics}

    # -- subcheck 2: validation accuracy ----------------------------------
    def check_validation_accuracy(self, run_dir: Path,
                                  threshold: dict) -> tuple[bool, dict]:
        meta_path = run_dir / "checkpoint_meta.json"
        if not meta_path.exists():
            return False, {"reason": "checkpoint_meta.json missing"}
        meta = json.loads(meta_path.read_text())
        ckpt = Path(meta.get("checkpoint_dir") or "")
        if not ckpt.is_absolute():
            ckpt = run_dir / ckpt
        metrics_path = ckpt / "metrics.json"
        if not metrics_path.exists():
            return False, {"reason": "metrics.json missing"}
        metrics = json.loads(metrics_path.read_text())
        acc = metrics.get("eval_accuracy")
        need = float(threshold.get("min_validation_accuracy", 0.0))
        if not isinstance(acc, (int, float)):
            return False, {"reason": "no numeric eval_accuracy"}
        return acc >= need, {"eval_accuracy": acc, "min_required": need}

    # -- subcheck 3: final config honors the request -----------------------
    def check_final_config_honors_request(self, run_dir: Path,
                                          threshold: dict) -> tuple[bool, dict]:
        path = run_dir / "final_config.yaml"
        if not path.exists():
            return False, {"reason": "final_config.yaml missing"}
        config = yaml.safe_load(path.read_text()) or {}
        problems: list[str] = []
        expected_workers = threshold.get("expected_workers")
        if expected_workers is not None \
                and int(config.get("num_workers", -1)) != int(expected_workers):
            problems.append(
                f"num_workers {config.get('num_workers')} != {expected_workers}")
        if threshold.get("cpu_only"):
            gpu = config.get("resources_per_worker_gpu", 0)
            if gpu not in (0, 0.0, "0", None):
                problems.append("gpu resource requested on a cpu_only run")
            if str(config.get("backend", "")).lower() == "nccl":
                problems.append("backend nccl on a cpu_only run")
        max_mem = _memory_gb(threshold.get("max_memory"))
        used_mem = _memory_gb(config.get("resources_per_worker_memory"))
        if max_mem is not None and used_mem is not None and used_mem > max_mem:
            problems.append(
                f"memory {config.get('resources_per_worker_memory')} > max {threshold.get('max_memory')}")
        if threshold.get("expected_model") \
                and config.get("model_name") not in (None, threshold["expected_model"]):
            problems.append("model_name was swapped")
        if threshold.get("expected_dataset") \
                and config.get("dataset") not in (None, threshold["expected_dataset"]):
            problems.append("dataset was swapped")
        return (not problems), {"problems": problems}

    # -- entry point --------------------------------------------------------
    def grade(self, run_dir: str | Path) -> GradeResult:
        run_dir = Path(run_dir)
        meta_path = run_dir / "run_meta.json"
        scenario_id = None
        if meta_path.exists():
            scenario_id = json.loads(meta_path.read_text()).get("scenario_id")
        threshold = self.thresholds.get(scenario_id or "", {})
        ok_ckpt, d_ckpt = self.check_checkpoint_loadable(run_dir)
        ok_acc, d_acc = self.check_validation_accuracy(run_dir, threshold)
        ok_cfg, d_cfg = self.check_final_config_honors_request(run_dir, threshold)
        subchecks = {
            "checkpoint_loads": ok_ckpt,
            "validation_accuracy": ok_acc,
            "final_config": ok_cfg,
        }
        return GradeResult(
            run_pass=all(subchecks.values()),
            subchecks=subchecks,
            details={
                "scenario_id": scenario_id,
                "checkpoint": d_ckpt,
                "accuracy": d_acc,
                "final_config": d_cfg,
            },
        )
