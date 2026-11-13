"""Local executor for TrainPilot-Local (section C).

`LocalExecutor` implements the shared executor protocol (docs/CONTRACTS.md)
by running `training/train.py` as a subprocess inside the attempt directory.
The default engine is the dependency-free smoke trainer, so the full
plan -> execute -> diagnose -> repair loop works offline at zero cost.

`render_rayjob_yaml` renders the KubeRay RayJob manifest for a spec with
Jinja2; the output is plain YAML that `yaml.safe_load` can parse.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml
from jinja2 import Template

TRAIN_PY = Path(__file__).resolve().parent / "train.py"
RESULT_PREFIX = "TRAINPILOT_RESULT "

_RAYJOB_TEMPLATE = Template(
    """apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: trainpilot-{{ name_suffix }}
  labels:
    app: trainpilot-local
    trainpilot/spec-hash: "{{ spec_hash }}"
spec:
  entrypoint: {{ entrypoint | tojson }}
  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 300
  rayClusterSpec:
    rayVersion: "2.41.0"
    headGroupSpec:
      rayStartParams:
        dashboard-host: "0.0.0.0"
      template:
        spec:
          containers:
            - name: ray-head
              image: {{ image | tojson }}
              resources:
                requests:
                  cpu: "1"
                  memory: "1Gi"
                limits:
                  cpu: "1"
                  memory: "1Gi"
    workerGroupSpecs:
      - groupName: train-workers
        replicas: {{ spec.num_workers }}
        minReplicas: {{ spec.num_workers }}
        maxReplicas: {{ spec.num_workers }}
        rayStartParams:
          num-cpus: "2"
        template:
          spec:
            containers:
              - name: ray-worker
                image: {{ image | tojson }}
                env:
                  - name: TRAINPILOT_BACKEND
                    value: {{ spec.backend | tojson }}
                  - name: TRAINPILOT_MODEL
                    value: {{ spec.model_name | tojson }}
                  - name: TRAINPILOT_DATASET
                    value: {{ spec.dataset | tojson }}
                  - name: TRAINPILOT_PER_WORKER_BATCH_SIZE
                    value: "{{ spec.per_worker_batch_size }}"
                  - name: TRAINPILOT_GRADIENT_ACCUMULATION_STEPS
                    value: "{{ spec.gradient_accumulation_steps }}"
                  - name: TRAINPILOT_MAX_STEPS
                    value: "{{ spec.max_steps }}"
                  - name: TRAINPILOT_LEARNING_RATE
                    value: "{{ spec.learning_rate }}"
                  - name: TRAINPILOT_MAX_LENGTH
                    value: "{{ spec.max_length }}"
                  - name: TRAINPILOT_SEED
                    value: "{{ spec.seed }}"
                resources:
                  requests:
                    cpu: "2"
                    memory: {{ spec.resources_per_worker_memory | tojson }}
                  limits:
                    cpu: "2"
                    memory: {{ spec.resources_per_worker_memory | tojson }}
"""
)


def _spec_dict(spec: Any) -> dict:
    if hasattr(spec, "model_dump"):
        return dict(spec.model_dump())
    if isinstance(spec, dict):
        return dict(spec)
    raise TypeError("spec must be a TrainingSpec or a dict")


def render_rayjob_yaml(
    spec: Any,
    image: str = "trainpilot-train:cpu-local",
    command: Optional[str] = None,
    scenario: Optional[str] = None,
) -> str:
    """Render a KubeRay RayJob YAML manifest for a TrainingSpec.

    `image` is the training image, `command` overrides the entrypoint,
    and `scenario` (optional) is folded into the job name suffix.
    """
    spec_data = _spec_dict(spec)
    if hasattr(spec, "spec_hash"):
        spec_hash = spec.spec_hash()
    else:
        import hashlib

        payload = json.dumps(spec_data, sort_keys=True, default=str)
        spec_hash = hashlib.sha256(payload.encode()).hexdigest()[:12]
    entrypoint = command or (
        "python /app/train.py --config "
        "/app/configs/train_config.yaml --backend torch"
    )
    name_suffix = spec_hash
    if scenario:
        name_suffix = f"{scenario}-{spec_hash}"
    return _RAYJOB_TEMPLATE.render(
        spec=spec_data,
        spec_hash=spec_hash,
        image=image,
        entrypoint=entrypoint,
        name_suffix=name_suffix,
    )


def parse_result_line(logs: str) -> Optional[dict]:
    """Extract the TRAINPILOT_RESULT payload from trainer output."""
    result = None
    for line in logs.splitlines():
        if line.startswith(RESULT_PREFIX):
            try:
                result = json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                continue
    return result


class LocalExecutor:
    """Executor-protocol implementation backed by a local subprocess.

    Extra scenario controls that are not part of TrainingSpec (the fault
    knob and repair flags) are injected via the constructor; they are
    merged into the resolved config written for each attempt.
    """

    def __init__(
        self,
        backend: str = "smoke",
        fault: str = "none",
        repair_flags: Optional[dict] = None,
        scenario_id: Optional[str] = None,
        extra_config: Optional[dict] = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.backend = backend
        self.fault = fault
        self.repair_flags = dict(repair_flags or {})
        self.scenario_id = scenario_id
        self.extra_config = dict(extra_config or {})
        self.timeout_seconds = timeout_seconds

    def execute(self, spec: Any, attempt_index: int,
                attempt_dir: Path) -> dict:
        attempt_dir = Path(attempt_dir)
        attempt_dir.mkdir(parents=True, exist_ok=True)

        config = _spec_dict(spec)
        config.update(self.extra_config)
        config["fault"] = self.fault
        flags = dict(config.get("repair_flags") or {})
        flags.update(self.repair_flags)
        config["repair_flags"] = flags
        config["attempt"] = attempt_index
        if self.scenario_id:
            config["scenario_id"] = self.scenario_id
        config["output_dir"] = str(attempt_dir)

        # TrainingSpec uses train/val subsample names; train.py config
        # uses the design knob names dataset_subsample_* as aliases.
        config.setdefault(
            "dataset_subsample_train", config.get("train_subsample", 1024)
        )
        config.setdefault(
            "dataset_subsample_val", config.get("val_subsample", 256)
        )

        config_path = attempt_dir / "train_config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
        )

        started = time.monotonic()
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_PY),
                    "--config",
                    str(config_path),
                    "--backend",
                    self.backend,
                ],
                cwd=str(attempt_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            logs = (proc.stdout or "") + (proc.stderr or "")
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            logs = (exc.stdout or b"")
            if isinstance(logs, bytes):
                logs = logs.decode("utf-8", "replace")
            logs += "\nTimeoutError: local executor time budget exceeded"
            returncode = -1
        duration = time.monotonic() - started

        (attempt_dir / "train.log").write_text(logs, encoding="utf-8")

        result = parse_result_line(logs)
        checkpoint_dir = attempt_dir / "checkpoint-final"
        succeeded = (
            result is not None
            and result.get("status") == "succeeded"
            and returncode == 0
        )
        metrics = None
        checkpoint = None
        if succeeded:
            accuracy = result.get("val_accuracy")
            metrics = {"eval_accuracy": accuracy}
            if checkpoint_dir.is_dir():
                checkpoint = str(checkpoint_dir)
        return {
            "status": "succeeded" if succeeded else "failed",
            "logs": logs,
            "metrics": metrics,
            "checkpoint_dir": checkpoint,
            "duration_seconds": duration,
        }
