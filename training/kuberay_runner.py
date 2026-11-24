"""Opt-in local Kind executor for small offline RayJobs.

The Ray job schedules the trainer; the smoke and tiny-LM trainers themselves
do not implement distributed training. Resources remain in the cluster for
inspection. This executor deliberately targets only the local Kind context.
"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from training.local_runner import _spec_dict, parse_result_line

ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "trainpilot-jobs"
CONTEXT = "kind-trainpilot"


class KubeRayExecutor:
    def __init__(self, timeout_seconds: int = 240, poll_seconds: float = 2):
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds

    def _kubectl(self, *args: str, input_text: str | None = None) -> str:
        result = subprocess.run(
            ["kubectl", "--context", CONTEXT, "-n", NAMESPACE, *args],
            input=input_text, capture_output=True, text=True, timeout=30,
            check=True,
        )
        return result.stdout

    def execute(self, spec: Any, attempt_index: int, attempt_dir: Path) -> dict:
        started = time.monotonic()
        attempt_dir = Path(attempt_dir).resolve()
        attempt_dir.mkdir(parents=True, exist_ok=True)
        name = "tp-" + uuid.uuid4().hex[:12]
        output_dir = f"/trainpilot/workspaces/{name}"
        config = _spec_dict(spec)
        config.update(output_dir=output_dir, attempt=attempt_index, fault="none")
        config_map = {
            "apiVersion": "v1", "kind": "ConfigMap",
            "metadata": {"name": name, "namespace": NAMESPACE},
            "data": {"train_config.yaml": yaml.safe_dump(config)},
        }
        job = yaml.safe_load((ROOT / "infra/k8s/smoke-rayjob.yaml").read_text())
        job["metadata"]["name"] = name
        job["metadata"]["labels"].pop("trainpilot/spec-hash", None)
        job["spec"]["entrypoint"] = (
            "python /app/training/train.py --config /app/config/train_config.yaml --backend smoke"
        )
        job["spec"].pop("ttlSecondsAfterFinished", None)
        cluster = job["spec"]["rayClusterSpec"]
        workers = cluster["workerGroupSpecs"][0]
        count = int(config.get("num_workers", 2))
        if not 1 <= count <= 2:
            return {"status": "failed", "logs": "KubeRay local mode supports 1 or 2 CPU workers", "metrics": None, "checkpoint_dir": None, "duration_seconds": 0}
        workers.update(replicas=count, minReplicas=count, maxReplicas=count)
        head_spec = cluster["headGroupSpec"]["template"]["spec"]
        head_spec["volumes"].append({"name": "train-config", "configMap": {"name": name}})
        head_spec["containers"][0]["volumeMounts"].append(
            {"name": "train-config", "mountPath": "/app/config", "readOnly": True}
        )
        logs = ""
        try:
            self._kubectl("create", "-f", "-", input_text=yaml.safe_dump(config_map))
            self._kubectl("create", "-f", "-", input_text=yaml.safe_dump(job))
            deadline = time.monotonic() + self.timeout_seconds
            status = {}
            while time.monotonic() < deadline:
                status = json.loads(self._kubectl("get", "rayjob", name, "-o", "json")).get("status", {})
                if status.get("jobStatus") in ("SUCCEEDED", "FAILED", "STOPPED"):
                    break
                time.sleep(self.poll_seconds)
            else:
                raise TimeoutError(f"RayJob {name} did not finish in {self.timeout_seconds}s")
            logs = self._kubectl("logs", "job/" + name, "--tail=1000")
            result = parse_result_line(logs)
            if status.get("jobStatus") != "SUCCEEDED" or not result or result.get("status") != "succeeded":
                raise RuntimeError(f"RayJob {name} failed or did not emit a successful trainer result")
            head = status.get("rayClusterStatus", {}).get("head", {}).get("podName")
            if not head:
                raise RuntimeError(f"RayJob {name} has no head pod in status")
            checkpoint = attempt_dir / "checkpoint-final"
            self._kubectl("cp", f"{head}:{output_dir}/checkpoint-final", str(checkpoint))
            if not checkpoint.is_dir():
                raise RuntimeError("checkpoint was not copied from the Ray head")
            if result.get("eval_perplexity") is not None:
                metrics = {"eval_perplexity": result.get("eval_perplexity"), "baseline_perplexity": result.get("baseline_perplexity")}
            else:
                metrics = {"eval_accuracy": result.get("val_accuracy")}
            outcome = {"status": "succeeded", "logs": logs, "metrics": metrics,
                       "checkpoint_dir": str(checkpoint), "duration_seconds": time.monotonic() - started,
                       "rayjob_name": name}
            if result.get("samples") is not None:
                outcome["samples"] = result["samples"]
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError, TimeoutError) as exc:
            detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            logs += f"\nKubeRay execution failed ({name}): {detail}"
            outcome = {"status": "failed", "logs": logs, "metrics": None,
                       "checkpoint_dir": None, "duration_seconds": time.monotonic() - started,
                       "rayjob_name": name}
        (attempt_dir / "train.log").write_text(logs, encoding="utf-8")
        return outcome
