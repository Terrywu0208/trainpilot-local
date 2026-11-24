import json
from pathlib import Path

import yaml

from training.kuberay_runner import KubeRayExecutor


def test_kuberay_executor_passes_spec_and_copies_checkpoint(tmp_path, monkeypatch):
    calls = []

    def fake_kubectl(*args, input_text=None):
        calls.append((args, input_text))
        if args[:2] == ("get", "rayjob"):
            return json.dumps({"status": {"jobStatus": "SUCCEEDED", "rayClusterStatus": {"head": {"podName": "head"}}}})
        if args[0] == "logs":
            return 'TRAINPILOT_RESULT {"status":"succeeded","eval_perplexity":2,"baseline_perplexity":4}\n'
        if args[0] == "cp":
            Path(args[-1]).mkdir()
        return ""

    executor = KubeRayExecutor(poll_seconds=0)
    monkeypatch.setattr(executor, "_kubectl", fake_kubectl)
    result = executor.execute({"task_type": "pretrain", "dataset": "tinytales", "num_workers": 2, "epochs": 3}, 0, tmp_path)
    configmap = yaml.safe_load(calls[0][1])
    config = yaml.safe_load(configmap["data"]["train_config.yaml"])
    job = yaml.safe_load(calls[1][1])
    assert config["dataset"] == "tinytales" and config["epochs"] == 3
    assert job["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["replicas"] == 2
    assert result["status"] == "succeeded" and result["metrics"]["eval_perplexity"] == 2
    assert Path(result["checkpoint_dir"]).is_dir()


def test_kuberay_failure_never_reports_success(tmp_path, monkeypatch):
    executor = KubeRayExecutor(poll_seconds=0)

    def fake_kubectl(*args, **kwargs):
        if args[:2] == ("get", "rayjob"):
            return json.dumps({"status": {"jobStatus": "FAILED"}})
        return "trainer failed" if args[0] == "logs" else ""

    monkeypatch.setattr(executor, "_kubectl", fake_kubectl)
    result = executor.execute({"num_workers": 1}, 0, tmp_path)
    assert result["status"] == "failed"
    assert result["checkpoint_dir"] is None and "trainer failed" in result["logs"]
