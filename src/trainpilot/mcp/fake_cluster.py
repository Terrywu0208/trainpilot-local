"""FakeClusterClient: an in-memory stand-in for Kubernetes + KubeRay.

There is no docker/kind/kubectl in this environment, and the real MCP
servers would talk to the cluster through the ``kubernetes`` Python
client. This fake keeps just enough state (RayJobs, pods, logs, metrics,
built images, applied runs) for the read/action servers and their tests
to exercise the full tool contract offline. Its method shapes mirror
what the real client wrappers will expose, so swapping the fake for the
real client does not change the servers.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_JOB_STATUSES = {"SUCCEEDED", "FAILED", "STOPPED"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FakeClusterClient:
    def __init__(self, workspace_root: str | Path | None = None):
        self.workspace_root = Path(workspace_root) if workspace_root else None
        self.rayjobs: dict[str, dict[str, Any]] = {}
        self.pods: dict[str, dict[str, Any]] = {}
        self.logs: dict[str, list[dict[str, Any]]] = {}
        self.built_images: dict[str, str] = {}  # cache key -> digest
        self.applied_runs: set[str] = set()

    # ------------------------------------------------------------- rayjobs
    def create_rayjob(
        self,
        job_name: str,
        *,
        run_id: str = "",
        attempt_id: str = "",
        spec_hash: str = "",
        num_workers: int = 2,
    ) -> dict[str, Any]:
        cluster_name = f"{job_name}-cluster"
        job = {
            "job_name": job_name,
            "uid": str(uuid.uuid5(uuid.NAMESPACE_DNS, job_name)),
            "resource_version": "1",
            "job_status": "RUNNING",
            "job_deployment_status": "Running",
            "ray_cluster_name": cluster_name,
            "start_time": _now_iso(),
            "end_time": None,
            "message": None,
            "failed_attempts": 0,
            "spec_hash": spec_hash,
            "run_id": run_id,
            "attempt_id": attempt_id,
        }
        self.rayjobs[job_name] = job
        self.applied_runs.add(run_id)
        self._spawn_pods(job_name, cluster_name, num_workers)
        self.logs[job_name] = [
            {
                "pod_name": f"{job_name}-submitter-0",
                "container": "submitter",
                "content": (
                    "fake-cluster: entrypoint started\n"
                    "fake-cluster: python train.py --config config.yaml\n"
                ),
            },
            {
                "pod_name": f"{cluster_name}-head-0",
                "container": "head",
                "content": "fake-cluster: ray head started, dashboard on :8265\n",
            },
            {
                "pod_name": f"{cluster_name}-worker-0",
                "container": "worker",
                "content": "fake-cluster: worker registered with head\n",
            },
        ]
        return job

    def get_rayjob(self, job_name: str) -> dict[str, Any] | None:
        return self.rayjobs.get(job_name)

    def delete_rayjob(self, job_name: str) -> bool:
        existed = self.rayjobs.pop(job_name, None) is not None
        for pod_name in [n for n, p in self.pods.items() if p["job_name"] == job_name]:
            del self.pods[pod_name]
        self.logs.pop(job_name, None)
        return existed

    def suspend_rayjob(self, job_name: str) -> str | None:
        job = self.rayjobs.get(job_name)
        if job is None:
            return None
        previous = job["job_status"]
        job["job_status"] = "STOPPED"
        job["job_deployment_status"] = "Suspended"
        return previous

    def finish_job(self, job_name: str, success: bool = True) -> None:
        """Test/demo helper: move a job to a terminal state."""
        job = self.rayjobs[job_name]
        job["job_status"] = "SUCCEEDED" if success else "FAILED"
        job["job_deployment_status"] = "Complete" if success else "Failed"
        job["end_time"] = _now_iso()
        phase = "Succeeded" if success else "Failed"
        for pod in self.pods.values():
            if pod["job_name"] == job_name:
                pod["phase"] = phase

    # ---------------------------------------------------------------- pods
    def _spawn_pods(self, job_name: str, cluster_name: str, num_workers: int) -> None:
        def pod(name: str, role: str, node: str) -> dict[str, Any]:
            return {
                "name": name,
                "job_name": job_name,
                "phase": "Running",
                "role": role,
                "node_name": node,
                "restarts": 0,
                "ready": True,
                "created_at": _now_iso(),
                "conditions": [
                    {"type": "Ready", "status": "True", "reason": "", "message": ""}
                ],
                "events": [],
                "container_statuses": [
                    {
                        "name": role,
                        "ready": True,
                        "restart_count": 0,
                        "state_reason": None,
                        "state_message": None,
                    }
                ],
                "resource_requests": {"cpu": "500m", "memory": "512Mi"},
                "resource_limits": {"cpu": "2", "memory": "3Gi"},
            }

        self.pods[f"{job_name}-submitter-0"] = pod(
            f"{job_name}-submitter-0", "submitter", "trainpilot-worker"
        )
        self.pods[f"{cluster_name}-head-0"] = pod(
            f"{cluster_name}-head-0", "head", "trainpilot-worker"
        )
        for i in range(num_workers):
            name = f"{cluster_name}-worker-{i}"
            self.pods[name] = pod(name, "worker", f"trainpilot-worker{(i % 2) + 1}")

    def list_pods(self, job_name: str) -> list[dict[str, Any]]:
        return [p for p in self.pods.values() if p["job_name"] == job_name]

    def get_pod(self, pod_name: str) -> dict[str, Any] | None:
        return self.pods.get(pod_name)

    # ---------------------------------------------------------------- misc
    def get_logs(
        self, job_name: str, container: str | None = None, tail_lines: int = 500
    ) -> list[dict[str, Any]]:
        sources = []
        for entry in self.logs.get(job_name, []):
            if container is not None and entry["container"] != container:
                continue
            lines = entry["content"].splitlines()
            kept = lines[-tail_lines:]
            sources.append(
                {
                    "pod_name": entry["pod_name"],
                    "container": entry["container"],
                    "lines_returned": len(kept),
                    "truncated": len(kept) < len(lines),
                    "content": "\n".join(kept),
                }
            )
        return sources

    def get_metrics(self, job_name: str) -> dict[str, Any] | None:
        if job_name not in self.rayjobs:
            return None
        pods = [
            {
                "pod_name": p["name"],
                "cpu_usage_cores": 0.25,
                "memory_usage_bytes": 256 * 1024 * 1024,
            }
            for p in self.list_pods(job_name)
        ]
        return {
            "job_name": job_name,
            "pods": pods,
            "ray": {
                "num_running_tasks": 2,
                "object_store_used_bytes": 1024,
                "object_store_available_bytes": 1024 * 1024,
            },
            "source": "fake-cluster",
            "sampled_at": _now_iso(),
        }

    def machine_resources(self) -> dict[str, Any]:
        page_size = os.sysconf("SC_PAGE_SIZE")
        mem_total = page_size * os.sysconf("SC_PHYS_PAGES")
        mem_avail = page_size * os.sysconf("SC_AVPHYS_PAGES")
        anchor = self.workspace_root or Path.cwd()
        disk = shutil.disk_usage(anchor if anchor.exists() else Path.cwd())
        nodes = [
            {
                "name": name,
                "cpu_allocatable": "6",
                "memory_allocatable": "10Gi",
                "cpu_capacity": "6",
                "memory_capacity": "10Gi",
                "ready": True,
            }
            for name in ("trainpilot-control-plane", "trainpilot-worker", "trainpilot-worker2")
        ]
        return {
            "host": {
                "cpu_count": os.cpu_count() or 0,
                "memory_total_bytes": mem_total,
                "memory_available_bytes": mem_avail,
                "disk_free_bytes": disk.free,
            },
            "nodes": nodes,
            "quota": {
                "namespace": "trainpilot-jobs",
                "cpu_used": "0",
                "cpu_hard": "4",
                "memory_used": "0",
                "memory_hard": "6Gi",
            },
        }

    # --------------------------------------------------------------- images
    def build_image_cached(self, cache_key: str) -> str | None:
        return self.built_images.get(cache_key)

    def record_built_image(self, cache_key: str, material: str) -> str:
        digest = "sha256:" + hashlib.sha256(material.encode()).hexdigest()
        self.built_images[cache_key] = digest
        return digest


_default_client: FakeClusterClient | None = None


def resolve_client(context: Any) -> FakeClusterClient:
    """Return the cluster client bound to a context, or a shared default."""
    global _default_client
    cluster = getattr(context, "cluster", None)
    if cluster is None and isinstance(context, dict):
        cluster = context.get("cluster")
    if cluster is not None:
        return cluster
    if _default_client is None:
        _default_client = FakeClusterClient()
    return _default_client
