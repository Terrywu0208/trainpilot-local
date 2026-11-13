"""Infra YAML tests: every file under infra/ must parse, and the parsed
content must carry the design's invariants (pinned images, restricted
Pod Security, RBAC without pods/exec, RayJob-only writer)."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
INFRA = REPO_ROOT / "infra"


def _yaml_files() -> list[Path]:
    return sorted(INFRA.rglob("*.yaml")) + sorted(INFRA.rglob("*.yml"))


def _docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text()) if d is not None]


def _all_docs() -> dict[str, list[dict]]:
    return {str(p.relative_to(INFRA)): _docs(p) for p in _yaml_files()}


def test_infra_has_the_expected_files():
    names = {str(p.relative_to(INFRA)) for p in _yaml_files()}
    assert {
        "kind/cluster.yaml",
        "kuberay/values.yaml",
        "k8s/namespaces.yaml",
        "k8s/limits.yaml",
        "k8s/rbac.yaml",
    } <= names


def test_every_infra_yaml_parses():
    assert _yaml_files(), "no infra YAML files found"
    for path in _yaml_files():
        docs = _docs(path)
        assert docs, f"{path} contains no YAML documents"
        for doc in docs:
            assert isinstance(doc, dict), f"{path}: document is not a mapping"


def test_kind_cluster_shape():
    (cluster,) = _all_docs()["kind/cluster.yaml"]
    assert cluster["kind"] == "Cluster"
    assert cluster["name"] == "trainpilot"
    roles = [node["role"] for node in cluster["nodes"]]
    assert roles == ["control-plane", "worker", "worker"]
    for node in cluster["nodes"]:
        # Digest-pinned node image, never a floating tag alone.
        assert node["image"].startswith("kindest/node:v1.31.2@sha256:")
    workers = [n for n in cluster["nodes"] if n["role"] == "worker"]
    for worker in workers:
        mounts = worker.get("extraMounts", [])
        assert len(mounts) == 1
        assert mounts[0]["hostPath"] == "./workspaces"
        assert mounts[0]["containerPath"] == "/trainpilot/workspaces"


def test_kuberay_values_pin_operator_and_resources():
    (values,) = _all_docs()["kuberay/values.yaml"]
    assert values["image"]["repository"] == "quay.io/kuberay/operator"
    assert values["image"]["tag"] == "v1.3.0"
    assert values["resources"]["limits"]["memory"] == "512Mi"
    text = (INFRA / "kuberay" / "values.yaml").read_text()
    assert "1.3.0" in text and "2.41.0" in text  # chart + Ray versions pinned


def test_namespaces_and_pod_security():
    docs = _all_docs()["k8s/namespaces.yaml"]
    by_name = {d["metadata"]["name"]: d for d in docs}
    assert set(by_name) == {"kuberay-system", "trainpilot-jobs", "trainpilot-system"}
    jobs_labels = by_name["trainpilot-jobs"]["metadata"]["labels"]
    assert jobs_labels["pod-security.kubernetes.io/enforce"] == "restricted"


def test_limits_quota_limitrange_networkpolicy():
    docs = _all_docs()["k8s/limits.yaml"]
    kinds = {d["kind"] for d in docs}
    assert kinds == {"ResourceQuota", "LimitRange", "NetworkPolicy"}
    quota = next(d for d in docs if d["kind"] == "ResourceQuota")
    assert quota["spec"]["hard"]["count/rayjobs.ray.io"] == "3"
    assert quota["spec"]["hard"]["pods"] == "20"
    limit_range = next(d for d in docs if d["kind"] == "LimitRange")
    container_limits = limit_range["spec"]["limits"][0]
    assert container_limits["max"]["memory"] == "3Gi"
    policy = next(d for d in docs if d["kind"] == "NetworkPolicy")
    assert policy["spec"]["policyTypes"] == ["Egress"]


def _rules_by_role() -> dict[str, list[dict]]:
    docs = _all_docs()["k8s/rbac.yaml"]
    roles = {}
    for doc in docs:
        if doc["kind"] in ("Role", "ClusterRole"):
            roles[doc["metadata"]["name"]] = doc["rules"]
    return roles


def test_rbac_has_both_service_accounts():
    docs = _all_docs()["k8s/rbac.yaml"]
    accounts = {
        d["metadata"]["name"] for d in docs if d["kind"] == "ServiceAccount"
    }
    assert accounts == {"trainpilot-reader", "trainpilot-writer"}


def test_rbac_never_grants_pods_exec():
    for role_name, rules in _rules_by_role().items():
        for rule in rules:
            resources = rule["resources"]
            assert "pods/exec" not in resources, role_name
            assert "pods/attach" not in resources, role_name
            assert "secrets" not in resources, role_name


def test_rbac_reader_is_read_only():
    rules = _rules_by_role()["trainpilot-reader-role"]
    for rule in rules:
        assert set(rule["verbs"]) <= {"get", "list", "watch"}, rule


def test_rbac_writer_is_rayjob_crud_only():
    rules = _rules_by_role()["trainpilot-writer-role"]
    ray_rules = [r for r in rules if r["apiGroups"] == ["ray.io"]]
    granted = {(tuple(r["resources"]), tuple(sorted(r["verbs"]))) for r in ray_rules}
    # RayJobs full CRUD; rayclusters must not appear with create/delete.
    for resources, verbs in granted:
        if "rayclusters" in resources:
            assert not ({"create", "delete"} & set(verbs))
    job_rule = next(r for r in ray_rules if r["resources"] == ["rayjobs"])
    assert set(job_rule["verbs"]) == {"create", "get", "list", "patch", "delete"}
    # No RBAC self-escalation surface.
    for rule in rules:
        assert "serviceaccounts" not in rule["resources"]
        assert "roles" not in rule["resources"]
        assert "rolebindings" not in rule["resources"]
