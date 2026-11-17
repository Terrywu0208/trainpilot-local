#!/usr/bin/env bash
# deploy_vm.sh — one-command TrainPilot-Local cluster deployment.
#
# Brings up the full KubeRay path on a fresh Ubuntu 22.04+ VM:
# Docker → kind/kubectl/helm → Kind cluster (infra/kind/cluster.yaml)
# → KubeRay operator 1.3.0 → namespaces/RBAC/limits → training image
# build + kind load → digest pin → smoke RayJob → SUCCEEDED check.
#
# Usage:
#   scripts/deploy_vm.sh              # 3-node cluster (VM with >= 8 GB RAM)
#   scripts/deploy_vm.sh --small      # 1-node profile (VM with ~4 GB RAM)
#   scripts/deploy_vm.sh --skip-smoke # everything except the smoke RayJob
#   scripts/deploy_vm.sh --teardown   # delete the Kind cluster
#
# The script is idempotent: re-running skips steps already done.
# One intentional side effect outside the cluster: after building the
# training image it pins its measured digest into
# src/trainpilot/tools/context.py (replacing the local-build
# sentinel), exactly as OPERATIONS.md Flow F prescribes. The change
# is printed when it happens; `git diff` shows it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CLUSTER_NAME="trainpilot"
KIND_VERSION="v0.27.0"
KUBECTL_VERSION="v1.31.2"
HELM_VERSION="v3.16.3"
OPERATOR_CHART_VERSION="1.3.0"
TRAINING_IMAGE="trainpilot/training:2.41.0-cpu"
JOBS_NS="trainpilot-jobs"
SMALL=0
SKIP_SMOKE=0
TEARDOWN=0

for arg in "$@"; do
  case "$arg" in
    --small) SMALL=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --teardown) TEARDOWN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n=== %s ===\n' "$*"; }
ok() { printf 'OK: %s\n' "$*"; }

# docker may need sudo until the user re-logs into the docker group.
DOCKER="docker"
if ! docker ps >/dev/null 2>&1; then
  if sudo docker ps >/dev/null 2>&1; then DOCKER="sudo docker"; fi
fi

if [ "$TEARDOWN" = "1" ]; then
  step "Teardown"
  kind delete cluster --name "$CLUSTER_NAME" || true
  echo "Cluster deleted. Note: the training-image digest pin in"
  echo "src/trainpilot/tools/context.py (if set by a previous deploy)"
  echo "was left as-is; restore it with:"
  echo "  git checkout -- src/trainpilot/tools/context.py"
  exit 0
fi

step "Preflight"
[ -r /etc/os-release ] && . /etc/os-release
echo "OS: ${PRETTY_NAME:-unknown} | arch: $(uname -m)"
echo "RAM: $(free -h | awk '/^Mem:/{print $2}') total | disk free: $(df -h . | awk 'NR==2{print $4}')"
mkdir -p workspaces
ok "repo at $REPO_ROOT"

step "Docker"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
  DOCKER="sudo docker"
fi
$DOCKER ps >/dev/null 2>&1 || { sudo systemctl start docker || true; }
$DOCKER ps >/dev/null
ok "docker $($DOCKER --version)"

step "kind / kubectl / helm"
if ! command -v kind >/dev/null 2>&1; then
  curl -fsSL -o /tmp/kind "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-amd64"
  sudo install -m 0755 /tmp/kind /usr/local/bin/kind
fi
if ! command -v kubectl >/dev/null 2>&1; then
  curl -fsSL -o /tmp/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
  sudo install -m 0755 /tmp/kubectl /usr/local/bin/kubectl
fi
if ! command -v helm >/dev/null 2>&1; then
  curl -fsSL "https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz" | tar -xz -C /tmp
  sudo install -m 0755 /tmp/linux-amd64/helm /usr/local/bin/helm
fi
ok "kind $(kind --version) | kubectl $(kubectl version --client -o json 2>/dev/null | grep -o '"gitVersion": *"[^"]*"' | head -1 || echo present) | helm $(helm version --short)"

step "Kind cluster ($CLUSTER_NAME)"
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  ok "cluster already exists, skipping create"
else
  if [ "$SMALL" = "1" ]; then
    python3 - <<'PYEOF'
import re
src = open('infra/kind/cluster.yaml').read()
head, _, _ = src.partition('  - role: worker')
open('/tmp/kind-small.yaml', 'w').write(head.rstrip() + '\n')
PYEOF
    kind create cluster --config /tmp/kind-small.yaml
  else
    kind create cluster --config infra/kind/cluster.yaml
  fi
fi
kubectl cluster-info --context "kind-$CLUSTER_NAME" >/dev/null
ok "cluster API reachable: $(kubectl get nodes --no-headers | wc -l) node(s)"

step "KubeRay operator $OPERATOR_CHART_VERSION"
helm repo add kuberay https://ray-project.github.io/kuberay-helm/ >/dev/null 2>&1 || true
helm repo update >/dev/null
helm upgrade --install kuberay-operator kuberay/kuberay-operator \
  --version "$OPERATOR_CHART_VERSION" \
  -n kuberay-system --create-namespace \
  -f infra/kuberay/values.yaml --wait --timeout 5m
ok "operator Ready in kuberay-system"

step "Namespaces / limits / RBAC"
kubectl apply -f infra/k8s/namespaces.yaml
kubectl apply -f infra/k8s/limits.yaml
kubectl apply -f infra/k8s/rbac.yaml
ok "applied namespaces.yaml, limits.yaml, rbac.yaml"

step "Training image"
$DOCKER build -f training/Dockerfile -t "$TRAINING_IMAGE" .
kind load docker-image "$TRAINING_IMAGE" --name "$CLUSTER_NAME"
IMG_ID="$($DOCKER inspect --format '{{.Id}}' "$TRAINING_IMAGE")"
ok "built and loaded $TRAINING_IMAGE ($IMG_ID)"

step "Pin training-image digest"
if grep -q 'sha256:local-build-unpinned' src/trainpilot/tools/context.py; then
  sed -i "s|sha256:local-build-unpinned|$IMG_ID|" src/trainpilot/tools/context.py
  echo "Pinned TRAINING_IMAGE_DIGEST to $IMG_ID in src/trainpilot/tools/context.py"
else
  echo "context.py already pinned (sentinel absent); leaving as-is. Current value:"
  grep -n 'TRAINING_IMAGE_DIGEST =' src/trainpilot/tools/context.py || true
fi

if [ "$SKIP_SMOKE" = "1" ]; then
  step "Done (smoke skipped)"
  echo "Cluster is up. Submit the smoke job any time with:"
  echo "  kubectl apply -f infra/k8s/smoke-rayjob.yaml"
  exit 0
fi

step "Smoke RayJob"
SMOKE_YAML="infra/k8s/smoke-rayjob.yaml"
if [ "$SMALL" = "1" ]; then
  sed -e 's/replicas: 2/replicas: 1/' -e 's/minReplicas: 2/minReplicas: 1/' \
      -e 's/maxReplicas: 2/maxReplicas: 1/' -e 's/memory: "1536Mi"/memory: "1Gi"/g' \
      "$SMOKE_YAML" > /tmp/smoke-rayjob-small.yaml
  SMOKE_YAML="/tmp/smoke-rayjob-small.yaml"
  echo "small profile: 1 worker, tighter memory limits"
fi
kubectl delete rayjob trainpilot-smoke -n "$JOBS_NS" --ignore-not-found >/dev/null
kubectl apply -f "$SMOKE_YAML"
echo "waiting for the smoke job to finish (up to 15 minutes)..."
STATUS=""
for i in $(seq 1 90); do
  STATUS="$(kubectl get rayjob trainpilot-smoke -n "$JOBS_NS" \
    -o jsonpath='{.status.jobStatus}' 2>/dev/null || echo '')"
  case "$STATUS" in
    SUCCEEDED) break ;;
    FAILED|STOPPED) echo "smoke job ended in state: $STATUS" >&2; break ;;
    *) sleep 10 ;;
  esac
done
echo "--- pods in $JOBS_NS ---"
kubectl get pods -n "$JOBS_NS"
if [ "$STATUS" = "SUCCEEDED" ] && [ -f workspaces/smoke-001/checkpoint-final/metrics.json ]; then
  step "DEPLOYMENT VERIFIED"
  echo "RayJob trainpilot-smoke: SUCCEEDED"
  echo "Checkpoint on host: workspaces/smoke-001/checkpoint-final/"
  cat workspaces/smoke-001/checkpoint-final/metrics.json || true
  echo
  echo "Next: point the MCP action server at this cluster (OPERATIONS.md Flow F),"
  echo "or tear down with: scripts/deploy_vm.sh --teardown"
else
  echo "Smoke job did not verify (status: ${STATUS:-unknown}; checkpoint metrics present: $(test -f workspaces/smoke-001/checkpoint-final/metrics.json && echo yes || echo no))." >&2
  echo "Debug with:" >&2
  echo "  kubectl describe rayjob trainpilot-smoke -n $JOBS_NS" >&2
  echo "  kubectl logs -n $JOBS_NS -l ray.io/job-name=trainpilot-smoke --tail=100" >&2
  exit 1
fi
