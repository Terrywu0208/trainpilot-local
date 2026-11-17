# Deploying TrainPilot-Local on a VM (Kind + KubeRay)

This runbook takes a fresh Ubuntu VM from nothing to a verified
KubeRay deployment: a Kind cluster running the KubeRay operator, the
project's namespaces/RBAC/limits, the training image loaded, and a
smoke RayJob that finishes `SUCCEEDED` and writes a real checkpoint
back to the host.

## Overview map

```
fresh Ubuntu VM
  └─ 1. Docker Engine            (script: get.docker.com)
  └─ 2. kind + kubectl + helm    (pinned binaries in /usr/local/bin)
  └─ 3. Kind cluster "trainpilot" (infra/kind/cluster.yaml, digest-pinned node image)
  └─ 4. KubeRay operator 1.3.0    (Helm, infra/kuberay/values.yaml) → namespace kuberay-system
  └─ 5. Project platform          (infra/k8s/: namespaces, ResourceQuota/LimitRange/NetworkPolicy, RBAC)
  └─ 6. Training image            (training/Dockerfile → trainpilot/training:2.41.0-cpu, kind load)
  └─ 7. Digest pin                (image ID written into src/trainpilot/tools/context.py)
  └─ 8. Smoke RayJob              (infra/k8s/smoke-rayjob.yaml → SUCCEEDED + checkpoint on host)
```

Everything the agent later does (submit RayJobs, read status/logs,
apply repairs) goes through the same KubeRay API this deployment
proves.

## Prerequisites

| | Full profile | `--small` profile |
|---|---|---|
| VM | Ubuntu 22.04 or 24.04, x86_64 | same |
| CPU / RAM | ≥ 4 vCPU / 8 GB | ≥ 2 vCPU / 4 GB |
| Disk | ≥ 30 GB free | ≥ 25 GB free |
| Access | sudo, outbound internet for image/binary pulls | same |
| Repo | this repository cloned on the VM | same |

Why these sizes: the full profile runs 3 Kind nodes plus a head and
2 workers; the small profile collapses to 1 node and 1 worker so a
laptop-class VM (for example a 2 vCPU / 4 GB cloud instance) can
still prove the whole path.

## One-command deployment

```bash
git clone https://github.com/Terrywu0208/trainpilot-local
cd trainpilot-local
scripts/deploy_vm.sh            # or: scripts/deploy_vm.sh --small
```

The script is idempotent — re-running it skips whatever already
exists. It prints one `=== step ===` banner per stage and ends with
`DEPLOYMENT VERIFIED` plus the smoke checkpoint's metrics, or a
non-zero exit with the exact `kubectl` commands to debug.

What it changes on the machine: installs Docker (if absent) and the
three CLIs, creates the Kind cluster and everything inside it,
builds the training image, and makes **one source edit** — pinning
the training image's measured digest in
`src/trainpilot/tools/context.py`, replacing the
`sha256:local-build-unpinned` sentinel. That pin is what lets the
tool validators accept RayJobs that reference the locally built
image; until it exists, the image fails closed. `git diff` shows
the pin; it is machine-specific and should not be committed for
anyone else's machine.

## Manual path (the same steps, by hand)

Each step lists its success check; do not continue past a failed
check.

1. **Docker.** `curl -fsSL https://get.docker.com | sudo sh`, then
   `docker ps` must print a (possibly empty) container table.
2. **CLIs.** Install kind v0.27.0, kubectl v1.31.2, helm v3.16.3
   (URLs are in `scripts/deploy_vm.sh`). Success: `kind --version`,
   `kubectl version --client`, `helm version --short` all print.
3. **Cluster.** `mkdir -p workspaces && kind create cluster --config infra/kind/cluster.yaml`
   (small profile: keep only the control-plane node block).
   Success: `kubectl get nodes` shows the nodes `Ready`.
4. **Operator.**
   `helm repo add kuberay https://ray-project.github.io/kuberay-helm/ && helm repo update`
   then
   `helm install kuberay-operator kuberay/kuberay-operator --version 1.3.0 -n kuberay-system --create-namespace -f infra/kuberay/values.yaml --wait`.
   Success: `kubectl get pods -n kuberay-system` shows the operator
   `Running`, and `kubectl get crd rayjobs.ray.io` exists.
5. **Platform.** `kubectl apply -f infra/k8s/namespaces.yaml -f infra/k8s/limits.yaml -f infra/k8s/rbac.yaml`.
   Success: `kubectl get ns` lists `trainpilot-jobs`,
   `trainpilot-system`, `kuberay-system`.
6. **Training image.** `docker build -f training/Dockerfile -t trainpilot/training:2.41.0-cpu .`
   then `kind load docker-image trainpilot/training:2.41.0-cpu --name trainpilot`.
   Success: the load command ends with the image ID. For the real
   torch fine-tune image, rebuild with `--build-arg INSTALL_TORCH=1`.
7. **Pin the digest.**
   `docker inspect --format '{{.Id}}' trainpilot/training:2.41.0-cpu`
   and put the printed `sha256:...` value in place of the
   `sha256:local-build-unpinned` sentinel in
   `src/trainpilot/tools/context.py` (`TRAINING_IMAGE_DIGEST`).
8. **Smoke job.** `kubectl apply -f infra/k8s/smoke-rayjob.yaml`,
   then watch `kubectl get rayjob trainpilot-smoke -n trainpilot-jobs -w`.
   Success: `jobStatus` becomes `SUCCEEDED`, and
   `workspaces/smoke-001/checkpoint-final/metrics.json` exists on
   the host with a `TRAINPILOT_RESULT`-consistent accuracy
   (canonical smoke value 0.8771).

## Using the cluster

Point the MCP action server at the real cluster instead of the fake
client (OPERATIONS.md Flow F); the eval harness's `--mode cluster`
then runs against real RayJobs. The web application is unaffected —
it talks to the same graph and tools.

## Teardown

```bash
scripts/deploy_vm.sh --teardown   # kind delete cluster --name trainpilot
```

Docker, the CLIs and the built image stay on the machine (they are
shared tooling); only the cluster and its workloads go away. To
also undo the digest pin: `git checkout -- src/trainpilot/tools/context.py`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `kind create` hangs at image pull | slow network; node image is ~1 GB | re-run; the script skips create once the cluster exists |
| Operator pod `ImagePullBackOff` | quay.io unreachable from the VM | check outbound access to quay.io; the operator digest is recorded in `infra/kuberay/values.yaml` |
| RayJob stuck `PENDING`, pods `Pending` | ResourceQuota or node capacity (small VM) | `kubectl describe pod -n trainpilot-jobs`; use `--small`, or raise the VM size |
| RayJob `FAILED`, log shows `DATASET_NOT_AVAILABLE` | a config asked for a dataset the image does not bundle | the smoke path only bundles `sst2`; this is the trainer refusing honestly, not a cluster fault |
| Checkpoint missing on host after `SUCCEEDED` | the workspace mount is missing | the Kind config mounts `./workspaces` from the directory `kind create` ran in — deploy from the repo root |
| Validator refuses the training image | digest pin not applied on this machine | redo step 7; the pin is per-machine by design |

## Version pins (single source of truth)

| Component | Pin | Where |
|---|---|---|
| Kubernetes node image | kindest/node v1.31.2 @ sha256:18fbefc2… | `infra/kind/cluster.yaml` |
| KubeRay operator | chart 1.3.0, image quay.io/kuberay/operator:v1.3.0 | `infra/kuberay/values.yaml` |
| Ray | 2.41.0 (rayproject/ray:2.41.0-py310-cpu @ sha256:1c9c04bf…) | `training/Dockerfile`, `src/trainpilot/tools/context.py` |
| kind / kubectl / helm | v0.27.0 / v1.31.2 / v3.16.3 | `scripts/deploy_vm.sh` |
