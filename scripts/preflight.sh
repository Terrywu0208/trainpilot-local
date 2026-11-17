#!/usr/bin/env bash
# preflight.sh — check that a VM can host the TrainPilot-Local
# deployment BEFORE anything is installed. Prints one PASS/FAIL line
# per check and exits non-zero if any check fails.
#
# Usage: scripts/preflight.sh [--small]
#   full profile (default): >= 4 vCPU, >= 8 GB RAM, >= 30 GB disk
#   --small:                >= 2 vCPU, >= 4 GB RAM, >= 25 GB disk
set -uo pipefail

SMALL=0
[ "${1:-}" = "--small" ] && SMALL=1
if [ "$SMALL" = "1" ]; then NEED_CPU=2; NEED_RAM_GB=4; NEED_DISK_GB=25
else NEED_CPU=4; NEED_RAM_GB=8; NEED_DISK_GB=30; fi

FAILS=0
report() { # report <ok 0/1> <label> <detail>
  if [ "$1" = "0" ]; then printf 'PASS  %-28s %s\n' "$2" "$3"
  else printf 'FAIL  %-28s %s\n' "$2" "$3"; FAILS=$((FAILS + 1)); fi
}

# OS
OS_OK=1; OS_DETAIL="not Ubuntu"
if [ -r /etc/os-release ]; then
  . /etc/os-release
  case "${ID:-}:${VERSION_ID:-}" in
    ubuntu:20.04|ubuntu:22.04|ubuntu:24.04) OS_OK=0; OS_DETAIL="Ubuntu $VERSION_ID" ;;
    ubuntu:*) OS_DETAIL="Ubuntu ${VERSION_ID:-?} (untested; 20.04/22.04/24.04 expected)" ;;
    *) OS_DETAIL="${PRETTY_NAME:-unknown OS} (Ubuntu expected)" ;;
  esac
fi
report $OS_OK "OS" "$OS_DETAIL"

# Arch
[ "$(uname -m)" = "x86_64" ]; report $? "Architecture" "$(uname -m) (x86_64 required)"

# CPU
CPUS="$(nproc 2>/dev/null || echo 0)"
[ "$CPUS" -ge "$NEED_CPU" ]; report $? "CPU" "$CPUS vCPU (need >= $NEED_CPU)"

# RAM
RAM_KB="$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
RAM_GB=$((RAM_KB / 1024 / 1024))
[ "$RAM_GB" -ge "$NEED_RAM_GB" ]; report $? "RAM" "${RAM_GB} GB (need >= ${NEED_RAM_GB} GB)"

# Disk (free space on the filesystem that will hold Docker data)
DISK_GB="$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')"
[ "${DISK_GB:-0}" -ge "$NEED_DISK_GB" ]; report $? "Disk free" "${DISK_GB:-?} GB (need >= ${NEED_DISK_GB} GB)"

# Privilege: root or passwordless sudo
PRIV_OK=1; PRIV_DETAIL="no root / passwordless sudo"
if [ "$(id -u)" = "0" ]; then PRIV_OK=0; PRIV_DETAIL="running as root"
elif sudo -n true 2>/dev/null; then PRIV_OK=0; PRIV_DETAIL="passwordless sudo available"; fi
report $PRIV_OK "Privilege" "$PRIV_DETAIL"

# Outbound reachability of everything the deploy downloads from.
# 401 from a registry /v2/ is success (it means "auth required").
check_url() { # check_url <url> <label>
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$1" 2>/dev/null || echo 000)"
  case "$code" in 2*|3*|401|404) report 0 "$2" "$1 -> HTTP $code" ;;
    *) report 1 "$2" "$1 -> HTTP $code (unreachable)" ;; esac
}
check_url "https://registry-1.docker.io/v2/" "Docker Hub registry"
check_url "https://quay.io/v2/" "quay.io (operator image)"
check_url "https://get.docker.com" "Docker installer"
check_url "https://kind.sigs.k8s.io/dl/v0.27.0/kind-linux-amd64" "kind binary"
check_url "https://dl.k8s.io/release/v1.31.2/bin/linux/amd64/kubectl" "kubectl binary"
check_url "https://ray-project.github.io/kuberay-helm/" "KubeRay Helm repo"
check_url "https://github.com" "GitHub (repo clone)"

echo
if [ "$FAILS" = "0" ]; then
  echo "PREFLIGHT: all checks passed — this VM can run the deployment."
else
  echo "PREFLIGHT: $FAILS check(s) failed — fix them before deploying."
  exit 1
fi
