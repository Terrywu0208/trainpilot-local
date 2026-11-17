#!/usr/bin/env bash
# bootstrap.sh — zero-to-deployed on a bare Ubuntu VM.
#
#   curl -fsSL https://raw.githubusercontent.com/Terrywu0208/trainpilot-local/main/scripts/bootstrap.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/Terrywu0208/trainpilot-local/main/scripts/bootstrap.sh | bash -s -- --small
#
# Installs git if needed, clones the repository (or updates an
# existing clone in ./trainpilot-local), then hands off to
# scripts/deploy_vm.sh with any arguments passed through.
set -euo pipefail

REPO_URL="https://github.com/Terrywu0208/trainpilot-local"
TARGET_DIR="trainpilot-local"

if ! command -v git >/dev/null 2>&1; then
  echo "git not found; installing..."
  SUDO="sudo"; [ "$(id -u)" = "0" ] && SUDO=""
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq git
fi

if [ -d "$TARGET_DIR/.git" ]; then
  echo "Existing clone found in $TARGET_DIR; updating..."
  git -C "$TARGET_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"
exec scripts/deploy_vm.sh "$@"
