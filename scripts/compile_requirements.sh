#!/usr/bin/env bash
# Compile pinned requirements from requirements.in
set -euo pipefail

if ! command -v pip-compile >/dev/null 2>&1; then
  echo "pip-compile not found. Install pip-tools first: pip install pip-tools"
  exit 1
fi

pip-compile requirements.in --output-file requirements.txt
echo "Wrote requirements.txt"
