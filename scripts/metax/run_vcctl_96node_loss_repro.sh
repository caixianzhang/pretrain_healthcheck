#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_NODES=96
exec bash "${SCRIPT_DIR}/run_vcctl_node_loss_repro.sh" "$@"
