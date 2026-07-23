#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/metax/inspect_vcctl_node_loss_repro.sh results/vcctl/<RUN_ID>" >&2
  exit 2
fi

RESULT_DIR="$(cd "$1" 2>/dev/null && pwd)" || {
  echo "[node-loss-inspect] result directory not found: $1" >&2
  exit 2
}

echo "[node-loss-inspect] result: ${RESULT_DIR}"
if [[ -f "${RESULT_DIR}/run_summary.md" ]]; then
  cat "${RESULT_DIR}/run_summary.md"
fi

echo
echo "## Lost Nodes"
if [[ -s "${RESULT_DIR}/lost_nodes.tsv" ]]; then
  cat "${RESULT_DIR}/lost_nodes.tsv"
else
  echo "No lost_nodes.tsv was generated."
fi

echo
echo "## Error Signatures"
if [[ -s "${RESULT_DIR}/evidence/error_excerpt.log" ]]; then
  sed -n '1,120p' "${RESULT_DIR}/evidence/error_excerpt.log"
else
  echo "No matching error signature was collected."
fi

echo
echo "## Suggested Next Exclusions"
if [[ -s "${RESULT_DIR}/suggested_excluded_nodes.txt" ]]; then
  exclusions="$(paste -sd, "${RESULT_DIR}/suggested_excluded_nodes.txt")"
  echo "EXCLUDED_NODES=${exclusions}"
else
  echo "No newly lost hostname was suggested."
fi
