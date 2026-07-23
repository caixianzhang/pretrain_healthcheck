# Muxi 96/128-Node Loss Reproduction

This entry reproduces the observed correlation between high-concurrency
training-topology communication and Pod/GPU/RDMA health loss. It is a vendor
diagnostic workload, not a routine training admission test.

## Safety Boundary

- The workload uses TP=4, EP=32, ETP=1, PP=8, CP=1.
- TP and PP payloads are 32 MiB; EP payloads are 64 MiB.
- It runs bf16 with one warmup and one measured iteration.
- It does not run a 768/1024-rank global All-to-All.
- It does not run `mx-smi --kill-all-process`.
- Cleanup only targets processes containing the current `REPRO_RUN_ID`.
- Formal execution requires `CONFIRM_NODE_LOSS_REPRO=YES`.
- The developer-side topology exporter defaults to
  `DRIVER_PYTHON=/opt/conda/bin/python3` and `MACA_HOME=/opt/maca`.

Do not run it on a Job carrying training or another communication workload.

## Excluding Known Failed Nodes

`EXCLUDED_NODES` accepts exact, case-sensitive hostnames separated by commas,
spaces, or newlines:

```bash
EXCLUDED_NODES=host-10-12-144-79,host-10-12-144-168
```

The controller reads the complete Job first, validates every requested
hostname, removes matched nodes, validates all remaining Pods, and then selects
the first 96 or 128 Pods in stable task/ordinal order. Unknown hostnames are a
configuration error. If too few eligible nodes remain, the run exits with code
20 and never silently downgrades its scale.

Only add nodes confirmed by platform liveness loss, static hardware evidence,
or vendor diagnosis. Do not exclude every member of a failed communication
group.

## Preflight

Preflight writes the exact selection plan but does not start communication and
does not require the confirmation token:

```bash
cd /afs-a3-241ceshi-shared/zhangcaixian/scale_up10000/pretrain_healthcheck

JOB_NAME=muxi-128node-4 \
EXCLUDED_NODES=host-10-12-144-79,host-10-12-144-168 \
PREFLIGHT_ONLY=1 \
bash scripts/metax/run_vcctl_96node_loss_repro.sh
```

Review:

```text
results/vcctl/<RUN_ID>/
  excluded_nodes_requested.txt
  excluded_nodes_matched.tsv
  eligible_nodes.txt
  inputs/selected_nodes.txt
  standby_nodes.txt
  process_preflight.json
  topology_profile_summary.json
  reproduction_manifest.json
```

## Formal 96-Node Reproduction

```bash
JOB_NAME=muxi-128node-4 \
EXCLUDED_NODES=host-10-12-144-79,host-10-12-144-168 \
CONFIRM_NODE_LOSS_REPRO=YES \
bash scripts/metax/run_vcctl_96node_loss_repro.sh
```

## Formal 128-Node Reproduction

This requires 128 eligible Running/Ready Pods after exclusions:

```bash
JOB_NAME=muxi-128node-4 \
EXCLUDED_NODES= \
CONFIRM_NODE_LOSS_REPRO=YES \
bash scripts/metax/run_vcctl_128node_loss_repro.sh
```

Each formal run performs:

1. Pod membership, readiness, IP, 8-GPU, and process checks.
2. A 120-second idle liveness baseline sampled every 5 seconds.
3. Training-topology communication on the selected nodes.
4. Immediate stop and targeted cleanup when liveness degrades.
5. A 120-second observation window for delayed GPU/RDMA unhealthy states.
6. Error-signature extraction and a suggested exclusion list.

## Result Inspection

```bash
bash scripts/metax/inspect_vcctl_node_loss_repro.sh \
  results/vcctl/<RUN_ID>
```

Key files:

| File | Meaning |
| --- | --- |
| `run_summary.md/json` | Final status, exit code, elapsed time, and scale |
| `lost_nodes.tsv` | Lost Pod, hostname, target/standby membership, reason, message |
| `suggested_excluded_nodes.txt` | Hostnames suitable for explicit review before the next run |
| `liveness_events.jsonl` | Time-ordered Ready/IP/UID/phase observations |
| `job_liveness_alert.json` | First observed liveness transition |
| `job_state_first_failure.json` | Full Job snapshot at failure handling |
| `job_state_after_30s.json` | Full Job snapshot 30 seconds later |
| `job_state_after_120s.json` | Full Job snapshot after the observation window |
| `error_signatures.jsonl` | QP, ATU, illegal-memory, NET/IB, admission, and unhealthy signatures |
| `evidence/error_excerpt.log` | Human-readable excerpts from controller and result logs |
| `raw_output` | Symlink to developer-local `/tmp` detailed output |

The suggested list is not automatically applied. Inspect the reason and
platform status before adding it to the next `EXCLUDED_NODES` value.

## Exit Codes

| Code | Meaning |
| --- | --- |
| 0 | Communication completed without node loss, or preflight passed |
| 10 | A selected communication node lost readiness/health; reproduction succeeded |
| 11 | A non-selected standby Job node lost readiness/health |
| 20 | Preflight failed or too few eligible nodes remained |
| 30 | Communication failed but platform node status did not degrade |
| 40 | Controller error |

Exit code 30 is not proof of node loss. Exit code 10 requires a matching
platform liveness transition on a selected node.
