from __future__ import annotations

import json
import base64
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .torch_checks import (
    _all_gather_object,
    _busbw_factor,
    _collective_bandwidth_once,
    _prepare_collective_bandwidth_workspace,
    _routing_counts,
    _routing_output_counts,
    _synchronize,
    _timing_event_api,
    _dtype,
    append_jsonl,
    init_dist,
    percentile,
    size_to_label,
    write_json,
)
from .training_topology import (
    FAMILY_ORDER,
    TopologyGroup,
    load_training_topology_manifest,
    require_profile,
    topology_case_plan,
)


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _filtered_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    encoded = os.environ.get("TOPOLOGY_RETEST_PLAN_B64", "").strip()
    if not encoded:
        return cases
    plan = json.loads(base64.b64decode(encoded).decode("utf-8"))
    if not isinstance(plan, list):
        raise ValueError("TOPOLOGY_RETEST_PLAN_B64 must decode to a list")
    selected: list[dict[str, Any]] = []
    for case in cases:
        for item in plan:
            if not isinstance(item, dict):
                continue
            family = str(item.get("topology_family", item.get("family", "")))
            op = str(item.get("op_type", item.get("op", "")))
            pattern = str(item.get("payload_pattern", "none"))
            size = int(item.get("requested_message_bytes", item.get("message_bytes", 0)) or 0)
            if family == case["family"] and op == case["op"] and pattern == case["payload_pattern"] and size == case["message_bytes"]:
                selected.append(case)
                break
    if not selected:
        raise ValueError("topology retest plan did not match any manifest case")
    return selected


def _barrier(dist: Any, env: Any, group: Any | None = None) -> None:
    kwargs: dict[str, Any] = {"group": group}
    if env.local_rank >= 0:
        kwargs["device_ids"] = [env.local_rank]
    dist.barrier(**kwargs)


def _events(torch: Any, env: Any) -> tuple[Any | None, Any | None]:
    api = _timing_event_api(torch, env)
    event_type = getattr(api, "Event", None) if api is not None else None
    if event_type is None:
        return None, None
    try:
        return event_type(enable_timing=True), event_type(enable_timing=True)
    except TypeError:
        return event_type(), event_type()


def _measure(
    torch: Any,
    dist: Any,
    env: Any,
    group: Any,
    operation: Any,
    warmup: int,
    iters: int,
    batches: int,
) -> tuple[list[float], str]:
    for _ in range(max(0, warmup)):
        operation()
    _synchronize(torch)
    _barrier(dist, env, group)
    latencies: list[float] = []
    timing_mode = "steady_state_device_event"
    for _ in range(max(1, batches)):
        _barrier(dist, env, group)
        start_event, end_event = _events(torch, env)
        host_started = time.perf_counter()
        if start_event is not None:
            start_event.record()
        for _iteration in range(max(1, iters)):
            operation()
        if end_event is not None:
            end_event.record()
        _synchronize(torch)
        if start_event is not None and end_event is not None:
            elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
        else:
            timing_mode = "steady_state_host_fallback"
            elapsed = time.perf_counter() - host_started
        latencies.append(elapsed / max(1, iters))
    return latencies, timing_mode


def _fill_chunks(tensor: Any, counts: list[int], values: list[float]) -> None:
    offset = 0
    for count, value in zip(counts, values):
        if count:
            tensor[offset : offset + count].fill_(value)
        offset += count


def _collective_correctness(
    torch: Any,
    dist: Any,
    env: Any,
    spec: TopologyGroup,
    group: Any,
    group_rank: int,
    op: str,
    pattern: str,
    seed: int,
) -> bool:
    dtype = torch.float32
    group_size = len(spec.ranks)
    rank_value = float(env.rank + 1)
    expected_sum = float(sum(rank + 1 for rank in spec.ranks))
    if op == "all_reduce":
        tensor = torch.tensor([rank_value], device=env.device, dtype=dtype)
        dist.all_reduce(tensor, group=group)
        return math.isclose(float(tensor.item()), expected_sum, rel_tol=0.0, abs_tol=1e-4)
    if op == "broadcast":
        tensor = torch.tensor([rank_value], device=env.device, dtype=dtype)
        dist.broadcast(tensor, src=spec.ranks[0], group=group)
        return math.isclose(float(tensor.item()), float(spec.ranks[0] + 1), abs_tol=1e-4)
    if op == "reduce_scatter":
        chunks = [torch.tensor([rank_value], device=env.device, dtype=dtype) for _ in range(group_size)]
        output = torch.empty(1, device=env.device, dtype=dtype)
        dist.reduce_scatter(output, chunks, group=group)
        return math.isclose(float(output.item()), expected_sum, rel_tol=0.0, abs_tol=1e-4)
    if op == "all_gather":
        tensor = torch.tensor([rank_value], device=env.device, dtype=dtype)
        outputs = [torch.empty_like(tensor) for _ in range(group_size)]
        dist.all_gather(outputs, tensor, group=group)
        actual = [int(round(float(item.item()))) - 1 for item in outputs]
        return actual == list(spec.ranks)
    if op == "all_to_all":
        send = torch.tensor(
            [float(env.rank * 10000 + destination) for destination in spec.ranks],
            device=env.device,
            dtype=dtype,
        )
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        expected = [float(source * 10000 + env.rank) for source in spec.ranks]
        return all(math.isclose(float(actual), wanted, abs_tol=1e-3) for actual, wanted in zip(recv.tolist(), expected))
    if op == "all_to_allv":
        total = max(group_size * 4, group_size)
        send_counts = _routing_counts(pattern, group_size, total, group_rank, seed)
        recv_counts = _routing_output_counts(pattern, group_size, total, group_rank, seed)
        send = torch.empty(sum(send_counts), device=env.device, dtype=dtype)
        recv = torch.empty(sum(recv_counts), device=env.device, dtype=dtype)
        _fill_chunks(
            send,
            send_counts,
            [float(env.rank * 10000 + destination) for destination in spec.ranks],
        )
        dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=group,
        )
        offset = 0
        for source_rank, count in zip(spec.ranks, recv_counts):
            expected = float(source_rank * 10000 + env.rank)
            values = recv[offset : offset + count]
            if count and not bool(torch.allclose(values, torch.full_like(values, expected))):
                return False
            offset += count
        return True
    raise ValueError(op)


def _pp_operation(torch: Any, dist: Any, env: Any, spec: TopologyGroup, numel: int) -> tuple[Any, Any]:
    index = spec.ranks.index(env.rank)
    send = torch.full((numel,), float(env.rank + 1), device=env.device, dtype=torch.float32)
    recv_forward = torch.empty_like(send) if index > 0 else None
    recv_reverse = torch.empty_like(send) if index + 1 < len(spec.ranks) else None

    def operation() -> None:
        forward_ops = []
        if index + 1 < len(spec.ranks):
            forward_ops.append(dist.P2POp(dist.isend, send, spec.ranks[index + 1]))
        if index > 0:
            forward_ops.append(dist.P2POp(dist.irecv, recv_forward, spec.ranks[index - 1]))
        if forward_ops:
            for work in dist.batch_isend_irecv(forward_ops):
                work.wait()
        reverse_ops = []
        if index > 0:
            reverse_ops.append(dist.P2POp(dist.isend, send, spec.ranks[index - 1]))
        if index + 1 < len(spec.ranks):
            reverse_ops.append(dist.P2POp(dist.irecv, recv_reverse, spec.ranks[index + 1]))
        if reverse_ops:
            for work in dist.batch_isend_irecv(reverse_ops):
                work.wait()

    def correctness() -> bool:
        operation()
        ok = True
        if recv_forward is not None:
            ok = ok and bool(torch.allclose(recv_forward, torch.full_like(recv_forward, float(spec.ranks[index - 1] + 1))))
        if recv_reverse is not None:
            ok = ok and bool(torch.allclose(recv_reverse, torch.full_like(recv_reverse, float(spec.ranks[index + 1] + 1))))
        return ok

    return operation, correctness


def _workspace_for_case(
    torch: Any,
    env: Any,
    spec: TopologyGroup,
    group_rank: int,
    op: str,
    message_bytes: int,
    pattern: str,
    seed: int,
) -> tuple[Any, int]:
    dtype = _dtype(torch, os.environ.get("TOPOLOGY_DTYPE", "bf16"))
    element_size = torch.empty((), dtype=dtype).element_size()
    numel = max(1, message_bytes // element_size)
    group_size = len(spec.ranks)
    send_counts = recv_counts = None
    effective_bytes = message_bytes
    if op == "all_gather":
        tensor_numel = max(1, numel // group_size)
        effective_bytes = tensor_numel * element_size * group_size
    elif op == "all_to_allv":
        token_size = max(1, numel // group_size)
        send_counts = _routing_counts(pattern, group_size, token_size * group_size, group_rank, seed)
        recv_counts = _routing_output_counts(pattern, group_size, token_size * group_size, group_rank, seed)
        tensor_numel = sum(send_counts)
        effective_bytes = tensor_numel * element_size
    else:
        tensor_numel = numel
    workspace = _prepare_collective_bandwidth_workspace(
        torch,
        env,
        op,
        tensor_numel,
        group_rank,
        group_size,
        dtype,
        input_split_sizes=send_counts,
        output_split_sizes=recv_counts,
    )
    return workspace, effective_bytes


def _summaries_from_rows(case: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row["topology_group_id"]), []).append(row)
    group_summaries: list[dict[str, Any]] = []
    for topology_group_id, group_rows in sorted(by_group.items()):
        batch_count = max(int(row["measurement_batches"]) for row in group_rows)
        latencies: list[float] = []
        for batch_index in range(batch_count):
            values = [float(row["latencies"][batch_index]) for row in group_rows]
            latencies.append(max(values))
        effective_bytes = max(int(row["message_bytes"]) for row in group_rows)
        group_size = int(group_rows[0]["collective_group_size"])
        busbw = [effective_bytes / max(value, 1e-12) / 1e9 * _busbw_factor(str(case["op"]), group_size) for value in latencies]
        group_summaries.append(
            {
                "topology_family": case["family"],
                "topology_group_id": topology_group_id,
                "op_type": case["op"],
                "payload_pattern": case["payload_pattern"],
                "message_size": size_to_label(int(case["message_bytes"])),
                "message_bytes": effective_bytes,
                "requested_message_bytes": int(case["message_bytes"]),
                "collective_group_size": group_size,
                "measurement_batches": batch_count,
                "iterations_per_batch": int(group_rows[0]["iterations_per_batch"]),
                "timing_mode": str(group_rows[0]["timing_mode"]),
                "latency_p50": percentile(latencies, 0.50),
                "latency_p95": percentile(latencies, 0.95),
                "latency_p99": percentile(latencies, 0.99),
                "avg_busbw": sum(busbw) / len(busbw),
                "second_lowest_busbw": sorted(busbw)[1] if len(busbw) > 1 else busbw[0],
                "correctness_pass": all(bool(row["correctness_pass"]) for row in group_rows),
                "performance_pass": True,
                "error_type": "" if all(bool(row["correctness_pass"]) for row in group_rows) else "TOPOLOGY_CORRECTNESS_FAIL",
                "source_stage": "training_topology",
                "case_source": case["source"],
            }
        )
    cohort = {
        "topology_family": case["family"],
        "topology_group_id": "cohort",
        "op_type": case["op"],
        "payload_pattern": case["payload_pattern"],
        "message_size": size_to_label(int(case["message_bytes"])),
        "message_bytes": int(case["message_bytes"]),
        "requested_message_bytes": int(case["message_bytes"]),
        "collective_group_size": int(group_summaries[0]["collective_group_size"]),
        "measurement_batches": max(int(row["measurement_batches"]) for row in group_summaries),
        "iterations_per_batch": max(int(row["iterations_per_batch"]) for row in group_summaries),
        "timing_mode": "worst_subgroup",
        "latency_p50": max(float(row["latency_p50"]) for row in group_summaries),
        "latency_p95": max(float(row["latency_p95"]) for row in group_summaries),
        "latency_p99": max(float(row["latency_p99"]) for row in group_summaries),
        "avg_busbw": min(float(row["avg_busbw"]) for row in group_summaries),
        "second_lowest_busbw": min(float(row["second_lowest_busbw"]) for row in group_summaries),
        "correctness_pass": all(bool(row["correctness_pass"]) for row in group_summaries),
        "performance_pass": True,
        "error_type": "" if all(bool(row["correctness_pass"]) for row in group_summaries) else "TOPOLOGY_CORRECTNESS_FAIL",
        "source_stage": "training_topology",
        "case_source": case["source"],
        "subgroup_count": len(group_summaries),
        "aggregation": "worst_subgroup",
    }
    return group_summaries, cohort


def _run_overlap_canary(
    torch: Any,
    dist: Any,
    env: Any,
    selected: dict[str, tuple[TopologyGroup, Any, int]],
    batches: int,
) -> dict[str, Any] | None:
    tp_spec, tp_group, _ = selected["tp"]
    dp_spec, dp_group, _ = selected["dense_dp"]
    ep_spec, ep_group, ep_rank = selected["ep"]
    dtype = _dtype(torch, os.environ.get("TOPOLOGY_DTYPE", "bf16"))
    element_size = torch.empty((), dtype=dtype).element_size()
    numel = max(1, (1 << 20) // element_size)
    tp_tensor = torch.ones(numel, device=env.device, dtype=dtype)
    dp_tensor = torch.ones(numel, device=env.device, dtype=dtype)
    ep_workspace, _ = _workspace_for_case(
        torch, env, ep_spec, ep_rank, "all_to_all", 1 << 20, "none", 20260623
    )

    def operation() -> None:
        works = [
            dist.all_reduce(tp_tensor, group=tp_group, async_op=True),
            dist.all_reduce(dp_tensor, group=dp_group, async_op=True),
            dist.all_to_all_single(
                ep_workspace.output_tensor,
                ep_workspace.input_tensor,
                group=ep_group,
                async_op=True,
            ),
        ]
        for work in works:
            work.wait()

    latencies, timing_mode = _measure(torch, dist, env, None, operation, 1, 1, batches)
    local = {
        "rank": env.rank,
        "latencies": latencies,
        "correctness_pass": bool(torch.isfinite(tp_tensor).all() and torch.isfinite(dp_tensor).all()),
    }
    gathered = _all_gather_object(dist, local, env.world_size)
    if env.rank != 0:
        return None
    cohort_latencies = [
        max(float(row["latencies"][batch_index]) for row in gathered)
        for batch_index in range(len(latencies))
    ]
    busbw = [(1 << 20) / max(value, 1e-12) / 1e9 for value in cohort_latencies]
    return {
        "topology_family": "overlap",
        "topology_group_id": "all_communicators",
        "op_type": "multi_communicator_canary",
        "payload_pattern": "none",
        "message_size": "1M",
        "message_bytes": 1 << 20,
        "requested_message_bytes": 1 << 20,
        "collective_group_size": env.world_size,
        "measurement_batches": len(cohort_latencies),
        "iterations_per_batch": 1,
        "timing_mode": timing_mode,
        "latency_p50": percentile(cohort_latencies, 0.50),
        "latency_p95": percentile(cohort_latencies, 0.95),
        "latency_p99": percentile(cohort_latencies, 0.99),
        "avg_busbw": sum(busbw) / len(busbw),
        "second_lowest_busbw": sorted(busbw)[1] if len(busbw) > 1 else busbw[0],
        "correctness_pass": all(bool(row["correctness_pass"]) for row in gathered),
        "performance_pass": True,
        "error_type": "" if all(bool(row["correctness_pass"]) for row in gathered) else "CONCURRENT_COMM_RESOURCE_FAIL",
        "source_stage": "training_topology",
        "case_source": "overlap_canary",
        "subgroup_count": 3,
        "aggregation": "global_slowest_rank",
    }


def run_training_topology_suite(
    output_dir: Path,
    manifest_path: Path,
    ranks_per_node: int,
    dtype_name: str,
    warmup: int,
    iters: int,
    seed: int,
    test_round: str,
    group_id: str,
) -> None:
    torch, dist, env = init_dist()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_training_topology_manifest(manifest_path)
    expected_manifest_sha256 = os.environ.get("TOPOLOGY_MANIFEST_SHA256", "").strip()
    if expected_manifest_sha256 and manifest.sha256 != expected_manifest_sha256:
        raise RuntimeError(
            f"topology manifest checksum mismatch expected={expected_manifest_sha256} actual={manifest.sha256}"
        )
    profile = require_profile(manifest, env.world_size, ranks_per_node)
    retest_mode = bool(os.environ.get("TOPOLOGY_RETEST_PLAN_B64", "").strip())
    overlap_canary_enabled = not retest_mode and _truthy(
        os.environ.get("TOPOLOGY_OVERLAP_CANARY", "0")
    )
    cases = _filtered_cases(topology_case_plan(profile))
    env.group_id = group_id or env.group_id or f"{test_round}-{env.world_size}"
    selected: dict[str, tuple[TopologyGroup, Any, int]] = {}
    for family in FAMILY_ORDER:
        for spec in profile.groups[family]:
            handle = dist.new_group(ranks=list(spec.ranks))
            if env.rank in spec.ranks:
                selected[family] = (spec, handle, spec.ranks.index(env.rank))
    if set(selected) != set(FAMILY_ORDER):
        raise RuntimeError(f"rank {env.rank} did not join every topology family: {sorted(selected)}")

    if env.local_rank == 0:
        write_json(
            output_dir / "training_topology_plan.json",
            {
                "schema_version": 1,
                "manifest_sha256": manifest.sha256,
                "world_size": env.world_size,
                "ranks_per_node": ranks_per_node,
                "case_count": len(cases) + int(overlap_canary_enabled),
                "cases": cases,
                "group_counts": {family: len(profile.groups[family]) for family in FAMILY_ORDER},
                "overlap_canary_enabled": overlap_canary_enabled,
            },
        )

    batches = max(1, int(os.environ.get("DYNAMIC_COMPARE_MEASUREMENT_BATCHES", "1")))
    local_summary_dir = output_dir / "training_topology_rank_summaries"
    local_summary_dir.mkdir(parents=True, exist_ok=True)
    local_summary_path = local_summary_dir / f"rank_{env.rank:06d}.jsonl"
    local_group_summary_count = 0
    local_rows_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in FAMILY_ORDER
    }
    try:
        for case in cases:
            family = str(case["family"])
            op = str(case["op"])
            spec, handle, group_rank = selected[family]
            if op == "send_recv":
                element_size = torch.empty((), dtype=torch.float32).element_size()
                operation, correctness_check = _pp_operation(
                    torch, dist, env, spec, max(1, int(case["message_bytes"]) // element_size)
                )
                correctness = correctness_check()
                effective_bytes = int(case["message_bytes"])
            else:
                correctness = _collective_correctness(
                    torch, dist, env, spec, handle, group_rank, op, str(case["payload_pattern"]), seed
                )
                workspace, effective_bytes = _workspace_for_case(
                    torch,
                    env,
                    spec,
                    group_rank,
                    op,
                    int(case["message_bytes"]),
                    str(case["payload_pattern"]),
                    seed,
                )
                operation = lambda op=op, workspace=workspace, handle=handle: _collective_bandwidth_once(
                    dist, op, workspace, group=handle
                )
            latencies, timing_mode = _measure(
                torch, dist, env, handle, operation, warmup, iters, batches
            )
            local_row = {
                "rank": env.rank,
                "node_name": env.node_name,
                "pod_name": env.pod_name,
                "topology_family": family,
                "topology_group_id": spec.group_id,
                "collective_group_size": len(spec.ranks),
                "message_bytes": effective_bytes,
                "latencies": latencies,
                "measurement_batches": len(latencies),
                "iterations_per_batch": max(1, iters),
                "timing_mode": timing_mode,
                "correctness_pass": correctness,
            }
            local_rows_by_family[family].append(local_row)

        for family in FAMILY_ORDER:
            spec, handle, group_rank = selected[family]
            gathered = _all_gather_object(
                dist,
                local_rows_by_family[family],
                len(spec.ranks),
                group=handle,
            )
            if group_rank == 0:
                family_cases = [case for case in cases if str(case["family"]) == family]
                if any(len(rank_rows) != len(family_cases) for rank_rows in gathered):
                    raise RuntimeError(
                        f"subgroup gather row count mismatch family={family} group={spec.group_id}"
                    )
                for case_index, case in enumerate(family_cases):
                    case_rows = [rank_rows[case_index] for rank_rows in gathered]
                    group_summaries, _cohort = _summaries_from_rows(case, case_rows)
                    if len(group_summaries) != 1:
                        raise RuntimeError(
                            f"subgroup gather produced {len(group_summaries)} summaries for {spec.group_id}"
                        )
                    summary = group_summaries[0]
                    append_jsonl(local_summary_path, [summary])
                    local_group_summary_count += 1
                    if not bool(summary["correctness_pass"]):
                        raise RuntimeError(
                            f"training topology correctness failed family={family} "
                            f"op={case['op']} group={spec.group_id}"
                        )

        if overlap_canary_enabled:
            _barrier(dist, env)
            overlap = _run_overlap_canary(torch, dist, env, selected, batches)
            if env.rank == 0 and overlap is not None:
                append_jsonl(local_summary_path, [overlap])
                local_group_summary_count += 1

        if env.local_rank == 0:
            write_json(
                output_dir / "training_topology_gate.json",
                {
                    "status": "PASS",
                    "manifest_sha256": manifest.sha256,
                    "world_size": env.world_size,
                    "local_group_summary_count": local_group_summary_count,
                    "overlap_canary_enabled": overlap_canary_enabled,
                    "job_id": os.environ.get("HEALTHCHECK_JOB_ID", str(uuid.uuid4())),
                },
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
