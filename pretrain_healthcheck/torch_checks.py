from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import append_jsonl, hostname, percentile, size_to_label, write_json


@dataclass
class DistEnv:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: Any
    hostname: str
    dist_backend_requested: str
    dist_backend: str
    device_vendor: str
    comm_runtime: str
    group_id: str


def _import_torch():
    try:
        import torch
        import torch.distributed as dist

        return torch, dist
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(f"PyTorch with distributed support is required: {exc}") from exc


def _resolve_dist_backend() -> tuple[str, str]:
    if os.environ.get("FAULT_BACKEND", "").lower() in {"1", "true", "yes", "on"}:
        requested = os.environ.get("DIST_BACKEND", "nccl").strip().lower()
        return requested, "__fault_invalid_backend__"

    requested = os.environ.get("DIST_BACKEND", "nccl").strip().lower()
    aliases = {
        "cuda": "nccl",
        "gpu": "nccl",
        "nvidia": "nccl",
        "metax": "nccl",
        "maca": "nccl",
        "ascend": "hccl",
        "npu": "hccl",
        "cpu": "gloo",
    }
    return requested, aliases.get(requested, requested)


def _maybe_import_backend_extension(backend: str) -> None:
    if backend == "hccl":
        try:
            import torch_npu  # noqa: F401
        except Exception:
            pass


def _runtime_meta(backend: str) -> tuple[str, str]:
    vendor = os.environ.get("DEVICE_VENDOR") or os.environ.get("HC_DEVICE_TYPE") or "unknown"
    vendor = vendor.strip().lower()
    runtime = os.environ.get("COMM_RUNTIME", "").strip().lower()
    if not runtime:
        if vendor in {"metax", "maca"}:
            runtime = "mccl"
        elif vendor in {"nvidia", "gpu"} and backend == "nccl":
            runtime = "nccl"
        elif vendor in {"ascend", "npu"} and backend == "hccl":
            runtime = "hccl"
        else:
            runtime = "unknown"
    return vendor, runtime


def init_dist() -> tuple[Any, Any, DistEnv]:
    torch, dist = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    backend_requested, backend = _resolve_dist_backend()
    device_vendor, comm_runtime = _runtime_meta(backend)
    group_id = os.environ.get("HEALTHCHECK_GROUP_ID") or os.environ.get("HC_GROUP_ID") or ""
    _maybe_import_backend_extension(backend)
    if not dist.is_initialized():
        try:
            dist.init_process_group(backend=backend)
        except Exception as exc:
            registered = sorted(getattr(dist.Backend, "backend_type_map", {}).keys())
            raise RuntimeError(
                f"failed to initialize torch.distributed backend "
                f"requested={backend_requested!r} resolved={backend!r}; "
                f"registered_backends={registered}: {exc}"
            ) from exc
    return torch, dist, DistEnv(
        rank,
        local_rank,
        world_size,
        local_world_size,
        device,
        hostname(),
        backend_requested,
        backend,
        device_vendor,
        comm_runtime,
        group_id,
    )


def _rank_matches_env(env_name: str, rank: int) -> bool:
    value = os.environ.get(env_name, "").strip()
    return bool(value) and value == str(rank)


def _maybe_fault_sleep(env: DistEnv) -> None:
    if _rank_matches_env("FAULT_SLEEP_RANK", env.rank):
        time.sleep(float(os.environ.get("FAULT_SLEEP_SECONDS", "30")))


def _apply_faults(torch: Any, env: DistEnv, tensor: Any) -> Any:
    if _rank_matches_env("FAULT_NAN_RANK", env.rank) and tensor.numel() > 0:
        tensor = tensor.clone()
        tensor.reshape(-1)[0] = float("nan")
    if _rank_matches_env("FAULT_CORRUPT_RANK", env.rank) and tensor.numel() > 0:
        tensor = tensor.clone()
        tensor.reshape(-1)[0] = tensor.reshape(-1)[0] + torch.ones((), device=env.device, dtype=tensor.dtype)
    return tensor


def _dtype(torch: Any, name: str) -> Any:
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


def _sync_time(torch: Any, fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _repeat(fn, iters: int) -> None:
    for _ in range(max(1, iters)):
        fn()


def _tensor_checksum(tensor: Any) -> float:
    return float(tensor.float().sum().detach().cpu().item())


def _nan_inf_counts(torch: Any, tensor: Any) -> tuple[int, int]:
    f = tensor.float()
    return int(torch.isnan(f).sum().item()), int(torch.isinf(f).sum().item())


def _all_gather_object(dist: Any, obj: Any, world_size: int) -> list[Any]:
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


def _summary_from_rank_rows(
    dist: Any,
    env: DistEnv,
    rows: list[dict[str, Any]],
    output_dir: Path,
    group_base: dict[str, Any],
) -> bool:
    gathered = _all_gather_object(dist, rows, env.world_size)
    if env.rank != 0:
        return False

    flat = [row for rank_rows in gathered for row in rank_rows]
    append_jsonl(output_dir / "rank_detail.jsonl", flat)

    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in flat:
        key = (row["op_type"], row["message_size"], row.get("payload_pattern", "none"))
        by_key.setdefault(key, []).append(row)

    summaries = []
    for (op_type, message_size, payload_pattern), values in by_key.items():
        latencies = [v["rank_latency"] for v in values if v.get("rank_latency") is not None]
        errors = [v.get("rank_error_type", "") for v in values if v.get("rank_error_type")]
        nan_count = sum(int(v.get("rank_nan_count", 0)) for v in values)
        inf_count = sum(int(v.get("rank_inf_count", 0)) for v in values)
        checksums = [float(v.get("rank_checksum", 0.0)) for v in values]
        checksum = sum(checksums)
        max_abs_error = max(float(v.get("rank_max_abs_error", 0.0)) for v in values)
        max_rel_error = max(float(v.get("rank_max_rel_error", 0.0)) for v in values)
        latency_p50 = percentile(latencies, 0.50)
        latency_p95 = percentile(latencies, 0.95)
        latency_p99 = percentile(latencies, 0.99)
        msg_bytes = values[0].get("message_bytes", 0)
        elapsed = max(latency_p50, 1e-12)
        algbw = (msg_bytes / elapsed) / 1e9 if msg_bytes else 0.0
        if op_type == "all_reduce":
            busbw = algbw * 2 * max(0, env.world_size - 1) / max(1, env.world_size)
        else:
            busbw = algbw
        summaries.append(
            {
                **group_base,
                "op_type": op_type,
                "message_size": message_size,
                "payload_pattern": payload_pattern,
                "latency_p50": latency_p50,
                "latency_p95": latency_p95,
                "latency_p99": latency_p99,
                "algbw": algbw,
                "busbw": busbw,
                "gemm_tflops": max((float(v.get("gemm_tflops", 0.0)) for v in values), default=0.0),
                "memory_bandwidth": max((float(v.get("memory_bandwidth", 0.0)) for v in values), default=0.0),
                "checksum": checksum,
                "max_abs_error": max_abs_error,
                "max_rel_error": max_rel_error,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "correctness_pass": not errors and nan_count == 0 and inf_count == 0,
                "performance_pass": not errors,
                "timeout": any(v.get("timeout", False) for v in values),
                "error_type": ",".join(sorted(set(errors))),
            }
        )
    append_jsonl(output_dir / "group_summary.jsonl", summaries)
    return any(not row["correctness_pass"] or not row["performance_pass"] for row in summaries)


def _make_tensor(torch: Any, numel: int, dtype: Any, device: Any, seed: int, rank: int) -> Any:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)
    return torch.randn(numel, generator=gen, device=device, dtype=dtype)


def _routing_counts(pattern: str, world_size: int, total_tokens: int, rank: int, seed: int) -> list[int]:
    if pattern == "uniform":
        base = total_tokens // world_size
        counts = [base for _ in range(world_size)]
        counts[-1] += total_tokens - sum(counts)
        return counts
    if pattern == "empty_expert":
        counts = [0 for _ in range(world_size)]
        active = max(1, world_size // 2)
        base = total_tokens // active
        for idx in range(active):
            counts[idx] = base
        counts[active - 1] += total_tokens - sum(counts)
        return counts
    if pattern == "hot_expert":
        counts = [max(1, total_tokens // (world_size * 4)) for _ in range(world_size)]
        counts[rank % world_size] += total_tokens - sum(counts)
        return counts
    if pattern == "skewed":
        weights = [idx + 1 for idx in range(world_size)]
        total = sum(weights)
        counts = [max(0, total_tokens * w // total) for w in weights]
        counts[-1] += total_tokens - sum(counts)
        return counts
    if pattern == "random":
        import random

        rng = random.Random(seed + rank)
        weights = [rng.randint(1, 100) for _ in range(world_size)]
        total = sum(weights)
        counts = [total_tokens * w // total for w in weights]
        counts[-1] += total_tokens - sum(counts)
        return counts
    raise ValueError(f"unsupported MoE pattern: {pattern}")


def run_single_node(
    output_dir: Path,
    dtype_name: str,
    message_sizes: list[int],
    moe_patterns: list[str],
    warmup: int,
    iters: int,
    seed: int,
) -> None:
    _run_distributed_checks(
        output_dir=output_dir,
        dtype_name=dtype_name,
        message_sizes=message_sizes,
        moe_patterns=moe_patterns,
        warmup=warmup,
        iters=iters,
        seed=seed,
        test_round="single_node",
        group_id="",
    )


def ping_group(output_dir: Path, test_round: str, group_id: str) -> None:
    torch, dist, env = init_dist()
    output_dir.mkdir(parents=True, exist_ok=True)
    env.group_id = group_id or env.group_id or f"{test_round}-{env.hostname}"

    start = time.perf_counter()
    x = torch.ones(1, device=env.device, dtype=torch.float32) * (env.rank + 1)
    dist.all_reduce(x)
    dist.barrier(device_ids=[env.local_rank])
    elapsed = time.perf_counter() - start

    row = {
        "rank": env.rank,
        "local_rank": env.local_rank,
        "world_size": env.world_size,
        "local_world_size": env.local_world_size,
        "hostname": env.hostname,
        "group_id": env.group_id,
        "test_round": test_round,
        "dist_backend_requested": env.dist_backend_requested,
        "dist_backend": env.dist_backend,
        "device_vendor": env.device_vendor,
        "comm_runtime": env.comm_runtime,
        "all_reduce_value": float(x.detach().cpu().item()),
        "elapsed_seconds": elapsed,
    }
    gathered = _all_gather_object(dist, row, env.world_size)

    if env.rank == 0:
        expected = env.world_size * (env.world_size + 1) / 2
        pass_check = all(abs(float(item["all_reduce_value"]) - expected) < 1e-5 for item in gathered)
        write_json(
            output_dir / "ping_summary.json",
            {
                "status": "PASS" if pass_check else "FAIL",
                "expected_all_reduce_value": expected,
                "ranks": gathered,
            },
        )
    dist.destroy_process_group()


def run_group(
    output_dir: Path,
    dtype_name: str,
    message_sizes: list[int],
    moe_patterns: list[str],
    warmup: int,
    iters: int,
    seed: int,
    test_round: str,
    group_id: str,
) -> None:
    _run_distributed_checks(
        output_dir=output_dir,
        dtype_name=dtype_name,
        message_sizes=message_sizes,
        moe_patterns=moe_patterns,
        warmup=warmup,
        iters=iters,
        seed=seed,
        test_round=test_round,
        group_id=group_id,
    )


def _write_bandwidth_report(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# All-Reduce Bandwidth Gate Report",
        "",
        f"- input_dir: `{output_dir}`",
        f"- summary_count: {len(summaries)}",
        "",
        "| message_size | iters | min_gate GB/s | avg_gate GB/s | second_lowest_busbw GB/s | avg_busbw GB/s | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['message_size']} | {row['iters']} | "
            f"{row['min_busbw_gate']:.3f} | {row['avg_busbw_gate']:.3f} | "
            f"{row['second_lowest_busbw']:.3f} | {row['avg_busbw']:.3f} | "
            f"{row['bandwidth_pass']} |"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bandwidth_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bandwidth_gate(
    output_dir: Path,
    dtype_name: str,
    message_sizes: list[int],
    warmup: int,
    iters: int,
    seed: int,
    min_busbw: float,
    avg_busbw: float,
    test_round: str,
    group_id: str,
) -> None:
    torch, dist, env = init_dist()
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = _dtype(torch, dtype_name)
    job_id = os.environ.get("HEALTHCHECK_JOB_ID", str(uuid.uuid4()))
    hostnames = sorted(set(_all_gather_object(dist, env.hostname, env.world_size)))
    env.group_id = group_id or env.group_id or f"{test_round}-" + "-".join(hostnames)

    all_summaries: list[dict[str, Any]] = []
    element_size = torch.empty((), dtype=dtype).element_size()

    for size in message_sizes:
        numel = max(1, size // element_size)
        tensor = torch.empty(numel, device=env.device, dtype=dtype)

        for _ in range(max(0, warmup)):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier(device_ids=[env.local_rank])

        local_rows: list[dict[str, Any]] = []
        for idx in range(max(1, iters)):
            _maybe_fault_sleep(env)
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            algbw = (size / max(elapsed, 1e-12)) / 1e9
            busbw = algbw * 2 * max(0, env.world_size - 1) / max(1, env.world_size)
            local_rows.append(
                {
                    "job_id": job_id,
                    "test_round": test_round,
                    "group_id": env.group_id,
                    "hostnames": hostnames,
                    "hostname": env.hostname,
                    "rank": env.rank,
                    "local_rank": env.local_rank,
                    "gpu_id": env.local_rank,
                    "dtype": dtype_name,
                    "dist_backend_requested": env.dist_backend_requested,
                    "dist_backend": env.dist_backend,
                    "device_vendor": env.device_vendor,
                    "comm_runtime": env.comm_runtime,
                    "op_type": "all_reduce",
                    "message_size": size_to_label(size),
                    "message_bytes": size,
                    "round": idx,
                    "rank_latency": elapsed,
                    "rank_algbw": algbw,
                    "rank_busbw": busbw,
                }
            )

        gathered = _all_gather_object(dist, local_rows, env.world_size)
        if env.rank == 0:
            flat = [row for rank_rows in gathered for row in rank_rows]
            append_jsonl(output_dir / "bandwidth_round_detail.jsonl", flat)

            round_busbw: list[float] = []
            round_rows: list[dict[str, Any]] = []
            for idx in range(max(1, iters)):
                values = [row for row in flat if int(row["round"]) == idx]
                elapsed = max(float(row["rank_latency"]) for row in values)
                algbw = (size / max(elapsed, 1e-12)) / 1e9
                busbw = algbw * 2 * max(0, env.world_size - 1) / max(1, env.world_size)
                round_busbw.append(busbw)
                round_rows.append(
                    {
                        "job_id": job_id,
                        "test_round": test_round,
                        "group_id": env.group_id,
                        "hostnames": hostnames,
                        "op_type": "all_reduce",
                        "message_size": size_to_label(size),
                        "message_bytes": size,
                        "dtype": dtype_name,
                        "round": idx,
                        "latency": elapsed,
                        "algbw": algbw,
                        "busbw": busbw,
                    }
                )
            append_jsonl(output_dir / "bandwidth_round_summary.jsonl", round_rows)

            ordered = sorted(round_busbw)
            second_lowest = ordered[1] if len(ordered) >= 2 else ordered[0]
            avg_value = sum(round_busbw) / len(round_busbw)
            passed = second_lowest > min_busbw and avg_value > avg_busbw
            summary = {
                "job_id": job_id,
                "test_round": test_round,
                "group_id": env.group_id,
                "hostnames": hostnames,
                "op_type": "all_reduce",
                "message_size": size_to_label(size),
                "message_bytes": size,
                "dtype": dtype_name,
                "iters": max(1, iters),
                "warmup": max(0, warmup),
                "latency_p50": percentile([row["latency"] for row in round_rows], 0.50),
                "latency_p95": percentile([row["latency"] for row in round_rows], 0.95),
                "latency_p99": percentile([row["latency"] for row in round_rows], 0.99),
                "min_busbw_gate": min_busbw,
                "avg_busbw_gate": avg_busbw,
                "second_lowest_busbw": second_lowest,
                "avg_busbw": avg_value,
                "min_busbw": min(round_busbw),
                "max_busbw": max(round_busbw),
                "bandwidth_pass": passed,
                "correctness_pass": True,
                "performance_pass": passed,
                "error_type": "" if passed else "BandwidthGateFailed",
                "dist_backend_requested": env.dist_backend_requested,
                "dist_backend": env.dist_backend,
                "device_vendor": env.device_vendor,
                "comm_runtime": env.comm_runtime,
            }
            all_summaries.append(summary)
            append_jsonl(output_dir / "bandwidth_summary.jsonl", [summary])

        dist.barrier(device_ids=[env.local_rank])

    if env.rank == 0:
        write_json(
            output_dir / "bandwidth_gate.json",
            {
                "status": "PASS" if all(row["bandwidth_pass"] for row in all_summaries) else "FAIL",
                "summaries": all_summaries,
            },
        )
        _write_bandwidth_report(output_dir, all_summaries)
        failed = [row for row in all_summaries if not row["bandwidth_pass"]]
    else:
        failed = []

    dist.barrier(device_ids=[env.local_rank])
    dist.destroy_process_group()
    if failed:
        raise RuntimeError("all-reduce bandwidth gate failed")


def _run_distributed_checks(
    output_dir: Path,
    dtype_name: str,
    message_sizes: list[int],
    moe_patterns: list[str],
    warmup: int,
    iters: int,
    seed: int,
    test_round: str,
    group_id: str,
) -> None:
    torch, dist, env = init_dist()
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = _dtype(torch, dtype_name)
    job_id = os.environ.get("HEALTHCHECK_JOB_ID", str(uuid.uuid4()))
    hostnames = sorted(set(_all_gather_object(dist, env.hostname, env.world_size)))
    env.group_id = group_id or env.group_id or f"{test_round}-" + "-".join(hostnames)
    group_base = {
        "job_id": job_id,
        "test_round": test_round,
        "group_id": env.group_id,
        "hostnames": hostnames,
        "dtype": dtype_name,
        "dist_backend_requested": env.dist_backend_requested,
        "dist_backend": env.dist_backend,
        "device_vendor": env.device_vendor,
        "comm_runtime": env.comm_runtime,
    }

    rows: list[dict[str, Any]] = []

    # GEMM check.
    for n in [2048, 4096]:
        a = torch.randn((n, n), device=env.device, dtype=dtype)
        b = torch.randn((n, n), device=env.device, dtype=dtype)
        for _ in range(max(1, warmup)):
            _ = a @ b
        elapsed = _sync_time(torch, lambda: _repeat(lambda: (_maybe_fault_sleep(env), a @ b)[1], iters)) / max(1, iters)
        c = _apply_faults(torch, env, a @ b)
        nan_count, inf_count = _nan_inf_counts(torch, c)
        flops = 2.0 * n * n * n
        error_type = "FaultInjectedCorrupt" if _rank_matches_env("FAULT_CORRUPT_RANK", env.rank) else ""
        rows.append(
            _rank_row(
                env,
                op_type="gemm",
                message_size=f"{n}x{n}",
                message_bytes=a.numel() * a.element_size() + b.numel() * b.element_size(),
                payload_pattern="none",
                latency=elapsed,
                checksum=_tensor_checksum(c),
                nan_count=nan_count,
                inf_count=inf_count,
                error_type=error_type,
                gemm_tflops=(flops / elapsed) / 1e12,
            )
        )

    # Memory bandwidth check.
    for size in message_sizes:
        numel = max(1, size // torch.empty((), dtype=dtype).element_size())
        x = _make_tensor(torch, numel, dtype, env.device, seed, env.rank)
        y = torch.empty_like(x)
        for _ in range(max(1, warmup)):
            y.copy_(x)
        elapsed = _sync_time(torch, lambda: _repeat(lambda: (_maybe_fault_sleep(env), y.copy_(x))[1], iters)) / max(1, iters)
        checked = _apply_faults(torch, env, y)
        nan_count, inf_count = _nan_inf_counts(torch, checked)
        error_type = "FaultInjectedCorrupt" if _rank_matches_env("FAULT_CORRUPT_RANK", env.rank) else ""
        rows.append(
            _rank_row(
                env,
                op_type="memory_copy",
                message_size=size_to_label(size),
                message_bytes=size,
                payload_pattern="none",
                latency=elapsed,
                checksum=_tensor_checksum(checked),
                nan_count=nan_count,
                inf_count=inf_count,
                error_type=error_type,
                memory_bandwidth=(size / elapsed) / 1e9,
            )
        )

    # Dense collectives.
    for size in message_sizes:
        numel = max(1, size // torch.empty((), dtype=dtype).element_size())
        for op in ["all_reduce", "reduce_scatter", "all_gather", "broadcast", "all_to_all"]:
            rows.append(_run_collective(torch, dist, env, op, numel, dtype, seed, warmup, iters, size))

        # Variable all_to_allv style MoE payloads.
        for pattern in moe_patterns:
            rows.append(_run_all_to_allv_pattern(torch, dist, env, pattern, numel, dtype, seed, warmup, iters, size))

    has_failed_summary = _summary_from_rank_rows(dist, env, rows, output_dir, group_base)
    dist.barrier(device_ids=[env.local_rank])
    dist.destroy_process_group()
    if has_failed_summary:
        raise RuntimeError("healthcheck group summary contains correctness/performance failures")


def _rank_row(
    env: DistEnv,
    op_type: str,
    message_size: str,
    message_bytes: int,
    payload_pattern: str,
    latency: float,
    checksum: float,
    nan_count: int,
    inf_count: int,
    max_abs_error: float = 0.0,
    max_rel_error: float = 0.0,
    error_type: str = "",
    timeout: bool = False,
    gemm_tflops: float = 0.0,
    memory_bandwidth: float = 0.0,
) -> dict[str, Any]:
    return {
        "group_id": env.group_id or f"group-{env.hostname}",
        "hostname": env.hostname,
        "rank": env.rank,
        "local_rank": env.local_rank,
        "gpu_id": env.local_rank,
        "dist_backend_requested": env.dist_backend_requested,
        "dist_backend": env.dist_backend,
        "device_vendor": env.device_vendor,
        "comm_runtime": env.comm_runtime,
        "op_type": op_type,
        "message_size": message_size,
        "message_bytes": message_bytes,
        "payload_pattern": payload_pattern,
        "rank_latency": latency,
        "rank_checksum": checksum,
        "rank_max_abs_error": max_abs_error,
        "rank_max_rel_error": max_rel_error,
        "rank_nan_count": nan_count,
        "rank_inf_count": inf_count,
        "rank_error_type": error_type,
        "timeout": timeout,
        "gemm_tflops": gemm_tflops,
        "memory_bandwidth": memory_bandwidth,
    }


def _run_collective(
    torch: Any,
    dist: Any,
    env: DistEnv,
    op: str,
    numel: int,
    dtype: Any,
    seed: int,
    warmup: int,
    iters: int,
    message_bytes: int,
) -> dict[str, Any]:
    x = _make_tensor(torch, numel, dtype, env.device, seed, env.rank)
    error_type = ""

    def once() -> Any:
        if op == "all_reduce":
            y = x.clone()
            dist.all_reduce(y)
            return y
        if op == "reduce_scatter":
            if numel % env.world_size != 0:
                padded = ((numel + env.world_size - 1) // env.world_size) * env.world_size
                inp = torch.zeros(padded, device=env.device, dtype=dtype)
                inp[:numel].copy_(x)
            else:
                inp = x
            chunks = list(inp.chunk(env.world_size))
            out = torch.empty_like(chunks[env.rank])
            dist.reduce_scatter(out, chunks)
            return out
        if op == "all_gather":
            out = [torch.empty_like(x) for _ in range(env.world_size)]
            dist.all_gather(out, x)
            return torch.cat(out)
        if op == "broadcast":
            y = x.clone()
            dist.broadcast(y, src=0)
            return y
        if op == "all_to_all":
            if numel % env.world_size != 0:
                padded = ((numel + env.world_size - 1) // env.world_size) * env.world_size
                inp = torch.zeros(padded, device=env.device, dtype=dtype)
                inp[:numel].copy_(x)
            else:
                inp = x
            out = torch.empty_like(inp)
            split = inp.numel() // env.world_size
            dist.all_to_all_single(
                out,
                inp,
                output_split_sizes=[split for _ in range(env.world_size)],
                input_split_sizes=[split for _ in range(env.world_size)],
            )
            return out
        raise ValueError(op)

    try:
        for _ in range(max(1, warmup)):
            once()
        elapsed = _sync_time(torch, lambda: _repeat(lambda: (_maybe_fault_sleep(env), once())[1], iters)) / max(1, iters)
        y = _apply_faults(torch, env, once())
        nan_count, inf_count = _nan_inf_counts(torch, y)
        checksum = _tensor_checksum(y)
        if _rank_matches_env("FAULT_CORRUPT_RANK", env.rank):
            error_type = "FaultInjectedCorrupt"
    except Exception as exc:
        elapsed = 0.0
        nan_count = 0
        inf_count = 0
        checksum = 0.0
        error_type = type(exc).__name__
    return _rank_row(
        env,
        op_type=op,
        message_size=size_to_label(message_bytes),
        message_bytes=message_bytes,
        payload_pattern="none",
        latency=elapsed,
        checksum=checksum,
        nan_count=nan_count,
        inf_count=inf_count,
        error_type=error_type,
    )


def _run_all_to_allv_pattern(
    torch: Any,
    dist: Any,
    env: DistEnv,
    pattern: str,
    numel: int,
    dtype: Any,
    seed: int,
    warmup: int,
    iters: int,
    message_bytes: int,
) -> dict[str, Any]:
    # all_to_all_single supports split sizes on modern PyTorch/NCCL.
    token_size = max(1, numel // max(1, env.world_size))
    send_counts = _routing_counts(pattern, env.world_size, token_size * env.world_size, env.rank, seed)
    input_split_sizes = send_counts
    gathered_counts = [None for _ in range(env.world_size)]
    dist.all_gather_object(gathered_counts, input_split_sizes)
    output_split_sizes = [counts[env.rank] for counts in gathered_counts]

    total_in = sum(input_split_sizes)
    total_out = sum(output_split_sizes)
    x = _make_tensor(torch, total_in, dtype, env.device, seed, env.rank)
    out = torch.empty(total_out, device=env.device, dtype=dtype)
    error_type = ""

    def once() -> Any:
        y = torch.empty_like(out)
        dist.all_to_all_single(
            y,
            x,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )
        return y

    try:
        for _ in range(max(1, warmup)):
            once()
        elapsed = _sync_time(torch, lambda: _repeat(lambda: (_maybe_fault_sleep(env), once())[1], iters)) / max(1, iters)
        y = _apply_faults(torch, env, once())
        nan_count, inf_count = _nan_inf_counts(torch, y)
        checksum = _tensor_checksum(y)
        if _rank_matches_env("FAULT_CORRUPT_RANK", env.rank):
            error_type = "FaultInjectedCorrupt"
    except Exception as exc:
        elapsed = 0.0
        nan_count = 0
        inf_count = 0
        checksum = 0.0
        error_type = type(exc).__name__
    return _rank_row(
        env,
        op_type="all_to_allv",
        message_size=size_to_label(message_bytes),
        message_bytes=message_bytes,
        payload_pattern=pattern,
        latency=elapsed,
        checksum=checksum,
        nan_count=nan_count,
        inf_count=inf_count,
        error_type=error_type,
    )
