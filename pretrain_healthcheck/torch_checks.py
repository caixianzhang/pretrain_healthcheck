from __future__ import annotations

import os
import pickle
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
    pod_name: str
    node_name: str


def _import_torch():
    try:
        import torch
        import torch.distributed as dist

        return torch, dist
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(f"PyTorch with distributed support is required: {exc}") from exc


def _resolve_dist_backend() -> tuple[str, str]:
    if _dynamic_pre_fault_matches("backend_fail"):
        requested = os.environ.get("DIST_BACKEND", "nccl").strip().lower()
        return requested, "__dynamic_fault_invalid_backend__"
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


def _device_api(torch: Any, backend: str, vendor: str) -> tuple[str, Any]:
    if backend == "hccl" or vendor in {"ascend", "npu"}:
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise RuntimeError("torch_npu is required for Ascend/NPU dynamic checks")
        return "npu", npu
    return "cuda", torch.cuda


def _synchronize(torch: Any) -> None:
    npu = getattr(torch, "npu", None)
    if npu is not None:
        try:
            if npu.is_available():
                npu.synchronize()
                return
        except Exception:
            pass
    torch.cuda.synchronize()


def _timing_event_api(torch: Any, env: DistEnv) -> Any | None:
    if env.dist_backend == "hccl" or env.device_vendor in {"ascend", "npu"}:
        return getattr(torch, "npu", None)
    return getattr(torch, "cuda", None)


def _new_timing_event(torch: Any, env: DistEnv) -> Any | None:
    api = _timing_event_api(torch, env)
    event_type = getattr(api, "Event", None) if api is not None else None
    if event_type is None:
        return None
    try:
        return event_type(enable_timing=True)
    except TypeError:
        return event_type()


def _steady_state_timings(
    torch: Any,
    dist: Any,
    env: DistEnv,
    operation: Any,
    iters: int,
    measurement_batches: int | None = None,
) -> tuple[list[float], str]:
    """Measure continuous collective loops; allocation and warmup stay outside."""
    batch_count = max(
        1,
        int(
            measurement_batches
            if measurement_batches is not None
            else os.environ.get("DYNAMIC_COMPARE_MEASUREMENT_BATCHES", "1")
        ),
    )
    iteration_count = max(1, iters)
    latencies: list[float] = []
    timing_mode = "steady_state_device_event"
    for _ in range(batch_count):
        _synchronize(torch)
        dist.barrier(device_ids=[env.local_rank])
        start_event = _new_timing_event(torch, env)
        end_event = _new_timing_event(torch, env)
        host_start = time.perf_counter()
        if start_event is not None and end_event is not None:
            start_event.record()
        for _idx in range(iteration_count):
            _maybe_fault_sleep(env)
            operation()
        if start_event is not None and end_event is not None:
            end_event.record()
        _synchronize(torch)
        if start_event is not None and end_event is not None:
            elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
        else:
            timing_mode = "steady_state_host_fallback"
            elapsed = time.perf_counter() - host_start
        latencies.append(elapsed / iteration_count)
    return latencies, timing_mode


def _diagnostic_round_timings(
    torch: Any,
    env: DistEnv,
    operation: Any,
    iters: int,
) -> list[float]:
    """Per-round synchronized timing used only by targeted diagnostic retests."""
    latencies: list[float] = []
    for _ in range(max(1, iters)):
        _synchronize(torch)
        started = time.perf_counter()
        _maybe_fault_sleep(env)
        operation()
        _synchronize(torch)
        latencies.append(time.perf_counter() - started)
    return latencies


def init_dist() -> tuple[Any, Any, DistEnv]:
    _maybe_fault_before_dist_init()
    torch, dist = _import_torch()
    backend_requested, backend = _resolve_dist_backend()
    device_vendor, comm_runtime = _runtime_meta(backend)
    _maybe_import_backend_extension(backend)
    device_type, device_api = _device_api(torch, backend, device_vendor)
    if not device_api.is_available():
        raise RuntimeError(f"{device_type.upper()} is not available")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(device_api.device_count())))
    device_api.set_device(local_rank)
    device = torch.device(device_type, local_rank)
    group_id = os.environ.get("HEALTHCHECK_GROUP_ID") or os.environ.get("HC_GROUP_ID") or ""
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
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        device=device,
        hostname=hostname(),
        dist_backend_requested=backend_requested,
        dist_backend=backend,
        device_vendor=device_vendor,
        comm_runtime=comm_runtime,
        group_id=group_id,
        pod_name=os.environ.get("HC_POD_NAME", ""),
        node_name=os.environ.get("HC_NODE_NAME", ""),
    )


def _rank_matches_env(env_name: str, rank: int) -> bool:
    value = os.environ.get(env_name, "").strip()
    return bool(value) and value == str(rank)


def _name_matches_env(env_name: str, value: str) -> bool:
    target = os.environ.get(env_name, "").strip()
    return bool(target) and bool(value) and target == value


def _value_matches_csv_env(env_name: str, value: str | int) -> bool:
    actual = str(value).strip()
    if not actual:
        return False
    targets = {item.strip() for item in os.environ.get(env_name, "").split(",") if item.strip()}
    return actual in targets


def _fault_target_matches(prefix: str, env: DistEnv) -> bool:
    return (
        _rank_matches_env(f"{prefix}_RANK", env.rank)
        or _value_matches_csv_env(f"{prefix}_RANKS", env.rank)
        or _name_matches_env(f"{prefix}_POD", env.pod_name)
        or _value_matches_csv_env(f"{prefix}_PODS", env.pod_name)
        or _name_matches_env(f"{prefix}_NODE", env.node_name)
        or _value_matches_csv_env(f"{prefix}_NODES", env.node_name)
    )


def _pre_dist_fault_target_matches(prefix: str) -> bool:
    rank = int(os.environ.get("RANK", "-1"))
    return (
        _rank_matches_env(f"{prefix}_RANK", rank)
        or _value_matches_csv_env(f"{prefix}_RANKS", rank)
        or _name_matches_env(f"{prefix}_POD", os.environ.get("HC_POD_NAME", ""))
        or _value_matches_csv_env(f"{prefix}_PODS", os.environ.get("HC_POD_NAME", ""))
        or _name_matches_env(f"{prefix}_NODE", os.environ.get("HC_NODE_NAME", ""))
        or _value_matches_csv_env(f"{prefix}_NODES", os.environ.get("HC_NODE_NAME", ""))
    )


def _dynamic_fault_type() -> str:
    return os.environ.get("DYNAMIC_FAULT_TYPE", "").strip().lower().replace("-", "_")


def _dynamic_pod_target_matches() -> bool:
    pod_target = os.environ.get("DYNAMIC_FAULT_POD", "").strip()
    node_target = os.environ.get("DYNAMIC_FAULT_NODE", "").strip()
    if pod_target and pod_target != os.environ.get("HC_POD_NAME", "").strip():
        return False
    if node_target and node_target != os.environ.get("HC_NODE_NAME", "").strip():
        return False
    return True


def _dynamic_local_rank_matches(local_rank: int) -> bool:
    target = os.environ.get("DYNAMIC_FAULT_LOCAL_RANK", os.environ.get("DYNAMIC_FAULT_RANK", "")).strip()
    return not target or target == str(local_rank)


def _dynamic_pre_fault_matches(fault_type: str) -> bool:
    if _dynamic_fault_type() != fault_type:
        return False
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    except ValueError:
        local_rank = -1
    return _dynamic_pod_target_matches() and _dynamic_local_rank_matches(local_rank)


def _dynamic_fault_matches(env: DistEnv, fault_type: str) -> bool:
    return (
        _dynamic_fault_type() == fault_type
        and _dynamic_pod_target_matches()
        and _dynamic_local_rank_matches(env.local_rank)
    )


def _maybe_fault_before_dist_init() -> None:
    if _dynamic_pre_fault_matches("sleep_timeout"):
        time.sleep(float(os.environ.get("DYNAMIC_FAULT_SLEEP_SECONDS", os.environ.get("FAULT_SLEEP_SECONDS", "300"))))
    if _pre_dist_fault_target_matches("FAULT_COMM_ENV_BAD"):
        os.environ["MCCL_IB_HCA"] = "__fault_bad_hca__"
        os.environ["MCCL_IB_GID_INDEX"] = "999"
        os.environ["MCCL_SOCKET_IFNAME"] = "__fault_bad_ifname__"
        os.environ["NCCL_IB_HCA"] = "__fault_bad_hca__"
        os.environ["NCCL_SOCKET_IFNAME"] = "__fault_bad_ifname__"
        os.environ["HCCL_SOCKET_IFNAME"] = "__fault_bad_ifname__"
        os.environ["GLOO_SOCKET_IFNAME"] = "__fault_bad_ifname__"
    if _pre_dist_fault_target_matches("FAULT_ETH_FALLBACK"):
        os.environ["MCCL_SOCKET_IFNAME"] = "eth0"
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
        os.environ["HCCL_SOCKET_IFNAME"] = "eth0"
        os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["MCCL_IB_DISABLE"] = "1"
    if _pre_dist_fault_target_matches("FAULT_JOIN_TIMEOUT"):
        seconds = float(os.environ.get("FAULT_JOIN_TIMEOUT_SECONDS", os.environ.get("FAULT_SLEEP_SECONDS", "300")))
        time.sleep(seconds)
    if _pre_dist_fault_target_matches("FAULT_RANK_EXIT"):
        raise SystemExit(int(os.environ.get("FAULT_RANK_EXIT_CODE", "17")))


def _maybe_fault_sleep(env: DistEnv) -> None:
    if _dynamic_fault_matches(env, "sleep_timeout"):
        time.sleep(float(os.environ.get("DYNAMIC_FAULT_SLEEP_SECONDS", os.environ.get("FAULT_SLEEP_SECONDS", "300"))))
    if _dynamic_fault_matches(env, "slow_rank"):
        time.sleep(float(os.environ.get("DYNAMIC_FAULT_SLEEP_SECONDS", "0.2")))
    if _fault_target_matches("FAULT_SLEEP", env):
        time.sleep(float(os.environ.get("FAULT_SLEEP_SECONDS", "30")))
    if _fault_target_matches("FAULT_NET_SLOW", env):
        time.sleep(float(os.environ.get("FAULT_NET_SLOW_SECONDS", "0.2")))


def _apply_faults(torch: Any, env: DistEnv, tensor: Any) -> Any:
    if _dynamic_fault_matches(env, "nan") and tensor.numel() > 0:
        tensor = tensor.clone()
        tensor.reshape(-1)[0] = float("nan")
    if _dynamic_fault_matches(env, "corrupt") and tensor.numel() > 0:
        tensor = tensor.clone()
        tensor.reshape(-1)[0] = tensor.reshape(-1)[0] + torch.ones((), device=env.device, dtype=tensor.dtype)
    if _fault_target_matches("FAULT_NAN", env) and tensor.numel() > 0:
        tensor = tensor.clone()
        tensor.reshape(-1)[0] = float("nan")
    if _fault_target_matches("FAULT_CORRUPT", env) and tensor.numel() > 0:
        tensor = tensor.clone()
        tensor.reshape(-1)[0] = tensor.reshape(-1)[0] + torch.ones((), device=env.device, dtype=tensor.dtype)
    return tensor


def _fault_error_type(env: DistEnv, default: str = "") -> str:
    if _dynamic_fault_matches(env, "nan"):
        return "DynamicFaultInjectedNaN"
    if _dynamic_fault_matches(env, "corrupt"):
        return "DynamicFaultInjectedCorrupt"
    if _dynamic_fault_matches(env, "sleep_timeout"):
        return "DynamicFaultInjectedSleepTimeout"
    if _fault_target_matches("FAULT_NAN", env):
        return "FaultInjectedNaN"
    if _fault_target_matches("FAULT_CORRUPT", env):
        return "FaultInjectedCorrupt"
    if _fault_target_matches("FAULT_NET_SLOW", env):
        return "FaultInjectedNetSlow"
    return default


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
    _synchronize(torch)
    start = time.perf_counter()
    fn()
    _synchronize(torch)
    return time.perf_counter() - start


def _repeat(fn, iters: int) -> None:
    for _ in range(max(1, iters)):
        fn()


def _tensor_checksum(tensor: Any) -> float:
    return float(tensor.float().sum().detach().cpu().item())


def _nan_inf_counts(torch: Any, tensor: Any) -> tuple[int, int]:
    f = tensor.float()
    return int(torch.isnan(f).sum().item()), int(torch.isinf(f).sum().item())


def _object_packet(obj: Any) -> bytes:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("serialized healthcheck metadata exceeds 4 GiB")
    return len(payload).to_bytes(4, "big") + payload


def _object_from_packet(packet: bytes) -> Any:
    if len(packet) < 4:
        raise ValueError("healthcheck metadata packet is missing its length header")
    payload_size = int.from_bytes(packet[:4], "big")
    if payload_size > len(packet) - 4:
        raise ValueError("healthcheck metadata packet is truncated")
    return pickle.loads(packet[4 : 4 + payload_size])


def _all_gather_object(dist: Any, obj: Any, world_size: int) -> list[Any]:
    """Gather bounded metadata through fixed-size accelerator tensors.

    PyTorch's variable-length object collective has produced corrupted object
    sizes on large MCCL groups. A fixed uint8 packet keeps the size protocol
    explicit and avoids allocating from an untrusted decoded length.
    """
    torch, _ = _import_torch()
    backend = str(dist.get_backend()).lower()
    vendor = os.environ.get("DEVICE_VENDOR", "").strip().lower()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_type = "npu" if backend == "hccl" or vendor in {"ascend", "npu"} else "cuda"
    device = torch.device(device_type, local_rank)
    packet = _object_packet(obj)
    max_bytes = int(os.environ.get("HC_OBJECT_GATHER_MAX_BYTES", str(16 * 1024 * 1024)))
    if len(packet) > max_bytes:
        raise ValueError(f"healthcheck metadata packet is {len(packet)} bytes; limit is {max_bytes}")

    packet_size = torch.tensor([len(packet)], dtype=torch.int32, device=device)
    dist.all_reduce(packet_size, op=dist.ReduceOp.MAX)
    padded_size = int(packet_size.item())
    if padded_size <= 0 or padded_size > max_bytes:
        raise RuntimeError(f"invalid gathered healthcheck metadata size: {padded_size}")

    send = torch.zeros(padded_size, dtype=torch.uint8, device=device)
    send[: len(packet)].copy_(torch.tensor(list(packet), dtype=torch.uint8, device=device))
    received = [torch.empty_like(send) for _ in range(world_size)]
    dist.all_gather(received, send)
    if int(dist.get_rank()) != 0:
        return []
    return [_object_from_packet(bytes(tensor.cpu().tolist())) for tensor in received]


def _comm_env_snapshot() -> dict[str, str]:
    prefixes = (
        "MCCL_",
        "NCCL_",
        "HCCL_",
        "ASCEND_",
        "GLOO_",
    )
    names = {
        "DIST_BACKEND",
        "DEVICE_VENDOR",
        "COMM_RUNTIME",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "CUDA_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
    }
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key in names or key.endswith("_SOCKET_IFNAME") or key.endswith("_IB_HCA") or key.startswith(prefixes)
    }


def _write_comm_path_debug(dist: Any, env: DistEnv, output_dir: Path, label: str) -> None:
    if os.environ.get("COMM_PATH_DEBUG", "").lower() not in {"1", "true", "yes", "on"}:
        return
    row = {
        "label": label,
        "rank": env.rank,
        "local_rank": env.local_rank,
        "world_size": env.world_size,
        "hostname": env.hostname,
        "pod_name": env.pod_name,
        "node_name": env.node_name,
        "dist_backend_requested": env.dist_backend_requested,
        "dist_backend": env.dist_backend,
        "device_vendor": env.device_vendor,
        "comm_runtime": env.comm_runtime,
        "env": _comm_env_snapshot(),
    }
    gathered = _all_gather_object(dist, row, env.world_size)
    if env.rank != 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [item for item in gathered if isinstance(item, dict)]
    append_jsonl(output_dir / "comm_path_summary.jsonl", rows)
    write_json(
        output_dir / "comm_path_summary.json",
        {
            "schema_version": 1,
            "label": label,
            "rank_count": len(rows),
            "nodes": sorted({str(item.get("node_name", "")) for item in rows}),
            "rows": rows,
        },
    )


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

        # Build one deterministic random distribution and rotate it by source
        # rank. This keeps per-rank routes different while allowing every peer
        # to reconstruct receive splits locally without a metadata collective.
        rng = random.Random(seed)
        weights = [rng.randint(1, 100) for _ in range(world_size)]
        total = sum(weights)
        base_counts = [total_tokens * w // total for w in weights]
        base_counts[-1] += total_tokens - sum(base_counts)
        shift = rank % world_size
        return base_counts[-shift:] + base_counts[:-shift] if shift else base_counts
    raise ValueError(f"unsupported MoE pattern: {pattern}")


def _routing_output_counts(
    pattern: str,
    world_size: int,
    total_tokens: int,
    destination_rank: int,
    seed: int,
) -> list[int]:
    """Reconstruct one destination's receive splits in source-rank order."""
    if not 0 <= destination_rank < world_size:
        raise ValueError(f"destination rank {destination_rank} is outside world size {world_size}")
    if pattern in {"uniform", "empty_expert", "skewed"}:
        count = _routing_counts(pattern, world_size, total_tokens, 0, seed)[destination_rank]
        return [count for _ in range(world_size)]
    if pattern == "hot_expert":
        base = max(1, total_tokens // (world_size * 4))
        remainder = total_tokens - base * world_size
        counts = [base for _ in range(world_size)]
        counts[destination_rank] += remainder
        return counts
    if pattern == "random":
        base_counts = _routing_counts(pattern, world_size, total_tokens, 0, seed)
        return [base_counts[(destination_rank - source_rank) % world_size] for source_rank in range(world_size)]
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
        _synchronize(torch)
        dist.barrier(device_ids=[env.local_rank])

        local_latencies, timing_mode = _steady_state_timings(
            torch,
            dist,
            env,
            lambda: dist.all_reduce(tensor),
            iters,
        )
        local_rows: list[dict[str, Any]] = []
        for idx, elapsed in enumerate(local_latencies):
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
                    "measurement_batch": idx,
                    "iterations_per_batch": max(1, iters),
                    "timing_mode": timing_mode,
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
            for idx in range(len(local_latencies)):
                values = [row for row in flat if int(row["measurement_batch"]) == idx]
                slowest = max(values, key=lambda row: float(row["rank_latency"]))
                elapsed = float(slowest["rank_latency"])
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
                        "measurement_batch": idx,
                        "iterations_per_batch": max(1, iters),
                        "timing_mode": timing_mode,
                        "slowest_rank": int(slowest["rank"]),
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
                "collective_group_size": env.world_size,
                "measurement_batches": len(round_rows),
                "iterations_per_batch": max(1, iters),
                "timing_mode": timing_mode,
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
    # Performance gates are classification inputs, not process/correctness failures.


def _busbw_factor(op_type: str, world_size: int) -> float:
    if op_type == "all_reduce":
        return 2 * max(0, world_size - 1) / max(1, world_size)
    if op_type in {"reduce_scatter", "all_gather", "all_to_all", "all_to_allv"}:
        return max(0, world_size - 1) / max(1, world_size)
    return 1.0


def _write_collective_bandwidth_report(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Collective Bandwidth Gate Report",
        "",
        f"- input_dir: `{output_dir}`",
        f"- summary_count: {len(summaries)}",
        "",
        "| op | pattern | group_size | message_size | iters | min_gate GB/s | avg_gate GB/s | second_lowest_busbw GB/s | avg_busbw GB/s | pass |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['op_type']} | {row.get('payload_pattern', 'none')} | {row['collective_group_size']} | "
            f"{row['message_size']} | {row['iters']} | "
            f"{row['min_busbw_gate']:.3f} | {row['avg_busbw_gate']:.3f} | "
            f"{row['second_lowest_busbw']:.3f} | {row['avg_busbw']:.3f} | "
            f"{row['bandwidth_pass']} |"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "collective_bandwidth_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_ep_group(dist: Any, env: DistEnv, ep_size: int) -> tuple[Any, int, int, list[int]]:
    group_size = env.world_size if ep_size <= 0 else min(ep_size, env.world_size)
    if env.world_size % group_size != 0:
        raise RuntimeError(f"world_size={env.world_size} must be divisible by ep_size={group_size}")
    selected_group = None
    selected_ranks: list[int] = []
    selected_group_rank = 0
    for group_start in range(0, env.world_size, group_size):
        ranks = list(range(group_start, group_start + group_size))
        group = dist.new_group(ranks=ranks)
        if env.rank in ranks:
            selected_group = group
            selected_ranks = ranks
            selected_group_rank = env.rank - group_start
    if selected_group is None:
        raise RuntimeError(f"rank {env.rank} did not join any EP group")
    return selected_group, selected_group_rank, group_size, selected_ranks


@dataclass
class _CollectiveBandwidthWorkspace:
    input_tensor: Any
    output_tensor: Any | None = None
    output_tensors: list[Any] | None = None
    input_chunks: list[Any] | None = None
    input_split_sizes: list[int] | None = None
    output_split_sizes: list[int] | None = None


def _prepare_collective_bandwidth_workspace(
    torch: Any,
    env: DistEnv,
    op: str,
    tensor_numel: int,
    group_rank: int,
    group_size: int,
    dtype: Any,
    input_split_sizes: list[int] | None = None,
    output_split_sizes: list[int] | None = None,
) -> _CollectiveBandwidthWorkspace:
    # Keep buffer allocation outside warmup and timed collective iterations.
    if op in {"all_reduce", "broadcast"}:
        return _CollectiveBandwidthWorkspace(
            input_tensor=torch.zeros(tensor_numel, device=env.device, dtype=dtype)
        )
    if op == "reduce_scatter":
        padded_numel = ((tensor_numel + group_size - 1) // group_size) * group_size
        input_tensor = torch.zeros(padded_numel, device=env.device, dtype=dtype)
        input_chunks = list(input_tensor.chunk(group_size))
        return _CollectiveBandwidthWorkspace(
            input_tensor=input_tensor,
            output_tensor=torch.empty_like(input_chunks[group_rank]),
            input_chunks=input_chunks,
        )
    if op == "all_gather":
        input_tensor = torch.zeros(tensor_numel, device=env.device, dtype=dtype)
        return _CollectiveBandwidthWorkspace(
            input_tensor=input_tensor,
            output_tensors=[torch.empty_like(input_tensor) for _ in range(group_size)],
        )
    if op == "all_to_all":
        padded_numel = ((tensor_numel + group_size - 1) // group_size) * group_size
        input_tensor = torch.zeros(padded_numel, device=env.device, dtype=dtype)
        return _CollectiveBandwidthWorkspace(
            input_tensor=input_tensor,
            output_tensor=torch.empty_like(input_tensor),
        )
    if op == "all_to_allv":
        if input_split_sizes is None or output_split_sizes is None:
            raise RuntimeError("all_to_allv requires split sizes")
        return _CollectiveBandwidthWorkspace(
            input_tensor=torch.zeros(sum(input_split_sizes), device=env.device, dtype=dtype),
            output_tensor=torch.empty(sum(output_split_sizes), device=env.device, dtype=dtype),
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
        )
    raise ValueError(op)


def _collective_bandwidth_once(
    dist: Any,
    op: str,
    workspace: _CollectiveBandwidthWorkspace,
    group: Any | None = None,
) -> None:
    if op == "all_reduce":
        dist.all_reduce(workspace.input_tensor, group=group)
        return
    if op == "broadcast":
        dist.broadcast(workspace.input_tensor, src=0, group=group)
        return
    if op == "reduce_scatter":
        if workspace.output_tensor is None or workspace.input_chunks is None:
            raise RuntimeError("reduce_scatter workspace is incomplete")
        dist.reduce_scatter(workspace.output_tensor, workspace.input_chunks, group=group)
        return
    if op == "all_gather":
        if workspace.output_tensors is None:
            raise RuntimeError("all_gather workspace is incomplete")
        dist.all_gather(workspace.output_tensors, workspace.input_tensor, group=group)
        return
    if op == "all_to_all":
        if workspace.output_tensor is None:
            raise RuntimeError("all_to_all workspace is incomplete")
        # Let the backend use its native equal-split path. Passing a large
        # explicit split vector has triggered MCCL kernel faults at 512 ranks.
        dist.all_to_all_single(workspace.output_tensor, workspace.input_tensor, group=group)
        return
    if op == "all_to_allv":
        if workspace.output_tensor is None:
            raise RuntimeError(f"{op} workspace is incomplete")
        dist.all_to_all_single(
            workspace.output_tensor,
            workspace.input_tensor,
            output_split_sizes=workspace.output_split_sizes,
            input_split_sizes=workspace.input_split_sizes,
            group=group,
        )
        return
    raise ValueError(op)


def _collective_bandwidth_case(
    torch: Any,
    dist: Any,
    env: DistEnv,
    output_dir: Path,
    job_id: str,
    test_round: str,
    group_id: str,
    hostnames: list[str],
    dtype_name: str,
    dtype: Any,
    op: str,
    message_size: int,
    warmup: int,
    iters: int,
    seed: int,
    min_busbw: float,
    avg_busbw: float,
    payload_pattern: str = "none",
    collective_group: Any | None = None,
    collective_group_rank: int | None = None,
    collective_group_size: int | None = None,
    diagnostic_iters: int = 0,
    source_stage: str = "collective_bandwidth",
) -> dict[str, Any] | None:
    collective_group_rank = env.rank if collective_group_rank is None else collective_group_rank
    collective_group_size = env.world_size if collective_group_size is None else collective_group_size
    element_size = torch.empty((), dtype=dtype).element_size()
    numel = max(1, message_size // element_size)

    input_split_sizes = None
    output_split_sizes = None
    effective_message_size = message_size
    if op == "all_gather":
        tensor_numel = max(1, numel // max(1, collective_group_size))
        effective_message_size = tensor_numel * element_size * collective_group_size
    elif op == "all_to_allv":
        token_size = max(1, numel // max(1, collective_group_size))
        input_split_sizes = _routing_counts(
            payload_pattern,
            collective_group_size,
            token_size * collective_group_size,
            collective_group_rank,
            seed,
        )
        output_split_sizes = _routing_output_counts(
            payload_pattern,
            collective_group_size,
            token_size * collective_group_size,
            collective_group_rank,
            seed,
        )
        tensor_numel = sum(input_split_sizes)
        effective_message_size = tensor_numel * element_size
    else:
        tensor_numel = numel

    workspace = _prepare_collective_bandwidth_workspace(
        torch,
        env,
        op,
        tensor_numel,
        collective_group_rank,
        collective_group_size,
        dtype,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
    )
    for _ in range(max(0, warmup)):
        _collective_bandwidth_once(
            dist,
            op,
            workspace,
            group=collective_group,
        )
    _synchronize(torch)
    dist.barrier(device_ids=[env.local_rank])

    def collective_once() -> None:
        _collective_bandwidth_once(
            dist,
            op,
            workspace,
            group=collective_group,
        )

    local_latencies, timing_mode = _steady_state_timings(
        torch,
        dist,
        env,
        collective_once,
        iters,
    )
    diagnostic_latencies = (
        _diagnostic_round_timings(torch, env, collective_once, diagnostic_iters)
        if diagnostic_iters > 0
        else []
    )
    local_rows: list[dict[str, Any]] = []
    for idx, elapsed in enumerate(local_latencies):
        algbw = (effective_message_size / max(elapsed, 1e-12)) / 1e9
        busbw = algbw * _busbw_factor(op, collective_group_size)
        local_rows.append(
            {
                "job_id": job_id,
                "test_round": test_round,
                "group_id": group_id,
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
                "op_type": op,
                "payload_pattern": payload_pattern,
                "message_size": size_to_label(message_size),
                "message_bytes": effective_message_size,
                "requested_message_bytes": message_size,
                "round": idx,
                "measurement_batch": idx,
                "iterations_per_batch": max(1, iters),
                "timing_mode": timing_mode,
                "collective_group_rank": collective_group_rank,
                "collective_group_size": collective_group_size,
                "rank_latency": elapsed,
                "rank_algbw": algbw,
                "rank_busbw": busbw,
                "diagnostic_rank_latencies": diagnostic_latencies,
            }
        )

    gathered = _all_gather_object(dist, local_rows, env.world_size)
    if env.rank != 0:
        return None

    flat = [row for rank_rows in gathered for row in rank_rows]
    append_jsonl(output_dir / "collective_bandwidth_round_detail.jsonl", flat)

    round_busbw: list[float] = []
    round_rows: list[dict[str, Any]] = []
    for idx in range(len(local_latencies)):
        values = [row for row in flat if int(row["measurement_batch"]) == idx]
        slowest = max(values, key=lambda row: float(row["rank_latency"]))
        elapsed = float(slowest["rank_latency"])
        algbw = (effective_message_size / max(elapsed, 1e-12)) / 1e9
        busbw = algbw * _busbw_factor(op, collective_group_size)
        round_busbw.append(busbw)
        round_rows.append(
            {
                "job_id": job_id,
                "test_round": test_round,
                "group_id": group_id,
                "hostnames": hostnames,
                "op_type": op,
                "payload_pattern": payload_pattern,
                "message_size": size_to_label(message_size),
                "message_bytes": effective_message_size,
                "requested_message_bytes": message_size,
                "dtype": dtype_name,
                "round": idx,
                "measurement_batch": idx,
                "iterations_per_batch": max(1, iters),
                "timing_mode": timing_mode,
                "slowest_rank": int(slowest["rank"]),
                "collective_group_size": collective_group_size,
                "latency": elapsed,
                "algbw": algbw,
                "busbw": busbw,
            }
        )
    append_jsonl(output_dir / "collective_bandwidth_round_summary.jsonl", round_rows)

    ordered = sorted(round_busbw)
    second_lowest = ordered[1] if len(ordered) >= 2 else ordered[0]
    avg_value = sum(round_busbw) / len(round_busbw)
    diagnostic_group_latencies: list[float] = []
    diagnostic_slowest_ranks: list[int] = []
    for idx in range(diagnostic_iters):
        slowest = max(flat, key=lambda row: float((row.get("diagnostic_rank_latencies") or [])[idx]))
        diagnostic_group_latencies.append(float(slowest["diagnostic_rank_latencies"][idx]))
        diagnostic_slowest_ranks.append(int(slowest["rank"]))
    passed = second_lowest > min_busbw and avg_value > avg_busbw
    summary = {
        "job_id": job_id,
        "test_round": test_round,
        "group_id": group_id,
        "hostnames": hostnames,
        "op_type": op,
        "source_stage": source_stage,
        "payload_pattern": payload_pattern,
        "message_size": size_to_label(message_size),
        "message_bytes": effective_message_size,
        "requested_message_bytes": message_size,
        "dtype": dtype_name,
        "iters": max(1, iters),
        "warmup": max(0, warmup),
        "measurement_batches": len(round_rows),
        "iterations_per_batch": max(1, iters),
        "timing_mode": timing_mode,
        "collective_group_size": collective_group_size,
        "latency_p50": percentile([row["latency"] for row in round_rows], 0.50),
        "latency_p95": percentile([row["latency"] for row in round_rows], 0.95),
        "latency_p99": percentile([row["latency"] for row in round_rows], 0.99),
        "min_busbw_gate": min_busbw,
        "avg_busbw_gate": avg_busbw,
        "second_lowest_busbw": second_lowest,
        "avg_busbw": avg_value,
        "min_busbw": min(round_busbw),
        "max_busbw": max(round_busbw),
        "diagnostic_timing_mode": "per_round_synchronized_host" if diagnostic_group_latencies else "",
        "diagnostic_latency_p50": percentile(diagnostic_group_latencies, 0.50) if diagnostic_group_latencies else None,
        "diagnostic_latency_p95": percentile(diagnostic_group_latencies, 0.95) if diagnostic_group_latencies else None,
        "diagnostic_latency_p99": percentile(diagnostic_group_latencies, 0.99) if diagnostic_group_latencies else None,
        "diagnostic_slowest_ranks": diagnostic_slowest_ranks,
        "bandwidth_pass": passed,
        "correctness_pass": True,
        "performance_pass": passed,
        "error_type": "" if passed else "CollectiveBandwidthGateFailed",
        "dist_backend_requested": env.dist_backend_requested,
        "dist_backend": env.dist_backend,
        "device_vendor": env.device_vendor,
        "comm_runtime": env.comm_runtime,
    }
    append_jsonl(output_dir / "collective_bandwidth_summary.jsonl", [summary])
    return summary


def run_collective_bandwidth_gate(
    output_dir: Path,
    dtype_name: str,
    message_sizes: list[int],
    ops: list[str],
    moe_patterns: list[str],
    ep_size: int,
    warmup: int,
    iters: int,
    seed: int,
    min_busbw: float,
    avg_busbw: float,
    test_round: str,
    group_id: str,
) -> None:
    allowed = {"all_reduce", "reduce_scatter", "all_gather", "broadcast", "all_to_all", "all_to_allv"}
    invalid = sorted(set(ops) - allowed)
    if invalid:
        raise ValueError(f"unsupported collective bandwidth ops: {invalid}")

    torch, dist, env = init_dist()
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = _dtype(torch, dtype_name)
    job_id = os.environ.get("HEALTHCHECK_JOB_ID", str(uuid.uuid4()))
    hostnames = sorted(set(_all_gather_object(dist, env.hostname, env.world_size)))
    env.group_id = group_id or env.group_id or f"{test_round}-" + "-".join(hostnames)
    ep_group = None
    ep_group_rank = env.rank
    ep_group_size = env.world_size
    if "all_to_allv" in ops:
        ep_group, ep_group_rank, ep_group_size, _ = _make_ep_group(dist, env, ep_size)

    all_summaries: list[dict[str, Any]] = []
    try:
        for size in message_sizes:
            for op in ops:
                if op == "all_to_allv":
                    for pattern in moe_patterns:
                        summary = _collective_bandwidth_case(
                            torch,
                            dist,
                            env,
                            output_dir,
                            job_id,
                            test_round,
                            env.group_id,
                            hostnames,
                            dtype_name,
                            dtype,
                            op,
                            size,
                            warmup,
                            iters,
                            seed,
                            min_busbw,
                            avg_busbw,
                            payload_pattern=pattern,
                            collective_group=ep_group,
                            collective_group_rank=ep_group_rank,
                            collective_group_size=ep_group_size,
                        )
                        if summary is not None:
                            all_summaries.append(summary)
                        dist.barrier(device_ids=[env.local_rank])
                else:
                    summary = _collective_bandwidth_case(
                        torch,
                        dist,
                        env,
                        output_dir,
                        job_id,
                        test_round,
                        env.group_id,
                        hostnames,
                        dtype_name,
                        dtype,
                        op,
                        size,
                        warmup,
                        iters,
                        seed,
                        min_busbw,
                        avg_busbw,
                    )
                    if summary is not None:
                        all_summaries.append(summary)
                    dist.barrier(device_ids=[env.local_rank])

        if env.rank == 0:
            write_json(
                output_dir / "collective_bandwidth_gate.json",
                {
                    "status": "PASS" if all(row["bandwidth_pass"] for row in all_summaries) else "FAIL",
                    "summaries": all_summaries,
                },
            )
            _write_collective_bandwidth_report(output_dir, all_summaries)
            failed = [row for row in all_summaries if not row["bandwidth_pass"]]
        else:
            failed = []
    finally:
        dist.barrier(device_ids=[env.local_rank])
        dist.destroy_process_group()
    # Performance gates are classification inputs, not process/correctness failures.


def run_collective_case_retest(
    output_dir: Path,
    dtype_name: str,
    cases: list[dict[str, Any]],
    ep_size: int,
    warmup: int,
    iters: int,
    diagnostic_iters: int,
    seed: int,
    test_round: str,
    group_id: str,
) -> None:
    """Retest an exact frozen case plan with one process-group initialization."""
    torch, dist, env = init_dist()
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = _dtype(torch, dtype_name)
    job_id = os.environ.get("HEALTHCHECK_JOB_ID", str(uuid.uuid4()))
    hostnames = sorted(set(_all_gather_object(dist, env.hostname, env.world_size)))
    env.group_id = group_id or env.group_id or f"{test_round}-" + "-".join(hostnames)
    ep_group = None
    ep_group_rank = env.rank
    ep_group_size = env.world_size
    if any(str(case.get("op_type", "")) == "all_to_allv" for case in cases):
        ep_group, ep_group_rank, ep_group_size, _ = _make_ep_group(dist, env, ep_size)
    summaries: list[dict[str, Any]] = []
    try:
        for case in cases:
            op = str(case.get("op_type", ""))
            size = int(case.get("message_bytes", 0) or 0)
            pattern = str(case.get("payload_pattern", "none") or "none")
            if not op or size <= 0:
                raise ValueError(f"invalid retest case: {case}")
            kwargs: dict[str, Any] = {}
            if op == "all_to_allv":
                kwargs = {
                    "collective_group": ep_group,
                    "collective_group_rank": ep_group_rank,
                    "collective_group_size": ep_group_size,
                }
            summary = _collective_bandwidth_case(
                torch,
                dist,
                env,
                output_dir,
                job_id,
                test_round,
                env.group_id,
                hostnames,
                dtype_name,
                dtype,
                op,
                size,
                warmup,
                iters,
                seed,
                0.0,
                0.0,
                payload_pattern=pattern,
                diagnostic_iters=diagnostic_iters,
                source_stage=str(case.get("stage", "collective_bandwidth")),
                **kwargs,
            )
            if summary is not None:
                summary["retest"] = True
                summaries.append(summary)
            dist.barrier(device_ids=[env.local_rank])
        if env.rank == 0:
            write_json(
                output_dir / "collective_bandwidth_gate.json",
                {"status": "PASS", "retest": True, "summaries": summaries},
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_dynamic_suite(
    output_dir: Path,
    dtype_name: str,
    message_sizes: list[int],
    moe_patterns: list[str],
    warmup: int,
    iters: int,
    seed: int,
    bandwidth_message_sizes: list[int],
    bandwidth_warmup: int,
    bandwidth_iters: int,
    bandwidth_min_busbw: float,
    bandwidth_avg_busbw: float,
    collective_bandwidth_message_sizes: list[int],
    collective_bandwidth_ops: list[str],
    collective_bandwidth_moe_patterns: list[str],
    collective_bandwidth_ep_size: int,
    collective_bandwidth_warmup: int,
    collective_bandwidth_iters: int,
    collective_bandwidth_min_busbw: float,
    collective_bandwidth_avg_busbw: float,
    test_round: str,
    group_id: str,
) -> None:
    torch, dist, env = init_dist()
    output_dir.mkdir(parents=True, exist_ok=True)
    env.group_id = group_id or env.group_id or f"{test_round}-{env.hostname}"
    _write_comm_path_debug(dist, env, output_dir, "dynamic-suite")
    if env.rank == 0:
        collective_cases_per_size = sum(
            len(collective_bandwidth_moe_patterns) if op == "all_to_allv" else 1
            for op in collective_bandwidth_ops
        )
        write_json(
            output_dir / "dynamic_suite_plan.json",
            {
                "schema_version": 1,
                "expected_world_size": env.world_size,
                "bandwidth_message_sizes": bandwidth_message_sizes,
                "collective_message_sizes": collective_bandwidth_message_sizes,
                "collective_ops": collective_bandwidth_ops,
                "collective_moe_patterns": collective_bandwidth_moe_patterns,
                "expected_bandwidth_case_count": len(bandwidth_message_sizes),
                "expected_collective_case_count": len(collective_bandwidth_message_sizes) * collective_cases_per_size,
                "expected_case_count": len(bandwidth_message_sizes)
                + len(collective_bandwidth_message_sizes) * collective_cases_per_size,
            },
        )

    real_destroy = dist.destroy_process_group

    def deferred_destroy(*_args, **_kwargs) -> None:
        return None

    dist.destroy_process_group = deferred_destroy
    try:
        ping_group(output_dir / "smoke", test_round=f"{test_round}_smoke", group_id=env.group_id)
        run_single_node(
            output_dir=output_dir / "quick",
            dtype_name=dtype_name,
            message_sizes=message_sizes,
            moe_patterns=moe_patterns,
            warmup=warmup,
            iters=iters,
            seed=seed,
        )
        if env.rank == 0:
            try:
                from .analyze import analyze_results

                analyze_results(output_dir / "quick", output_dir / "quick" / "report.md")
            except Exception:
                pass
        run_bandwidth_gate(
            output_dir=output_dir / "bandwidth",
            dtype_name=dtype_name,
            message_sizes=bandwidth_message_sizes,
            warmup=bandwidth_warmup,
            iters=bandwidth_iters,
            seed=seed,
            min_busbw=bandwidth_min_busbw,
            avg_busbw=bandwidth_avg_busbw,
            test_round=f"{test_round}_bandwidth",
            group_id=env.group_id,
        )
        run_collective_bandwidth_gate(
            output_dir=output_dir / "collective_bandwidth",
            dtype_name=dtype_name,
            message_sizes=collective_bandwidth_message_sizes,
            ops=collective_bandwidth_ops,
            moe_patterns=collective_bandwidth_moe_patterns,
            ep_size=collective_bandwidth_ep_size,
            warmup=collective_bandwidth_warmup,
            iters=collective_bandwidth_iters,
            seed=seed,
            min_busbw=collective_bandwidth_min_busbw,
            avg_busbw=collective_bandwidth_avg_busbw,
            test_round=f"{test_round}_collective_bandwidth",
            group_id=env.group_id,
        )
    finally:
        dist.destroy_process_group = real_destroy
        if dist.is_initialized():
            real_destroy()


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
        error_type = _fault_error_type(env)
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
        error_type = _fault_error_type(env)
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
        "pod_name": env.pod_name,
        "node_name": env.node_name,
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
            dist.all_to_all_single(out, inp)
            return out
        raise ValueError(op)

    try:
        for _ in range(max(1, warmup)):
            once()
        elapsed = _sync_time(torch, lambda: _repeat(lambda: (_maybe_fault_sleep(env), once())[1], iters)) / max(1, iters)
        y = _apply_faults(torch, env, once())
        nan_count, inf_count = _nan_inf_counts(torch, y)
        checksum = _tensor_checksum(y)
        error_type = _fault_error_type(env, error_type)
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
    output_split_sizes = _routing_output_counts(
        pattern,
        env.world_size,
        token_size * env.world_size,
        env.rank,
        seed,
    )

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
        error_type = _fault_error_type(env, error_type)
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
