from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
FAMILY_ORDER = ("tp", "dense_dp", "expert_dp", "ep", "pp")
FAMILY_OPS = {
    "tp": ("all_reduce", "reduce_scatter", "all_gather"),
    "dense_dp": ("all_reduce", "reduce_scatter", "all_gather"),
    "expert_dp": ("all_reduce",),
    "ep": ("all_to_all", "all_to_allv"),
    "pp": ("send_recv",),
}
REPRESENTATIVE_MESSAGE_SIZES = (1 << 20, 128 << 20, 1 << 30)


@dataclass(frozen=True)
class TopologyGroup:
    family: str
    group_id: str
    ranks: tuple[int, ...]


@dataclass(frozen=True)
class TopologyProfile:
    world_size: int
    groups: dict[str, tuple[TopologyGroup, ...]]
    workload_shapes: tuple[dict[str, Any], ...]
    parallelism: dict[str, Any]


@dataclass(frozen=True)
class TrainingTopologyManifest:
    path: Path
    sha256: str
    ranks_per_node: int
    framework: dict[str, Any]
    model: dict[str, Any]
    profiles: dict[int, TopologyProfile]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _as_group(family: str, index: int, raw: Any) -> TopologyGroup:
    if isinstance(raw, dict):
        ranks_raw = raw.get("ranks")
        group_id = str(raw.get("group_id") or raw.get("id") or f"{family}_{index:04d}")
    else:
        ranks_raw = raw
        group_id = f"{family}_{index:04d}"
    if not isinstance(ranks_raw, list) or not ranks_raw:
        raise ValueError(f"{family} group {group_id} must contain a non-empty ranks list")
    if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks_raw):
        raise ValueError(f"{family} group {group_id} contains a non-integer rank")
    ranks = tuple(int(rank) for rank in ranks_raw)
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"{family} group {group_id} contains duplicate ranks")
    return TopologyGroup(family=family, group_id=group_id, ranks=ranks)


def _validate_family_coverage(family: str, groups: Iterable[TopologyGroup], world_size: int) -> None:
    owners: dict[int, str] = {}
    for group in groups:
        for rank in group.ranks:
            if rank < 0 or rank >= world_size:
                raise ValueError(f"{family} group {group.group_id} rank {rank} is outside [0,{world_size})")
            if rank in owners:
                raise ValueError(
                    f"{family} rank {rank} appears in both {owners[rank]} and {group.group_id}"
                )
            owners[rank] = group.group_id
    missing = sorted(set(range(world_size)) - set(owners))
    if missing:
        preview = ",".join(str(rank) for rank in missing[:16])
        raise ValueError(f"{family} groups do not cover all ranks; missing={preview}")


def _parse_profile(world_key: str, raw: Any) -> TopologyProfile:
    if not isinstance(raw, dict):
        raise ValueError(f"profile {world_key} must be an object")
    world_size = int(raw.get("world_size", world_key))
    if world_size <= 0 or str(world_size) != str(world_key):
        raise ValueError(f"profile key {world_key} and world_size={world_size} differ")
    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, dict):
        raise ValueError(f"profile {world_key} must contain groups")
    groups: dict[str, tuple[TopologyGroup, ...]] = {}
    for family in FAMILY_ORDER:
        family_raw = groups_raw.get(family)
        if not isinstance(family_raw, list) or not family_raw:
            raise ValueError(f"profile {world_key} is missing non-empty {family} groups")
        parsed = tuple(_as_group(family, index, item) for index, item in enumerate(family_raw))
        ids = [group.group_id for group in parsed]
        if len(ids) != len(set(ids)):
            raise ValueError(f"profile {world_key} contains duplicate {family} group ids")
        _validate_family_coverage(family, parsed, world_size)
        groups[family] = parsed
    shapes_raw = raw.get("workload_shapes", [])
    if not isinstance(shapes_raw, list):
        raise ValueError(f"profile {world_key} workload_shapes must be a list")
    shapes: list[dict[str, Any]] = []
    for index, shape in enumerate(shapes_raw):
        if not isinstance(shape, dict):
            raise ValueError(f"profile {world_key} workload shape {index} must be an object")
        family = str(shape.get("family", ""))
        op = str(shape.get("op", ""))
        size = int(shape.get("message_bytes", 0))
        if family not in FAMILY_OPS or op not in FAMILY_OPS[family] or size <= 0:
            raise ValueError(f"profile {world_key} workload shape {index} is invalid")
        shapes.append(dict(shape))
    parallelism = raw.get("parallelism", {})
    if not isinstance(parallelism, dict):
        raise ValueError(f"profile {world_key} parallelism must be an object")
    return TopologyProfile(
        world_size=world_size,
        groups=groups,
        workload_shapes=tuple(shapes),
        parallelism=dict(parallelism),
    )


def load_training_topology_manifest(path: Path) -> TrainingTopologyManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("training topology manifest must be a JSON object")
    if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported training topology schema_version={raw.get('schema_version')}")
    ranks_per_node = int(raw.get("ranks_per_node", 0))
    if ranks_per_node <= 0:
        raise ValueError("ranks_per_node must be positive")
    framework = raw.get("framework")
    if not isinstance(framework, dict):
        raise ValueError("framework metadata is required")
    required_framework = ("name", "code_sha256", "config_sha256", "rank_order")
    missing_framework = [key for key in required_framework if not str(framework.get(key, "")).strip()]
    if missing_framework:
        raise ValueError("framework metadata missing: " + ",".join(missing_framework))
    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ValueError("profiles must be a non-empty object")
    profiles = {int(key): _parse_profile(str(key), value) for key, value in profiles_raw.items()}
    model = raw.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("model metadata must be an object")
    return TrainingTopologyManifest(
        path=path,
        sha256=manifest_sha256(raw),
        ranks_per_node=ranks_per_node,
        framework=dict(framework),
        model=dict(model),
        profiles=profiles,
    )


def require_profile(
    manifest: TrainingTopologyManifest,
    world_size: int,
    ranks_per_node: int,
) -> TopologyProfile:
    if manifest.ranks_per_node != ranks_per_node:
        raise ValueError(
            f"manifest ranks_per_node={manifest.ranks_per_node} differs from runtime={ranks_per_node}"
        )
    try:
        return manifest.profiles[world_size]
    except KeyError as exc:
        raise ValueError(f"manifest does not contain world_size={world_size} profile") from exc


def topology_case_plan(profile: TopologyProfile) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for family in FAMILY_ORDER:
        for op in FAMILY_OPS[family]:
            patterns = ("uniform", "skewed", "hot_expert", "random", "empty_expert") if op == "all_to_allv" else ("none",)
            for size in REPRESENTATIVE_MESSAGE_SIZES:
                for pattern in patterns:
                    key = (family, op, size, pattern)
                    seen.add(key)
                    cases.append(
                        {
                            "case_id": f"{family}/{op}/{size}/{pattern}",
                            "family": family,
                            "op": op,
                            "message_bytes": size,
                            "payload_pattern": pattern,
                            "source": "representative",
                        }
                    )
    for index, shape in enumerate(profile.workload_shapes):
        family = str(shape["family"])
        op = str(shape["op"])
        size = int(shape["message_bytes"])
        pattern = str(shape.get("payload_pattern", "none"))
        key = (family, op, size, pattern)
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            {
                "case_id": str(shape.get("case_id") or f"training_shape_{index:04d}"),
                "family": family,
                "op": op,
                "message_bytes": size,
                "payload_pattern": pattern,
                "source": "training_shape",
            }
        )
    return cases
