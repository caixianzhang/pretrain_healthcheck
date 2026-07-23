#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_generator(cls: Any, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> Any:
    signature = inspect.signature(cls)
    values = {
        "tp": tp,
        "ep": ep,
        "dp": dp,
        "pp": pp,
        "cp": cp,
        "order": order,
        "rank_offset": 0,
    }
    if all(name in values for name in signature.parameters):
        return cls(**{name: values[name] for name in signature.parameters})
    return cls(tp, ep, dp, pp, cp, order, rank_offset=0)


def groups(generator: Any, token: str, independent_ep: bool = False) -> list[list[int]]:
    method = generator.get_ranks
    signature = inspect.signature(method)
    kwargs = {"independent_ep": independent_ep} if "independent_ep" in signature.parameters else {}
    result = method(token, **kwargs)
    return [[int(rank) for rank in group] for group in result]


def group_rows(family: str, rank_groups: list[list[int]]) -> list[dict[str, Any]]:
    return [
        {"group_id": f"{family}_{index:04d}", "ranks": ranks}
        for index, ranks in enumerate(rank_groups)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="export healthcheck topology from Megatron RankGenerator")
    parser.add_argument("--megatron-path", type=Path, required=True)
    parser.add_argument("--rank-generator-module", default="megatron.core.parallel_state")
    parser.add_argument("--world-size", action="append", type=int, required=True)
    parser.add_argument("--ranks-per-node", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--mbs", type=int, required=True)
    parser.add_argument("--gbs", type=int, required=True)
    parser.add_argument("--rank-order", default="tp-cp-ep-dp-pp")
    parser.add_argument("--model-json", type=Path)
    parser.add_argument("--workload-shapes-json", type=Path)
    parser.add_argument(
        "--profile-overrides-json",
        type=Path,
        help="Per-world-size tp/ep/etp/pp/cp/rank_order overrides for scaled profiles.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.megatron_path.resolve()))
    module = importlib.import_module(args.rank_generator_module)
    generator_cls = getattr(module, "RankGenerator", None)
    if generator_cls is None:
        raise RuntimeError(f"{args.rank_generator_module} does not export RankGenerator")
    module_path = Path(module.__file__).resolve()
    model = json.loads(args.model_json.read_text(encoding="utf-8")) if args.model_json else {}
    workload_by_world = (
        json.loads(args.workload_shapes_json.read_text(encoding="utf-8"))
        if args.workload_shapes_json
        else {}
    )
    profile_overrides = (
        json.loads(args.profile_overrides_json.read_text(encoding="utf-8"))
        if args.profile_overrides_json
        else {}
    )
    if not isinstance(profile_overrides, dict):
        raise ValueError("profile overrides must be a JSON object keyed by world size")
    config = {
        "tp": args.tp,
        "ep": args.ep,
        "etp": args.etp,
        "pp": args.pp,
        "cp": args.cp,
        "mbs": args.mbs,
        "gbs": args.gbs,
        "rank_order": args.rank_order,
        "world_sizes": sorted(set(args.world_size)),
        "profile_overrides": profile_overrides,
    }
    profiles: dict[str, Any] = {}
    for world_size in config["world_sizes"]:
        override = profile_overrides.get(str(world_size), {})
        if not isinstance(override, dict):
            raise ValueError(f"profile override for world_size={world_size} must be an object")
        allowed_override_keys = {"tp", "ep", "etp", "pp", "cp", "rank_order"}
        unknown = sorted(set(override) - allowed_override_keys)
        if unknown:
            raise ValueError(f"unknown profile override keys for world_size={world_size}: {','.join(unknown)}")
        profile_parallelism = {
            "tp": int(override.get("tp", args.tp)),
            "ep": int(override.get("ep", args.ep)),
            "etp": int(override.get("etp", args.etp)),
            "pp": int(override.get("pp", args.pp)),
            "cp": int(override.get("cp", args.cp)),
            "rank_order": str(override.get("rank_order", args.rank_order)),
        }
        dense_denominator = profile_parallelism["tp"] * profile_parallelism["pp"] * profile_parallelism["cp"]
        if world_size % dense_denominator:
            raise ValueError(f"world_size={world_size} is not divisible by TP*PP*CP={dense_denominator}")
        dense_dp = world_size // dense_denominator
        expert_denominator = profile_parallelism["etp"] * profile_parallelism["ep"] * profile_parallelism["pp"]
        if world_size % expert_denominator:
            raise ValueError(
                f"world_size={world_size} is not divisible by ETP*EP*PP={expert_denominator}"
            )
        expert_dp = world_size // expert_denominator
        profile_parallelism["dense_dp"] = dense_dp
        profile_parallelism["expert_dp"] = expert_dp
        dense_generator = create_generator(
            generator_cls,
            profile_parallelism["tp"],
            1,
            dense_dp,
            profile_parallelism["pp"],
            profile_parallelism["cp"],
            profile_parallelism["rank_order"],
        )
        expert_generator = create_generator(
            generator_cls,
            profile_parallelism["etp"],
            profile_parallelism["ep"],
            expert_dp,
            profile_parallelism["pp"],
            1,
            profile_parallelism["rank_order"],
        )
        dense_pp = groups(dense_generator, "pp")
        expert_pp = groups(expert_generator, "pp")
        if dense_pp != expert_pp:
            raise ValueError(
                f"world_size={world_size} dense and expert PP groups differ; "
                "the exported topology does not match Megatron initialization"
            )
        profile_groups = {
            "tp": group_rows("tp", groups(dense_generator, "tp")),
            "dense_dp": group_rows("dense_dp", groups(dense_generator, "dp")),
            "expert_dp": group_rows("expert_dp", groups(expert_generator, "dp", independent_ep=True)),
            "ep": group_rows("ep", groups(expert_generator, "ep", independent_ep=True)),
            "pp": group_rows("pp", dense_pp),
        }
        profiles[str(world_size)] = {
            "world_size": world_size,
            "parallelism": profile_parallelism,
            "groups": profile_groups,
            "workload_shapes": workload_by_world.get(str(world_size), []),
        }
    manifest = {
        "schema_version": 1,
        "ranks_per_node": args.ranks_per_node,
        "framework": {
            "name": "Megatron/MindSpeed",
            "module": args.rank_generator_module,
            "module_path": str(module_path),
            "code_sha256": sha256_file(module_path),
            "config_sha256": canonical_sha256(config),
            "rank_order": args.rank_order,
        },
        "parallelism": config,
        "model": model,
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
