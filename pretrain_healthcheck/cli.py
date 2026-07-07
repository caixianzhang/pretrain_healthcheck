from __future__ import annotations

import argparse
from pathlib import Path

from .analyze import analyze_results
from .common import parse_size_list
from .static_checks import collect_static_checks


COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES = (
    "1K,2K,4K,8K,16K,32K,64K,128K,256K,512K,"
    "1M,2M,4M,8M,16M,32M,64M,128M,256M,512M,1G,2G"
)
COLLECTIVE_ACCEPTANCE_OPS = "all_reduce,reduce_scatter,all_gather,broadcast,all_to_all,all_to_allv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pretrain-healthcheck")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_static = sub.add_parser("static", help="run static machine/GPU/HCA checks")
    p_static.add_argument("--output", type=Path, default=Path("results/static.json"))

    p_run = sub.add_parser("run-single-node", help="run single-node GPU and collective checks")
    p_run.add_argument("--output-dir", type=Path, required=True)
    p_run.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p_run.add_argument("--message-sizes", default="1M,16M,64M")
    p_run.add_argument(
        "--moe-patterns",
        default="uniform,skewed,hot_expert,random,empty_expert",
        help="comma-separated MoE payload patterns",
    )
    p_run.add_argument("--warmup", type=int, default=2)
    p_run.add_argument("--iters", type=int, default=5)
    p_run.add_argument("--seed", type=int, default=20260623)

    p_group = sub.add_parser("run-group", help="run multi-node group GPU and collective checks")
    p_group.add_argument("--output-dir", type=Path, required=True)
    p_group.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p_group.add_argument("--message-sizes", default="1M,16M,64M")
    p_group.add_argument(
        "--moe-patterns",
        default="uniform,skewed,hot_expert,random,empty_expert",
        help="comma-separated MoE payload patterns",
    )
    p_group.add_argument("--warmup", type=int, default=2)
    p_group.add_argument("--iters", type=int, default=5)
    p_group.add_argument("--seed", type=int, default=20260623)
    p_group.add_argument("--test-round", default="current_vcjob")
    p_group.add_argument("--group-id", default="")

    p_ping = sub.add_parser("ping-group", help="run minimal distributed connectivity check")
    p_ping.add_argument("--output-dir", type=Path, required=True)
    p_ping.add_argument("--test-round", default="smoke")
    p_ping.add_argument("--group-id", default="")

    p_bw = sub.add_parser("run-bandwidth", help="run all-reduce bandwidth gate")
    p_bw.add_argument("--output-dir", type=Path, required=True)
    p_bw.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p_bw.add_argument("--message-sizes", default="1G,4G,8G,16G")
    p_bw.add_argument("--warmup", type=int, default=5)
    p_bw.add_argument("--iters", type=int, default=100)
    p_bw.add_argument("--seed", type=int, default=20260623)
    p_bw.add_argument("--min-busbw", type=float, default=270.0, help="second-lowest busbw gate in GB/s")
    p_bw.add_argument("--avg-busbw", type=float, default=290.0, help="average busbw gate in GB/s")
    p_bw.add_argument("--test-round", default="bandwidth")
    p_bw.add_argument("--group-id", default="")

    p_cbw = sub.add_parser("run-collective-bandwidth", help="run multi-collective bandwidth baseline")
    p_cbw.add_argument("--output-dir", type=Path, required=True)
    p_cbw.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p_cbw.add_argument("--message-sizes", default=COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES)
    p_cbw.add_argument(
        "--ops",
        default=COLLECTIVE_ACCEPTANCE_OPS,
        help="comma-separated ops: all_reduce,reduce_scatter,all_gather,broadcast,all_to_all,all_to_allv",
    )
    p_cbw.add_argument(
        "--moe-patterns",
        default="uniform,skewed,hot_expert,random,empty_expert",
        help="comma-separated MoE payload patterns for all_to_allv",
    )
    p_cbw.add_argument("--ep-size", type=int, default=8, help="EP group size for all_to_allv")
    p_cbw.add_argument("--warmup", type=int, default=5)
    p_cbw.add_argument("--iters", type=int, default=30)
    p_cbw.add_argument("--seed", type=int, default=20260623)
    p_cbw.add_argument("--min-busbw", type=float, default=0.0, help="second-lowest busbw gate in GB/s")
    p_cbw.add_argument("--avg-busbw", type=float, default=0.0, help="average busbw gate in GB/s")
    p_cbw.add_argument("--test-round", default="collective_bandwidth")
    p_cbw.add_argument("--group-id", default="")

    p_suite = sub.add_parser("run-dynamic-suite", help="run smoke/quick/bandwidth/collective-bandwidth in one process group")
    p_suite.add_argument("--output-dir", type=Path, required=True)
    p_suite.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p_suite.add_argument("--message-sizes", default="1M")
    p_suite.add_argument(
        "--moe-patterns",
        default="uniform,skewed,hot_expert,random,empty_expert",
        help="comma-separated MoE payload patterns",
    )
    p_suite.add_argument("--warmup", type=int, default=1)
    p_suite.add_argument("--iters", type=int, default=1)
    p_suite.add_argument("--bandwidth-message-sizes", default="1G")
    p_suite.add_argument("--bandwidth-warmup", type=int, default=1)
    p_suite.add_argument("--bandwidth-iters", type=int, default=3)
    p_suite.add_argument("--bandwidth-min-busbw", type=float, default=0.0)
    p_suite.add_argument("--bandwidth-avg-busbw", type=float, default=0.0)
    p_suite.add_argument("--collective-bandwidth-message-sizes", default=COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES)
    p_suite.add_argument(
        "--collective-bandwidth-ops",
        default=COLLECTIVE_ACCEPTANCE_OPS,
    )
    p_suite.add_argument(
        "--collective-bandwidth-moe-patterns",
        default="uniform,skewed,hot_expert,random,empty_expert",
    )
    p_suite.add_argument("--collective-bandwidth-ep-size", type=int, default=8)
    p_suite.add_argument("--collective-bandwidth-warmup", type=int, default=1)
    p_suite.add_argument("--collective-bandwidth-iters", type=int, default=3)
    p_suite.add_argument("--collective-bandwidth-min-busbw", type=float, default=0.0)
    p_suite.add_argument("--collective-bandwidth-avg-busbw", type=float, default=0.0)
    p_suite.add_argument("--seed", type=int, default=20260623)
    p_suite.add_argument("--test-round", default="dynamic_suite")
    p_suite.add_argument("--group-id", default="")

    p_analyze = sub.add_parser("analyze", help="analyze a result directory")
    p_analyze.add_argument("--input-dir", type=Path, required=True)
    p_analyze.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "static":
        result = collect_static_checks(args.output)
        print(f"static_check_status={result['summary']['static_check_status']}")
        print(f"wrote {args.output}")
        return
    if args.cmd == "run-single-node":
        import os

        from .torch_checks import run_single_node

        run_single_node(
            output_dir=args.output_dir,
            dtype_name=args.dtype,
            message_sizes=parse_size_list(args.message_sizes),
            moe_patterns=[x.strip() for x in args.moe_patterns.split(",") if x.strip()],
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"wrote {args.output_dir}")
        return
    if args.cmd == "run-group":
        import os

        from .torch_checks import run_group

        run_group(
            output_dir=args.output_dir,
            dtype_name=args.dtype,
            message_sizes=parse_size_list(args.message_sizes),
            moe_patterns=[x.strip() for x in args.moe_patterns.split(",") if x.strip()],
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            test_round=args.test_round,
            group_id=args.group_id,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"wrote {args.output_dir}")
        return
    if args.cmd == "ping-group":
        import os

        from .torch_checks import ping_group

        ping_group(
            output_dir=args.output_dir,
            test_round=args.test_round,
            group_id=args.group_id,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"wrote {args.output_dir}")
        return
    if args.cmd == "run-bandwidth":
        import os

        from .torch_checks import run_bandwidth_gate

        run_bandwidth_gate(
            output_dir=args.output_dir,
            dtype_name=args.dtype,
            message_sizes=parse_size_list(args.message_sizes),
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            min_busbw=args.min_busbw,
            avg_busbw=args.avg_busbw,
            test_round=args.test_round,
            group_id=args.group_id,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"wrote {args.output_dir}")
        return
    if args.cmd == "run-collective-bandwidth":
        import os

        from .torch_checks import run_collective_bandwidth_gate

        run_collective_bandwidth_gate(
            output_dir=args.output_dir,
            dtype_name=args.dtype,
            message_sizes=parse_size_list(args.message_sizes),
            ops=[x.strip() for x in args.ops.split(",") if x.strip()],
            moe_patterns=[x.strip() for x in args.moe_patterns.split(",") if x.strip()],
            ep_size=args.ep_size,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            min_busbw=args.min_busbw,
            avg_busbw=args.avg_busbw,
            test_round=args.test_round,
            group_id=args.group_id,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"wrote {args.output_dir}")
        return
    if args.cmd == "run-dynamic-suite":
        import os

        from .torch_checks import run_dynamic_suite

        run_dynamic_suite(
            output_dir=args.output_dir,
            dtype_name=args.dtype,
            message_sizes=parse_size_list(args.message_sizes),
            moe_patterns=[x.strip() for x in args.moe_patterns.split(",") if x.strip()],
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            bandwidth_message_sizes=parse_size_list(args.bandwidth_message_sizes),
            bandwidth_warmup=args.bandwidth_warmup,
            bandwidth_iters=args.bandwidth_iters,
            bandwidth_min_busbw=args.bandwidth_min_busbw,
            bandwidth_avg_busbw=args.bandwidth_avg_busbw,
            collective_bandwidth_message_sizes=parse_size_list(args.collective_bandwidth_message_sizes),
            collective_bandwidth_ops=[x.strip() for x in args.collective_bandwidth_ops.split(",") if x.strip()],
            collective_bandwidth_moe_patterns=[
                x.strip() for x in args.collective_bandwidth_moe_patterns.split(",") if x.strip()
            ],
            collective_bandwidth_ep_size=args.collective_bandwidth_ep_size,
            collective_bandwidth_warmup=args.collective_bandwidth_warmup,
            collective_bandwidth_iters=args.collective_bandwidth_iters,
            collective_bandwidth_min_busbw=args.collective_bandwidth_min_busbw,
            collective_bandwidth_avg_busbw=args.collective_bandwidth_avg_busbw,
            test_round=args.test_round,
            group_id=args.group_id,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"wrote {args.output_dir}")
        return
    if args.cmd == "analyze":
        text = analyze_results(args.input_dir, args.output)
        if args.output:
            print(f"wrote {args.output}")
        else:
            print(text)


if __name__ == "__main__":
    main()
