from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def analyze_results(input_dir: Path, output: Path | None = None) -> str:
    summaries = _read_jsonl(input_dir / "group_summary.jsonl")
    ranks = _read_jsonl(input_dir / "rank_detail.jsonl")
    lines = ["# Healthcheck Report", ""]
    lines.append(f"- input_dir: `{input_dir}`")
    lines.append(f"- group summaries: {len(summaries)}")
    lines.append(f"- rank details: {len(ranks)}")
    lines.append("")

    failed = [r for r in summaries if not r.get("correctness_pass", False) or not r.get("performance_pass", False)]
    lines.append(f"## Summary")
    lines.append("")
    lines.append(f"- failed summaries: {len(failed)}")
    if summaries:
        meta_keys = ["dist_backend", "dist_backend_requested", "device_vendor", "comm_runtime", "test_round", "group_id"]
        for key in meta_keys:
            values = sorted({str(row.get(key, "")) for row in summaries if row.get(key, "")})
            if values:
                lines.append(f"- {key}: `{','.join(values)}`")
    lines.append("")

    by_op: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_op[row.get("op_type", "unknown")].append(row)

    lines.append("## By Operation")
    lines.append("")
    lines.append("| op | count | max p99 latency s | min algbw GB/s | min busbw GB/s | errors |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for op, values in sorted(by_op.items()):
        max_p99 = max(float(v.get("latency_p99", 0.0)) for v in values)
        min_bw = min(float(v.get("algbw", 0.0)) for v in values)
        min_busbw = min(float(v.get("busbw", 0.0)) for v in values)
        errors = sorted({v.get("error_type", "") for v in values if v.get("error_type")})
        lines.append(f"| {op} | {len(values)} | {max_p99:.6f} | {min_bw:.3f} | {min_busbw:.3f} | {','.join(errors)} |")

    if ranks:
        slow = sorted(ranks, key=lambda row: float(row.get("rank_latency", 0.0)), reverse=True)[:10]
        lines.append("")
        lines.append("## Slowest Rank Samples")
        lines.append("")
        lines.append("| rank | host | op | size | pattern | latency s | error |")
        lines.append("| ---: | --- | --- | --- | --- | ---: | --- |")
        for row in slow:
            lines.append(
                "| {rank} | {host} | {op} | {size} | {pattern} | {lat:.6f} | {err} |".format(
                    rank=row.get("rank", ""),
                    host=row.get("hostname", ""),
                    op=row.get("op_type", ""),
                    size=row.get("message_size", ""),
                    pattern=row.get("payload_pattern", ""),
                    lat=float(row.get("rank_latency", 0.0)),
                    err=row.get("rank_error_type", ""),
                )
            )

    if failed:
        lines.append("")
        lines.append("## Failed / Suspicious Summaries")
        lines.append("")
        lines.append("| op | size | pattern | correctness | performance | error |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in failed:
            lines.append(
                "| {op} | {size} | {pattern} | {c} | {p} | {err} |".format(
                    op=row.get("op_type", ""),
                    size=row.get("message_size", ""),
                    pattern=row.get("payload_pattern", ""),
                    c=row.get("correctness_pass", ""),
                    p=row.get("performance_pass", ""),
                    err=row.get("error_type", ""),
                )
            )

    text = "\n".join(lines) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return text
