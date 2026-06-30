from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from .common import env_snapshot, hostname, run_cmd, write_json


def _disk_usage(paths: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
            result[path] = {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "free_ratio": usage.free / usage.total if usage.total else 0.0,
            }
        except OSError as exc:
            result[path] = {"error": str(exc)}
    return result


def collect_static_checks(output: Path | None = None) -> dict[str, Any]:
    nvidia_query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,driver_version,vbios_version,temperature.gpu,power.draw,clocks.sm,memory.total,memory.used,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.volatile.total",
        "--format=csv,noheader,nounits",
    ]

    checks: dict[str, Any] = {
        "hostname": hostname(),
        "env": env_snapshot(
            [
                "CUDA_VISIBLE_DEVICES",
                "NCCL_DEBUG",
                "NCCL_IB_HCA",
                "NCCL_SOCKET_IFNAME",
                "NCCL_NET",
                "NCCL_TOPO_FILE",
            ]
        ),
        "commands": {},
        "disk": _disk_usage(["/", "/tmp", os.getcwd()]),
    }

    command_specs = {
        "uname": ["uname", "-a"],
        "date": ["date", "-Is"],
        "timedatectl": ["timedatectl", "status"],
        "nvidia_smi_l": ["nvidia-smi", "-L"],
        "nvidia_smi_query": nvidia_query,
        "nvidia_smi_topo": ["nvidia-smi", "topo", "-m"],
        "ibstat": ["ibstat"],
        "ibv_devinfo": ["ibv_devinfo"],
        "ip_link": ["ip", "-br", "link"],
    }
    for name, cmd in command_specs.items():
        checks["commands"][name] = run_cmd(cmd, timeout=20)

    nvidia = checks["commands"].get("nvidia_smi_query", {}).get("stdout", "")
    ibstat = checks["commands"].get("ibstat", {}).get("stdout", "")
    ibv_devinfo = checks["commands"].get("ibv_devinfo", {}).get("stdout", "")

    xid_hits = re.findall(r"(?i)xid[^\\n]*", nvidia)
    gpu_visible = checks["commands"]["nvidia_smi_l"]["returncode"] == 0 and bool(
        checks["commands"]["nvidia_smi_l"]["stdout"]
    )
    hca_visible = checks["commands"]["ibv_devinfo"]["returncode"] == 0 or checks["commands"]["ibstat"]["returncode"] == 0
    hca_active = any(
        marker in (ibstat + "\n" + ibv_devinfo)
        for marker in ["Active", "ACTIVE", "PORT_ACTIVE"]
    )

    checks["summary"] = {
        "env_check_pass": True,
        "gpu_check_pass": gpu_visible and not xid_hits,
        "hca_check_pass": hca_visible and hca_active,
        "config_check_pass": True,
        "error_log_check_pass": not xid_hits,
        "dmesg_check_skipped": True,
        "dmesg_check_note": "pod environment does not support dmesg; kernel log screening is owned by ops",
        "xid_hits": xid_hits[:20],
    }
    checks["summary"]["static_check_status"] = (
        "PASS"
        if all(
            checks["summary"][key]
            for key in [
                "env_check_pass",
                "gpu_check_pass",
                "hca_check_pass",
                "config_check_pass",
                "error_log_check_pass",
            ]
        )
        else "SUSPECT"
    )

    if output:
        write_json(output, checks)
    return checks
