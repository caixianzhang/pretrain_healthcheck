#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Result:
    mode: str
    direction: str
    host: str
    port: int
    duration_seconds: float
    parallel: int
    bytes: int
    gbps: float
    mibps: float
    started_at: float
    finished_at: float
    errors: list[str]


def metrics(
    mode: str,
    direction: str,
    host: str,
    port: int,
    duration: float,
    parallel: int,
    total_bytes: int,
    started: float,
    finished: float,
    errors: list[str],
) -> Result:
    elapsed = max(0.001, finished - started)
    return Result(
        mode=mode,
        direction=direction,
        host=host,
        port=port,
        duration_seconds=round(elapsed, 3),
        parallel=parallel,
        bytes=total_bytes,
        gbps=round(total_bytes * 8 / elapsed / 1_000_000_000, 6),
        mibps=round(total_bytes / elapsed / 1024 / 1024, 3),
        started_at=started,
        finished_at=finished,
        errors=errors,
    )


def print_result(result: Result) -> None:
    data = asdict(result)
    print(json.dumps(data, ensure_ascii=False, sort_keys=True), flush=True)
    print(
        f"{result.mode} {result.direction}: "
        f"{result.gbps:.3f} Gbps, {result.mibps:.1f} MiB/s, "
        f"bytes={result.bytes}, duration={result.duration_seconds}s, "
        f"parallel={result.parallel}, errors={len(result.errors)}",
        flush=True,
    )


def run_server(args: argparse.Namespace) -> int:
    stop_at = time.time() + args.duration + args.grace_seconds
    total_lock = threading.Lock()
    total_bytes = 0
    errors: list[str] = []
    threads: list[threading.Thread] = []

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(max(16, args.parallel * 2))
    server.settimeout(0.5)
    started = time.time()

    def handle(conn: socket.socket) -> None:
        nonlocal total_bytes
        local_bytes = 0
        try:
            conn.settimeout(1.0)
            while time.time() < stop_at:
                try:
                    chunk = conn.recv(args.buffer_size)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                local_bytes += len(chunk)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"server_conn_error={type(exc).__name__}:{exc}")
        finally:
            with total_lock:
                total_bytes += local_bytes
            try:
                conn.close()
            except OSError:
                pass

    try:
        while time.time() < stop_at:
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                if len(threads) >= args.parallel and all(not t.is_alive() for t in threads):
                    break
                continue
            thread = threading.Thread(target=handle, args=(conn,), daemon=True)
            thread.start()
            threads.append(thread)
            if len(threads) >= args.parallel:
                break
        for thread in threads:
            remaining = max(0.1, stop_at - time.time())
            thread.join(timeout=remaining)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"server_error={type(exc).__name__}:{exc}")
    finally:
        try:
            server.close()
        except OSError:
            pass

    finished = time.time()
    result = metrics(
        "server",
        args.direction,
        args.host,
        args.port,
        args.duration,
        args.parallel,
        total_bytes,
        started,
        finished,
        errors,
    )
    print_result(result)
    return 0 if not errors and total_bytes > 0 else 1


def run_client(args: argparse.Namespace) -> int:
    stop_at = time.time() + args.duration
    payload = b"\0" * args.buffer_size
    total_lock = threading.Lock()
    total_bytes = 0
    errors: list[str] = []
    barrier = threading.Barrier(args.parallel + 1)

    def worker(index: int) -> None:
        nonlocal total_bytes
        local_bytes = 0
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(args.connect_timeout)
            sock.connect((args.host, args.port))
            sock.settimeout(1.0)
            barrier.wait(timeout=10)
            while time.time() < stop_at:
                sent = sock.send(payload)
                if sent <= 0:
                    break
                local_bytes += sent
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        except Exception as exc:  # noqa: BLE001
            errors.append(f"client_{index}_error={type(exc).__name__}:{exc}")
            try:
                barrier.abort()
            except threading.BrokenBarrierError:
                pass
        finally:
            with total_lock:
                total_bytes += local_bytes
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    threads = [threading.Thread(target=worker, args=(idx,), daemon=True) for idx in range(args.parallel)]
    started = time.time()
    for thread in threads:
        thread.start()
    try:
        barrier.wait(timeout=10)
    except threading.BrokenBarrierError:
        pass
    for thread in threads:
        thread.join(timeout=args.duration + args.connect_timeout + 5)
    finished = time.time()

    result = metrics(
        "client",
        args.direction,
        args.host,
        args.port,
        args.duration,
        args.parallel,
        total_bytes,
        started,
        finished,
        errors,
    )
    print_result(result)
    return 0 if not errors and total_bytes > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple Python TCP bandwidth probe.")
    parser.add_argument("mode", choices=["server", "client"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=1024 * 1024)
    parser.add_argument("--direction", default="")
    parser.add_argument("--connect-timeout", type=float, default=5)
    parser.add_argument("--grace-seconds", type=float, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")
    if args.mode == "server":
        return run_server(args)
    return run_client(args)


if __name__ == "__main__":
    raise SystemExit(main())
