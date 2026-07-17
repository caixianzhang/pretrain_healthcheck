#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


V1_PREFIX = "__HC_DYNAMIC_RESULT_JSON__ "
V2_PREFIX = "__HC_DYNAMIC_RESULT_V2__ "
CHUNK_MANIFEST_PREFIX = "__HC_DYNAMIC_CHUNK_MANIFEST__ "
CHUNK_PREFIX = "__HC_DYNAMIC_CHUNK__ "
DEFAULT_CHUNK_SIZE = 2048


class DynamicFrameError(ValueError):
    pass


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_v2_frame(payload: dict[str, Any]) -> str:
    raw = canonical_json_bytes(payload)
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    envelope = {
        "schema_version": 2,
        "encoding": "gzip+base64",
        "raw_bytes": len(raw),
        "raw_sha256": sha256_hex(raw),
        "compressed_bytes": len(compressed),
        "compressed_sha256": sha256_hex(compressed),
        "payload": base64.b64encode(compressed).decode("ascii"),
    }
    return V2_PREFIX + json.dumps(envelope, sort_keys=True, separators=(",", ":"))


def decode_frame_line(line: str) -> tuple[dict[str, Any], str]:
    if line.startswith(V1_PREFIX):
        try:
            payload = json.loads(line[len(V1_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise DynamicFrameError(f"legacy JSON decode failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise DynamicFrameError("legacy payload is not an object")
        return payload, "v1-json"
    if not line.startswith(V2_PREFIX):
        raise DynamicFrameError("unknown dynamic frame prefix")
    try:
        envelope = json.loads(line[len(V2_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise DynamicFrameError(f"V2 envelope JSON decode failed: {exc}") from exc
    if not isinstance(envelope, dict):
        raise DynamicFrameError("V2 envelope is not an object")
    if envelope.get("schema_version") != 2 or envelope.get("encoding") != "gzip+base64":
        raise DynamicFrameError("unsupported V2 envelope")
    try:
        compressed = base64.b64decode(str(envelope.get("payload", "")), validate=True)
    except Exception as exc:
        raise DynamicFrameError(f"V2 base64 decode failed: {exc}") from exc
    if len(compressed) != int(envelope.get("compressed_bytes", -1)):
        raise DynamicFrameError("V2 compressed length mismatch")
    if sha256_hex(compressed) != str(envelope.get("compressed_sha256", "")):
        raise DynamicFrameError("V2 compressed SHA256 mismatch")
    try:
        raw = gzip.decompress(compressed)
    except Exception as exc:
        raise DynamicFrameError(f"V2 gzip decode failed: {exc}") from exc
    if len(raw) != int(envelope.get("raw_bytes", -1)):
        raise DynamicFrameError("V2 raw length mismatch")
    if sha256_hex(raw) != str(envelope.get("raw_sha256", "")):
        raise DynamicFrameError("V2 raw SHA256 mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DynamicFrameError(f"V2 payload JSON decode failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise DynamicFrameError("V2 payload is not an object")
    return payload, "v2-gzip-base64"


def atomic_write_frame(path: Path, frame_line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(frame_line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, 0o444)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def chunk_manifest(data: bytes, chunk_size: int) -> dict[str, Any]:
    if chunk_size < 256:
        raise DynamicFrameError("chunk_size must be >= 256")
    total = (len(data) + chunk_size - 1) // chunk_size
    return {
        "schema_version": 1,
        "file_bytes": len(data),
        "file_sha256": sha256_hex(data),
        "chunk_size": chunk_size,
        "total_chunks": total,
    }


def iter_chunks(data: bytes, chunk_size: int, indexes: Iterable[int] | None = None) -> Iterable[dict[str, Any]]:
    manifest = chunk_manifest(data, chunk_size)
    total = int(manifest["total_chunks"])
    selected = list(range(total)) if indexes is None else sorted(set(indexes))
    for index in selected:
        if index < 0 or index >= total:
            raise DynamicFrameError(f"chunk index out of range: {index}")
        start = index * chunk_size
        chunk = data[start : start + chunk_size]
        yield {
            "index": index,
            "total_chunks": total,
            "offset": start,
            "chunk_bytes": len(chunk),
            "chunk_sha256": sha256_hex(chunk),
            "payload": base64.b64encode(chunk).decode("ascii"),
        }


def parse_indexes(text: str) -> list[int] | None:
    if not text:
        return None
    return [int(item) for item in text.split(",") if item.strip()]


def emit_chunks(path: Path, chunk_size: int, indexes: list[int] | None) -> None:
    data = path.read_bytes()
    print(CHUNK_MANIFEST_PREFIX + json.dumps(chunk_manifest(data, chunk_size), sort_keys=True, separators=(",", ":")))
    for row in iter_chunks(data, chunk_size, indexes):
        print(CHUNK_PREFIX + json.dumps(row, sort_keys=True, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(description="dynamic compact frame protocol helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)
    chunks = subparsers.add_parser("emit-chunks")
    chunks.add_argument("--path", type=Path, required=True)
    chunks.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    chunks.add_argument("--indexes", default="")
    args = parser.parse_args()
    if args.command == "emit-chunks":
        emit_chunks(args.path, args.chunk_size, parse_indexes(args.indexes))


if __name__ == "__main__":
    main()
