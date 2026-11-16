#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from meta_fdg.data.compact_qlib import (
    DEVELOPMENT_PARTITION,
    TEST_PARTITION,
    build_source_handler,
    compact_store_ids,
    load_source_handler,
    materialize_compact_handler,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve_config(
    config: Mapping[str, Any], *, repo_root: Path, cache_root: Path
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    data = resolved["data"]
    for key in ("provider_path", "cache_root"):
        if key not in data:
            continue
        value = Path(str(data[key])).expanduser()
        if not value.is_absolute():
            value = repo_root / value
        data[key] = str(value.resolve())
    data["materialized_cache_root"] = str(cache_root.resolve())
    data["use_compact_materialization"] = True
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize seed-shared compact Qlib factor/label stores."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = _read_json(manifest_path)
    cache_root = Path(str(manifest["cache_root"])).expanduser().resolve()
    if not cache_root.is_absolute():
        raise ValueError("manifest cache_root must be absolute")
    partitions = tuple(
        str(value)
        for value in manifest.get(
            "partitions", [DEVELOPMENT_PARTITION, TEST_PARTITION]
        )
    )
    if set(partitions) - {DEVELOPMENT_PARTITION, TEST_PARTITION}:
        raise ValueError(f"unsupported materialization partitions: {partitions}")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for entry in manifest["entries"]:
        config_path = Path(str(entry["config"])).expanduser().resolve()
        repo_root = Path(str(entry["repo_root"])).expanduser().resolve()
        config = _resolve_config(
            _read_json(config_path), repo_root=repo_root, cache_root=cache_root
        )
        factor_id, label_id = compact_store_ids(config)
        key = (factor_id, label_id, partitions)
        if key in seen:
            rows.append(
                {
                    "status": "deduplicated",
                    "config": str(config_path),
                    "repo_root": str(repo_root),
                    "factor_id": factor_id,
                    "label_id": label_id,
                    "partitions": list(partitions),
                }
            )
            continue
        seen.add(key)
        source_value = entry.get("source_handler")
        if source_value is None:
            handler = build_source_handler(config, partitions=partitions)
            source = {
                "kind": "built_from_raw_provider",
                "provider_path": config["data"]["provider_path"],
                "materialized_partitions": list(partitions),
            }
        else:
            source_path = Path(str(source_value)).expanduser().resolve()
            handler = load_source_handler(source_path)
            source = {
                "kind": "existing_qlib_handler",
                "path": str(source_path),
                "bytes": int(source_path.stat().st_size),
                "sha256": _sha256(source_path),
            }
        audit = materialize_compact_handler(
            config,
            handler,
            cache_root=cache_root,
            source=source,
            partitions=partitions,
        )
        rows.append(
            {
                "status": "materialized",
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "resolved_config_sha256": _canonical_sha256(config),
                "repo_root": str(repo_root),
                "job_id": config.get("job_id"),
                "method": config.get("method"),
                "seed": config.get("seed"),
                **audit,
            }
        )
        del handler
    output = {
        "schema": "meta_fdg_compact_materialization_batch_v1",
        "created_at_utc": _utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "cache_root": str(cache_root),
        "partitions": list(partitions),
        "unique_store_pairs": len(seen),
        "entries": rows,
        "status": "complete",
    }
    _atomic_json(args.output.expanduser().resolve(), output)
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
