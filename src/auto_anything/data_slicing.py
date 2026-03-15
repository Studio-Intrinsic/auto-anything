from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def load_json_manifest(path: Path) -> list[dict[str, Any]]:
    return list(json.loads(path.read_text(encoding="utf-8")))


def write_json_manifest(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return path


def slice_rows(rows: list[dict[str, Any]], *, offset: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is None:
        return list(rows[offset:])
    if limit < 0:
        raise ValueError("limit must be non-negative when provided")
    return list(rows[offset : offset + limit])


def sample_rows(rows: list[dict[str, Any]], *, count: int, seed: int = 0) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), count))
    return [rows[index] for index in indices]


def stratified_sample_rows(
    rows: list[dict[str, Any]],
    *,
    count: int,
    key_fn: Callable[[dict[str, Any]], str],
    seed: int = 0,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count >= len(rows):
        return list(rows)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)
    bucket_keys = sorted(buckets)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    while len(selected) < count and bucket_keys:
        next_round: list[str] = []
        for bucket_key in bucket_keys:
            bucket = buckets[bucket_key]
            if not bucket:
                continue
            selected.append(bucket.pop(rng.randrange(len(bucket))))
            if len(selected) >= count:
                break
            if bucket:
                next_round.append(bucket_key)
        bucket_keys = next_round
    return selected


__all__ = [
    "load_json_manifest",
    "sample_rows",
    "slice_rows",
    "stratified_sample_rows",
    "write_json_manifest",
]
