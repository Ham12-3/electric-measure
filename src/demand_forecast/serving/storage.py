"""Optional S3 sync for cloud deployment.

Locally everything lives on disk. In the cloud, containers are disposable, so the daily
pipeline pushes its outputs to S3 and the API pulls them:

    pipeline: pull (raw cache, so downloads stay incremental) -> run -> push
    api:      pull models + processed data at start-up, and again every few hours

Enabled by setting ARTIFACT_BUCKET (e.g. "my-bucket" or "my-bucket/prefix").
Without it, every function here is a no-op.
"""

from __future__ import annotations

import logging
import os

from demand_forecast.config import PROJECT_ROOT

log = logging.getLogger(__name__)

SERVING_DIRS = ["artifacts/models", "data/processed"]
PIPELINE_DIRS = SERVING_DIRS + ["data/raw"]


def _bucket() -> tuple[str, str] | None:
    value = os.environ.get("ARTIFACT_BUCKET", "").strip().removeprefix("s3://").strip("/")
    if not value:
        return None
    bucket, _, prefix = value.partition("/")
    return bucket, prefix


def enabled() -> bool:
    return _bucket() is not None


def _client():
    import boto3
    return boto3.client("s3")


def pull(dirs: list[str] = SERVING_DIRS) -> int:
    """Download every object under each dir. Returns the number of files fetched."""
    target = _bucket()
    if target is None:
        return 0
    bucket, prefix = target
    s3, n = _client(), 0
    for d in dirs:
        key_prefix = f"{prefix}/{d}/" if prefix else f"{d}/"
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix) + 1:] if prefix else obj["Key"]
                dest = PROJECT_ROOT / rel
                if dest.exists() and dest.stat().st_size == obj["Size"] and d == "data/raw":
                    continue  # raw cache files are immutable once complete
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, obj["Key"], str(dest))
                n += 1
    log.info("pulled %d files from s3://%s/%s", n, bucket, prefix)
    return n


def push(dirs: list[str] = PIPELINE_DIRS) -> int:
    target = _bucket()
    if target is None:
        return 0
    bucket, prefix = target
    s3, n = _client(), 0
    for d in dirs:
        root = PROJECT_ROOT / d
        for path in (p for p in root.rglob("*") if p.is_file() and not p.name.endswith(".tmp")):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            s3.upload_file(str(path), bucket, f"{prefix}/{rel}" if prefix else rel)
            n += 1
    log.info("pushed %d files to s3://%s/%s", n, bucket, prefix)
    return n
