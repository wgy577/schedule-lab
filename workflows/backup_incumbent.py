#!/usr/bin/env python3
"""Create an immutable, checksummed snapshot of a validated incumbent schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_schedule_sha256(payload: dict) -> str:
    schedule = payload.get("schedule", payload)
    encoded = json.dumps(
        schedule,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def copy_with_layout(source: Path, root: Path, destination: Path) -> list[Path]:
    if source.is_dir():
        target = destination / source.relative_to(root)
        shutil.copytree(source, target, dirs_exist_ok=True)
        return sorted(path for path in target.rglob("*") if path.is_file())

    target = destination / source.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return [target]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--include", type=Path, action="append", default=[])
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    incumbent = args.incumbent.resolve()
    backup_dir = (args.backup_root / args.name).resolve()
    if backup_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing backup: {backup_dir}")
    if project_root not in incumbent.parents:
        raise SystemExit("Incumbent must be inside project root")

    payload = json.loads(incumbent.read_text(encoding="utf-8"))
    schedule = payload.get("schedule", [])
    true_makespan = payload.get("meta", {}).get("trueMakespan")
    if true_makespan is None and schedule:
        true_makespan = max(float(operation["end"]) for operation in schedule)

    sources = [incumbent, *[path.resolve() for path in args.include]]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise SystemExit(f"Missing backup sources: {missing}")

    backup_dir.mkdir(parents=True)
    copied: list[Path] = []
    for source in sources:
        if project_root not in source.parents and source != project_root:
            raise SystemExit(f"Backup source must be inside project root: {source}")
        copied.extend(copy_with_layout(source, project_root, backup_dir))

    manifest = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "Frozen incumbent before adding Oracle-cut guided local improvement",
        "incumbent": {
            "source": str(incumbent.relative_to(project_root)),
            "trueMakespan": true_makespan,
            "operationCount": len(schedule),
            "normalizedScheduleSha256": normalized_schedule_sha256(payload),
            "domainConstructed": payload.get("meta", {}).get("domainConstructed"),
            "genericValidated": payload.get("meta", {}).get("genericValidated"),
        },
        "files": [
            {
                "path": str(path.relative_to(backup_dir)),
                "sha256": file_sha256(path),
                "sizeBytes": path.stat().st_size,
            }
            for path in sorted(set(copied))
        ],
        "restorePolicy": (
            "Restore files only after verifying their SHA-256 values. "
            "Never overwrite this backup in place."
        ),
    }
    manifest_path = backup_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    print(json.dumps(manifest["incumbent"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
