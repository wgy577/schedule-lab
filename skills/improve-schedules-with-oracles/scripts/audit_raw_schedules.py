#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FIELDS = ("job", "op", "machine", "start", "dur", "end")


def load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schedule = payload.get("schedule", payload.get("operations", []))
    if not isinstance(schedule, list) or not schedule:
        raise ValueError(f"no raw schedule found in {path}")
    return schedule


def digest(schedule: list[dict]) -> str:
    normalized = [
        {field: item[field] for field in FIELDS}
        for item in sorted(schedule, key=lambda item: (int(item["job"]), int(item["op"])))
    ]
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit two legacy raw schedule JSON files")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left = load(args.left)
    right = load(args.right)
    left_by_id = {(int(item["job"]), int(item["op"])): item for item in left}
    right_by_id = {(int(item["job"]), int(item["op"])): item for item in right}
    if set(left_by_id) != set(right_by_id):
        raise SystemExit("operation identity sets differ")
    left_hash, right_hash = digest(left), digest(right)
    if left_hash == right_hash:
        raise SystemExit("schedules are identical")
    machine_changes = [
        {
            "job": operation_id[0],
            "op": operation_id[1],
            "left": int(left_by_id[operation_id]["machine"]),
            "right": int(right_by_id[operation_id]["machine"]),
        }
        for operation_id in sorted(left_by_id)
        if int(left_by_id[operation_id]["machine"]) != int(right_by_id[operation_id]["machine"])
    ]
    timing_changes = sum(
        abs(float(left_by_id[operation_id]["start"]) - float(right_by_id[operation_id]["start"])) > 1e-6
        or abs(float(left_by_id[operation_id]["end"]) - float(right_by_id[operation_id]["end"])) > 1e-6
        for operation_id in left_by_id
    )
    report = {
        "left": {
            "path": str(args.left.resolve()),
            "hash": left_hash,
            "makespan": max(float(item["end"]) for item in left),
        },
        "right": {
            "path": str(args.right.resolve()),
            "hash": right_hash,
            "makespan": max(float(item["end"]) for item in right),
        },
        "machineBindingChanges": machine_changes,
        "timingChangeCount": timing_changes,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
