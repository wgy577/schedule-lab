from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import ExperimentRecord


def append_record(record: ExperimentRecord, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(record.model_dump_json() + "\n")


def load_records(path: str | Path) -> list[ExperimentRecord]:
    source = Path(path)
    if not source.exists():
        return []
    return [
        ExperimentRecord.model_validate(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_records(records: Iterable[ExperimentRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
