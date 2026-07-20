from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .carrier_worker import DEFAULT_LEGACY_ROOT
from .carrier_vns_worker import ALNS_OPERATORS


def search_carrier_alns(
    incumbent: str | Path,
    *,
    destroy_sizes: Sequence[int] = (2, 3),
    destroy_radius: int = 2,
    max_gaps: int = 4,
    max_jobs: int = 14,
    gap_ranks: Sequence[int] = (),
    expansion_jobs: Sequence[int] = (),
    operators: Sequence[str] = ALNS_OPERATORS,
    seed: int = 0,
    device: str = "cpu",
    legacy_root: str | Path = DEFAULT_LEGACY_ROOT,
    oracle_cuts: str | Path | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="schedule-lab-carrier-alns-") as directory:
        output = Path(directory) / "result.json"
        command = [
            sys.executable,
            "-m",
            "schedule_lab.carrier_vns_worker",
            "--incumbent",
            str(Path(incumbent).expanduser().resolve()),
            "--legacy-root",
            str(Path(legacy_root).expanduser().resolve()),
            "--plan-kind",
            "alns",
            "--destroy-sizes",
            *map(str, destroy_sizes),
            "--destroy-radius",
            str(destroy_radius),
            "--max-gaps",
            str(max_gaps),
            "--max-jobs",
            str(max_jobs),
            "--gap-ranks",
            *map(str, gap_ranks),
            "--expansion-jobs",
            *map(str, expansion_jobs),
            "--operators",
            *operators,
            "--seed",
            str(seed),
            "--device",
            device,
            "--output",
            str(output),
        ]
        if oracle_cuts is not None:
            command.extend(
                ["--oracle-cuts", str(Path(oracle_cuts).expanduser().resolve())]
            )
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "unknown carrier ALNS worker failure"
            raise RuntimeError(message)
        return json.loads(output.read_text(encoding="utf-8"))
