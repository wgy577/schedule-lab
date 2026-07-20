from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .carrier_worker import DEFAULT_LEGACY_ROOT
from .carrier_vns_worker import VNS_OPERATORS


def search_carrier_vns(
    incumbent: str | Path,
    *,
    radii: Sequence[int] = (2, 3, 4),
    max_gaps: int = 1,
    operators: Sequence[str] = VNS_OPERATORS,
    seed: int = 0,
    device: str = "cpu",
    legacy_root: str | Path = DEFAULT_LEGACY_ROOT,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="schedule-lab-carrier-vns-") as directory:
        output = Path(directory) / "result.json"
        command = [
            sys.executable,
            "-m",
            "schedule_lab.carrier_vns_worker",
            "--incumbent",
            str(Path(incumbent).expanduser().resolve()),
            "--legacy-root",
            str(Path(legacy_root).expanduser().resolve()),
            "--radii",
            *map(str, radii),
            "--max-gaps",
            str(max_gaps),
            "--operators",
            *operators,
            "--seed",
            str(seed),
            "--device",
            device,
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "unknown carrier VNS worker failure"
            raise RuntimeError(message)
        return json.loads(output.read_text(encoding="utf-8"))
