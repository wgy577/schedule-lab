from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .carrier_worker import DEFAULT_LEGACY_ROOT


def search_carrier_policy(
    *,
    rollouts: int = 8,
    seed: int = 0,
    device: str = "cpu",
    legacy_root: str | Path = DEFAULT_LEGACY_ROOT,
) -> dict[str, Any]:
    """Searches the legacy policy in an isolated, collision-aware subprocess.

    Isolation is intentional: the legacy project changes argv, cwd, sys.path,
    environment variables, and module globals. Keeping those changes outside
    the optimizer/MCP process makes repeated calls reproducible.
    """

    if rollouts < 1:
        raise ValueError("rollouts must be at least 1")
    with tempfile.TemporaryDirectory(prefix="schedule-lab-carrier-") as directory:
        output = Path(directory) / "result.json"
        command = [
            sys.executable,
            "-m",
            "schedule_lab.carrier_worker",
            "--legacy-root",
            str(Path(legacy_root).expanduser().resolve()),
            "--rollouts",
            str(rollouts),
            "--seed",
            str(seed),
            "--device",
            device,
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "unknown carrier worker failure"
            raise RuntimeError(message)
        return json.loads(output.read_text(encoding="utf-8"))


def save_carrier_search(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
