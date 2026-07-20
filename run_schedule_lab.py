#!/usr/bin/env python3
"""Repository-local CLI entry point that works without an editable install."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from schedule_lab.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
