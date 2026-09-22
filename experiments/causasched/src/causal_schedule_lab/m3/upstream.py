"""Upstream bridge: the four SHARED_REQUIRED experiment scripts that the whole
M3 pipeline depends on (frozen, never archived):

  - scripts/b5_1_train_pilot.py        (pilot)   M2 diagnostic / from_manifest_b5 / M2 runtime
  - scripts/t1_m3_makespan_utility.py  (m3util)  frozen SingleUtilityHead / DirectPairUtilityHead
  - scripts/run_v5_mainline_loop.py    (runmod)  D6 executor bridge (_d6_proposal_to_atoms / _d6_execute)
  - scripts/t1_b5_2_interaction_ranking.py (b52) proposal pool builder / pair features

These are loaded via importlib (the project's editable install is broken, see
memory [[venv-pth-hidden-flag-and-src-pythonpath]]: run with PYTHONPATH=src).

This module is the ONLY place in the canonical M3 package that touches scripts.
The legacy R-chain scripts (r1..r4, exploration-unblock, *_dbg_*, diag) are
ARCHIVED and MUST NOT be imported anywhere in src/ or the canonical entry point.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .config import ROOT

_SYS_PATH_SET = False


def _ensure_sys_path() -> None:
    global _SYS_PATH_SET
    if not _SYS_PATH_SET:
        for p in (str(ROOT / "src"), str(ROOT / "scripts")):
            if p not in sys.path:
                sys.path.insert(0, p)
        _SYS_PATH_SET = True


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


runmod = None
pilot = None
b52 = None
m3util = None
_LOADED = False


def load_upstream() -> None:
    """Load the four shared experiment scripts exactly once (importlib).
    Mirrors the legacy r1 loader order (runmod, pilot, b52, m3util)."""
    global runmod, pilot, b52, m3util, _LOADED
    if _LOADED:
        return
    _ensure_sys_path()
    runmod = _load("run_v5_mainline_loop", ROOT / "scripts" / "run_v5_mainline_loop.py")
    pilot = _load("b5_1_train_pilot", ROOT / "scripts" / "b5_1_train_pilot.py")
    b52 = _load("t1_b5_2_interaction_ranking", ROOT / "scripts" / "t1_b5_2_interaction_ranking.py")
    m3util = _load("t1_m3_makespan_utility", ROOT / "scripts" / "t1_m3_makespan_utility.py")
    _LOADED = True