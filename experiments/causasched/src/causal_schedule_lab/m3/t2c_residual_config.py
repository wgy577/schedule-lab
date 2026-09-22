"""T2-C -- residual-policy GPU config (extends T2-B-GPU's three-tier config).

Adds the residual experiment fields on RESEARCH_TUNABLE tier (they are architecture
hyper-parameters: changing them changes the checkpoint architecture, so they are
frozen into config.json + checkpoint arch metadata + resume fail-fast).

Scientific-frozen literals are inherited from T2-B-GPU VERBATIM and remain
immutable without --unsafe-research-override:
    reward  = R_i = Cmax(S_root)-Cmax(S_terminal_i)
    credit  = stagewise (A2 for M2, A3 for M3)
    action_space = lexicographic, old/new clipped ratio, same-state group-relative
    identified = False, formal_test_access = 0 (Formal TEST sealed)
No best-of-N reward, no GH/G1 shaping, no oracle features.  The residual cap is
calibrated ONLY on train-state pool base logits (source recorded in
residual_scale_calibration.json).
"""

from __future__ import annotations

import dataclasses
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causal_schedule_lab.m3 import config as C
from causal_schedule_lab.m3.t2b_gpu_config import (
    T2BGPUConfig,
    tiers as _t2b_tiers,
)

# residual fields: research_tunable (architecture identity; saved + arch-hashed).
RESIDUAL_TUNABLE = [
    "residual_mode", "residual_hidden", "residual_depth", "residual_cap",
    "residual_alpha_target", "residual_alpha_stop", "residual_warmup_fraction",
    "residual_cap_samples",
]

RESIDUAL_MODES = ("tiny", "large", "controlled")


@dataclass
class T2CResidualConfig(T2BGPUConfig):
    """T2-C resolved config == T2-B-GPU config + residual experiment fields."""

    # --- residual architecture (research tunable) -------------------------
    residual_mode: str = "tiny"                    # tiny|large|controlled (C0/C1/C2)
    residual_hidden: int = 256                     # MLP hidden width (C1/C2)
    residual_depth: int = 3                        # pre-registered 3 hidden layers (C1/C2):
                                                   # L->256 GELU -> 256 GELU -> 256 GELU -> 1
    residual_cap: float | str = "auto"             # auto|float (controlled only)
    residual_alpha_target: float = float(C.TO1_R19_ALPHA_TEMPORAL)
    residual_alpha_stop: float = float(C.TO1_R19_ALPHA_TEMPORAL)
    residual_warmup_fraction: float = 0.25         # of total cycles (monotone linear)
    residual_cap_samples: int = 48                 # TRAIN-only base-logit states

    # ------------------------------------------------------------------
    @staticmethod
    def tier_of(key: str) -> str:
        base = _t2b_tiers()
        for tier, keys in base.items():
            if key in keys:
                return tier
        if key in RESIDUAL_TUNABLE:
            return "research_tunable"
        return "unknown"

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = {}
        for f in dataclasses.fields(self):
            if f.name in ("overrides", "auto"):
                continue
            d[f.name] = getattr(self, f.name)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "T2CResidualConfig":
        cfg = cls()
        for k, v in d.items():
            if k in cls.SAVED_NONFIELD or k == "defaults":
                continue
            if k in {f.name for f in dataclasses.fields(cls)}:
                setattr(cfg, k, v)
        return cfg

    # ------------------------------------------------------------------
    def resolve(self) -> None:
        # run_id prefix distinct for this edition + mode (before base resolve()
        # auto-assigns a t2b_gpu_ id)
        if not self.run_id and "run_id" not in self.overrides:
            self.run_id = _next_t2c_run_id(C.CANONICAL_OUT_DIR,
                                           str(self.residual_mode))
        super().resolve()
        self.residual_mode = str(self.residual_mode)
        if self.residual_mode not in RESIDUAL_MODES:
            raise ValueError(
                f"residual_mode must be one of {RESIDUAL_MODES}, got "
                f"{self.residual_mode!r} (fail fast)")
        if self.residual_mode == "tiny":
            # C0 == canonical M3TemporalEvidencePolicy; no MLP knobs apply.
            self.auto["residual_hidden"] = "ignored (tiny/C0 uses canonical Linear)"
            self.auto["residual_depth"] = "ignored (tiny/C0 uses canonical Linear)"
            self.auto["residual_cap"] = "ignored (tiny/C0 has no cap)"
        elif self.residual_mode in ("large", "controlled"):
            if int(self.residual_hidden) < 1 or int(self.residual_depth) < 1:
                raise ValueError("residual_hidden>=1 and residual_depth>=1 required "
                                 f"(got {self.residual_hidden}/{self.residual_depth})")
        else:
            raise ValueError(f"residual_mode unknown: {self.residual_mode!r}")
        if str(self.residual_cap) != "auto":
            try:
                self.residual_cap = float(self.residual_cap)
            except (TypeError, ValueError):
                raise ValueError(f"residual_cap must be 'auto' or a float, got "
                                 f"{self.residual_cap!r} (fail fast)") from None
        if not (0.0 <= float(self.residual_warmup_fraction) <= 1.0):
            raise ValueError(f"residual_warmup_fraction must be in [0, 1], got "
                             f"{self.residual_warmup_fraction!r}")
        if int(self.residual_cap_samples) < 1:
            raise ValueError(f"residual_cap_samples must be >=1, got "
                             f"{self.residual_cap_samples!r}")
        if self.residual_mode == "controlled" and str(self.residual_cap) == "auto":
            self.auto["residual_cap"] = "AUTO (train-only calibration sweep)"
        elif self.residual_mode == "controlled":
            self.auto["residual_cap"] = "user-specified (no calibration sweep)"


def _next_t2c_run_id(base_dir: Path, mode: str) -> str:
    distance = "C" + {"tiny": "0", "large": "1", "controlled": "2"}.get(mode, mode)
    stem = f"t2c{distance}_" + datetime.datetime.now().strftime("%Y%m%d")
    n = 1
    while True:
        cand = f"{stem}_{n:03d}"
        if not (base_dir / "runs" / cand).exists():
            return cand
        n += 1