"""T2-B-GPU -- three-tier resolved config for the dedicated GPU training edition.

Independent of the canonical Mac/CPU path.  Priority:
    canonical defaults  <  --config <json>  <  CLI explicit override

Three tiers (per the T2-B-GPU directive):
    SCIENTIFIC_FROZEN  -- immutable without --unsafe-research-override.
    RESEARCH_TUNABLE   -- CLI-mutable, defaults = current frozen T2-B/R13 canonical.
    SCALE_TUNABLE      -- free: training budget / throughput / monitoring knobs.

Every run saves the FULL resolved config (config.json) + records which fields
were user-overridden vs auto-set, so the final report can separate
CANONICAL FROZEN DEFAULT / USER OVERRIDES / AUTO SYSTEM SETTINGS.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from causal_schedule_lab.m3 import config as C


# ---------------------------------------------------------------------------
# canonical default getters (live reads -- never re-typed frozen values)
# ---------------------------------------------------------------------------
def canonical_research_defaults() -> dict[str, Any]:
    """RESEARCH_TUNABLE defaults == current frozen canonical T2-B / R13 values."""
    return {
        "k": int(C.TO1_T2B_K),
        "horizon": int(C.TO1_T2B_HORIZON),
        "temperature": float(C.TO1_T2B_TEMP),
        "mixture_eps": float(C.TO1_T2B_MIX_EPS),
        "lr": float(C.TO1_R13_LR_M3),          # canonical joint update lr (stage C)
        "clip_eps": float(C.TO1_R13_CLIP_EPS),
        "kl_beta_m2": float(C.TO1_R14_BETA_M2),
        "kl_beta_m3": float(C.TO1_R14_BETA_M3),
        "update_epochs": int(C.TO1_R13_UPDATE_EPOCHS),
        "max_grad_norm": 10.0,                # grpo_update_joint clip_grad_norm_
        "temp_m2": float(C.TO1_R13_TEMP_M2),
    }


def canonical_scale_defaults() -> dict[str, Any]:
    """SCALE_TUNABLE defaults (canonical T2-B budgets where one exists)."""
    return {
        "cycles": int(C.TO1_T2B_TRAINING_CYCLES),
        "graphs_per_cycle": int(C.TO1_T2B_GRAPHS_PER_BATCH),
        "workers": "auto",                    # resolved by preflight/profile
        "device": "auto",                     # auto->cuda iff available else cpu
        "gpu_batch_size": 256,
        "gpu_microbatch": 0,                  # 0 -> gpu_batch_size
        "prefetch_depth": 0,                  # v1 synchronous collect (documented)
        "checkpoint_interval": 5,
        "save_interval": 5,
        "eval_interval": 1,
        "log_interval": 1,
        "train_source": "all",                # benchmark,aux-real,aux-syn,all
        "sampling_mode": "canonical",          # canonical|instance-balanced|family-balanced
        "seed": int(C.TO1_T2B_SEED_BASE),
        "run_name": None,
        "run_id": None,
        "resume": None,                       # path to a resume checkpoint
        "profile_workers": False,
        "unsafe_research_override": False,
        "halt_on_instability": False,
        "max_environment_interactions": 0,    # 0 = unlimited
        "plateau_patience": 0,                # 0 = canonical stopping rule only
        "warn_max_kl": 10.0,
        "warn_max_grad_norm": 50.0,
        "warn_min_diversity": 0.01,
    }


# Scientific-frozen literal values -- the GPU edition implements these verbatim
# and a normal run can NEVER change them (CLI has no flags for them).
SCIENTIFIC_FROZEN_LITERALS: dict[str, Any] = {
    "action_space": "lexicographic",     # runtime VERBATIM (canonical)
    "credit": "stagewise",               # A2/A3 separation verbatim
    "reward_definition": "R_i = Cmax(S_root) - Cmax(S_terminal_i)",
    "best_of_n_reward": False,           # post-hoc diagnostic only, never reward
    "same_state_group_relative": True,   # group_advantages_r12 semantics
    "old_new_clipped_ratio": True,       # PPO clipped-ratio semantics
    "identified": False,                 # NEVER flips
    "formal_test_access": 0,             # NEVER flips; no Formal TEST loader exists
}


def next_run_id(base_dir: Path) -> str:
    """t2b_gpu_<YYYYMMDD>_<NNN> -- next free id under base_dir/runs."""
    stem = "t2b_gpu_" + datetime.datetime.now().strftime("%Y%m%d")
    n = 1
    while True:
        cand = f"{stem}_{n:03d}"
        if not (base_dir / "runs" / cand).exists():
            return cand
        n += 1


def tiers() -> dict[str, list[str]]:
    return {
        "scientific_frozen": sorted(SCIENTIFIC_FROZEN_LITERALS),
        "research_tunable": sorted(canonical_research_defaults()),
        "scale_tunable": sorted(canonical_scale_defaults()),
    }


# ---------------------------------------------------------------------------
@dataclass
class T2BGPUConfig:
    """Resolved GPU training configuration.

    Fields mirror the canonical frozen values by default (live reads at __init__
    via default_factory).  `overrides` records user (CLI/file) vs canonical, and
    `auto` records auto-set fields -- the final report splits
    CANONICAL FROZEN DEFAULT / USER OVERRIDES / AUTO SYSTEM SETTINGS.
    """

    # --- research tunable -------------------------------------------------
    k: int = field(default_factory=lambda: int(C.TO1_T2B_K))
    horizon: int = field(default_factory=lambda: int(C.TO1_T2B_HORIZON))
    temperature: float = field(default_factory=lambda: float(C.TO1_T2B_TEMP))
    mixture_eps: float = field(default_factory=lambda: float(C.TO1_T2B_MIX_EPS))
    lr: float = field(default_factory=lambda: float(C.TO1_R13_LR_M3))
    clip_eps: float = field(default_factory=lambda: float(C.TO1_R13_CLIP_EPS))
    kl_beta_m2: float = field(default_factory=lambda: float(C.TO1_R14_BETA_M2))
    kl_beta_m3: float = field(default_factory=lambda: float(C.TO1_R14_BETA_M3))
    update_epochs: int = field(default_factory=lambda: int(C.TO1_R13_UPDATE_EPOCHS))
    max_grad_norm: float = 10.0
    temp_m2: float = field(default_factory=lambda: float(C.TO1_R13_TEMP_M2))

    # --- scale tunable ----------------------------------------------------
    cycles: int = field(default_factory=lambda: int(C.TO1_T2B_TRAINING_CYCLES))
    graphs_per_cycle: int = field(
        default_factory=lambda: int(C.TO1_T2B_GRAPHS_PER_BATCH))
    graphs_per_batch: int | None = None
    workers: str | int = "auto"
    device: str = "auto"                       # "auto"|"cuda"|"cpu"
    gpu_batch_size: int = 256
    gpu_microbatch: int = 0
    prefetch_depth: int = 0
    checkpoint_interval: int = 5
    save_interval: int = 5
    eval_interval: int = 1
    log_interval: int = 1
    train_source: str = "all"
    sampling_mode: str = "canonical"
    seed: int = field(default_factory=lambda: int(C.TO1_T2B_SEED_BASE))
    run_name: str | None = None
    run_id: str | None = None
    resume: str | None = None
    profile_workers: bool = False
    unsafe_research_override: bool = False
    halt_on_instability: bool = False
    max_environment_interactions: int = 0
    plateau_patience: int = 0
    warn_max_kl: float = 10.0
    warn_max_grad_norm: float = 50.0
    warn_min_diversity: float = 0.01

    # --- bookkeeping (saved but not settings) -----------------------------
    overrides: dict[str, str] = field(default_factory=dict, repr=False)
    auto: dict[str, str] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = {}
        for f in dataclasses.fields(self):
            if f.name in ("overrides", "auto"):
                continue
            d[f.name] = getattr(self, f.name)
        return d

    SAVED_NONFIELD = {"overrides", "auto"}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "T2BGPUConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "T2BGPUConfig":
        cfg = cls()
        for k, v in d.items():
            if k in cls.SAVED_NONFIELD or k == "defaults":
                continue
            if k in {f.name for f in dataclasses.fields(cls)}:
                setattr(cfg, k, v)
        return cfg

    # ------------------------------------------------------------------
    # tier lookup
    # ------------------------------------------------------------------
    @staticmethod
    def tier_of(key: str) -> str:
        for tier, keys in tiers().items():
            if key in keys:
                return tier
        return "unknown"

    # ------------------------------------------------------------------
    # file/CLI application (file < CLI)
    # ------------------------------------------------------------------
    def apply_file(self, path: Path) -> None:
        d = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in dataclasses.fields(self)}
        for k, v in d.items():
            if k in self.SAVED_NONFIELD or k not in known:
                raise ValueError(f"config file {path}: unknown key {k!r} (fail fast)")
            self._set_tunable(k, v, source="file")

    def apply_cli(self, ns: argparse.Namespace) -> None:
        known = {f.name for f in dataclasses.fields(self)}
        cli_values = {k: v for k, v in vars(ns).items()
                      if k in known and v is not None and k != "config"}
        for k, v in cli_values.items():
            self._set_tunable(k, v, source="cli")

    def _set_tunable(self, key: str, value: Any, source: str) -> None:
        tier = self.tier_of(key)
        if tier == "scientific_frozen":
            raise ValueError(
                f"attempt to change SCIENTIFIC_FROZEN {key!r} "
                f"without --unsafe-research-override (fail fast)")
        setattr(self, key, value)
        self.overrides[key] = f"{source}: {value}"

    # ------------------------------------------------------------------
    # resolution (what the run actually executes with)
    # ------------------------------------------------------------------
    def resolve(self) -> None:
        """Compute auto-set fields (device/workers/graphs_per_batch/microbatch/
        run_id/run_name) + validate.  Fails fast on CUDA-required-missing."""
        import torch

        # device -- the GPU edition defaults to CUDA-first
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.auto["device"] = f"auto -> {device}"
        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested (--device cuda / auto-on-GPU) but unavailable; "
                "the dedicated GPU runner FAILS FAST rather than silently "
                "falling back to CPU.  --device cpu is for debug only.")

        # workers (auto resolved at preflight on the host carrying the data)
        if isinstance(self.workers, str) and self.workers.lower() == "auto":
            self.auto["workers"] = "auto (resolved by preflight)"
        else:
            self.workers = int(self.workers)

        # dependent scale fields
        if not self.gpu_microbatch:
            self.gpu_microbatch = int(self.gpu_batch_size)
            self.auto["gpu_microbatch"] = f"0 -> {self.gpu_microbatch}"
        if not self.graphs_per_batch:
            self.graphs_per_batch = int(self.graphs_per_cycle)
            self.auto["graphs_per_batch"] = f"None -> {self.graphs_per_batch}"

        # run identity
        if not self.run_id:
            self.run_id = next_run_id(C.CANONICAL_OUT_DIR)
            self.auto["run_id"] = self.run_id
        if not self.run_name:
            self.run_name = self.run_id
            self.auto["run_name"] = self.run_name

        # enumeration validation
        if self.sampling_mode not in ("canonical", "instance-balanced",
                                      "family-balanced"):
            raise ValueError(f"unknown sampling_mode {self.sampling_mode!r}")
        for src in str(self.train_source).split(","):
            if src not in ("benchmark", "aux-real", "aux-syn", "all", ""):
                raise ValueError(f"unknown train-source {src!r}")

        # force frozen flags to their literals no matter what
        self.identified = False
        self.formal_test_access = 0
        self.credit = SCIENTIFIC_FROZEN_LITERALS["credit"]
        self.action_space = SCIENTIFIC_FROZEN_LITERALS["action_space"]

    # scientific-frozen fields carried in the resolved config (set in resolve)
    action_space: str = SCIENTIFIC_FROZEN_LITERALS["action_space"]
    credit: str = SCIENTIFIC_FROZEN_LITERALS["credit"]
    reward_definition: str = SCIENTIFIC_FROZEN_LITERALS["reward_definition"]
    best_of_n_reward: bool = SCIENTIFIC_FROZEN_LITERALS["best_of_n_reward"]
    same_state_group_relative: bool = SCIENTIFIC_FROZEN_LITERALS[
        "same_state_group_relative"]
    old_new_clipped_ratio: bool = SCIENTIFIC_FROZEN_LITERALS["old_new_clipped_ratio"]
    identified: bool = SCIENTIFIC_FROZEN_LITERALS["identified"]
    formal_test_access: int = SCIENTIFIC_FROZEN_LITERALS["formal_test_access"]