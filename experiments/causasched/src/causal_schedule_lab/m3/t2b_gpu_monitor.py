"""T2-B-GPU -- TensorBoard + metrics.jsonl + live console + system stats.

Bundles every monitoring channel the GPU edition requires (§32-55):
  * TensorBoard `SummaryWriter` per run under outputs/canonical_m3/tensorboard/t2b_gpu/<run_id>,
    with hparams at run start and the §34-41 scalar sets per cycle + final.
  * `metrics.jsonl` -- one JSON line per cycle (machine-readable source for
    compare_runs.py and paper stats).
  * Live console line per cycle with ETA (recent-N wall-clock).
  * System stats (GPU mem/util via torch, CPU/RAM via stdlib -- no psutil).
  * NaN/instability flagging + emergency checkpoint through the trainer hook.

VAL is recorded only at final (split contract: bench_held IS the val-pairs ruler
and is the canonical per-cycle deployment metric; VAL never early-stops).
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from causal_schedule_lab.m3 import config as C


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (float, np.floating)):
        if v != v:
            return None
        if v in (float("inf"), -float("inf")):
            return None
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


@dataclass
class T2BSystemStats:
    device: str = "cpu"
    wall_elapsed: float = 0.0

    def sniff(self) -> dict[str, Any]:
        s = {"wall_elapsed": round(self.wall_elapsed, 1)}
        try:
            import torch
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(self.device) / 1024.0**3
                res = torch.cuda.memory_reserved(self.device) / 1024.0**3
                s["gpu_memory_used_gb"] = round(alloc, 2)
                s["gpu_memory_reserved_gb"] = round(res, 2)
                try:
                    s["gpu_utilization"] = torch.cuda.utilization(self.device) \
                        if hasattr(torch.cuda, "utilization") else None
                except Exception:  # noqa: BLE001
                    s["gpu_utilization"] = None
        except Exception:  # noqa: BLE001
            pass
        try:
            s["cpu_percent"] = None      # no psutil; omitted honestly
            ru = resource.getrusage(resource.RUSAGE_SELF)
            s["ram_used_max_gb"] = round(ru.ru_maxrss / 1024.0 / 1024.0, 2)
        except Exception:  # noqa: BLE001
            pass
        return s


class T2BMonitor:
    """Per-run monitoring bundle (TB + JSONL + console)."""

    def __init__(self, run_dir: Path, run_id: str, config: dict[str, Any],
                 hparams: dict[str, Any] | None = None, device="cpu",
                 tag_warn=None):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.device = device
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg_hash = hashlib.sha256(
            json.dumps(_jsonable(config), sort_keys=True).encode()).hexdigest()[:12]
        self.cfg_hash = cfg_hash
        self.line = 0
        self._t0 = time.time()
        self._recent_walls: list[float] = []
        self._total_cycles = 0
        self.warn_max_kl = float((config or {}).get("warn_max_kl", 10.0))
        self.warn_max_gn = float((config or {}).get("warn_max_grad_norm", 50.0))
        self.warn_min_div = float((config or {}).get("warn_min_diversity", 0.01))
        self.last_eval = None
        self.warnings: list[str] = []

        self._jsonl: Path | None = None
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            # T2-C edition: TB is collected under its own t2c/ tree (the package's
            # outputs/canonical_m3/tensorboard -- the Mac canonical tree is untouched).
            tb_dir = C.CANONICAL_OUT_DIR / "tensorboard" / "t2c" / run_id
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(tb_dir))
            self._tb_dir = tb_dir
            if hparams:
                try:
                    self._writer.add_hparams(
                        {k: _scalar_hp(v) for k, v in (hparams or {}).items()},
                        {"run/started": 0.0})
                except Exception as exc:  # noqa: BLE001
                    self.warnings.append(f"hparams failed: {exc}")
        except Exception as exc:           # noqa: BLE001 - TB optional
            self.warnings.append(f"TensorBoard unavailable: {exc}")
            self._writer = None
            self._tb_dir = None
        self._jsonl = run_dir / "metrics.jsonl"

    # ------------------------------------------------------------------
    def _scalar(self, tag, value):
        if self._writer is not None and value is not None:
            try:
                self._writer.add_scalar(tag, float(value), self.line)
            except (TypeError, ValueError):
                pass

    def hparams_final(self, metrics: dict[str, Any]):
        if self._writer is not None:
            try:
                self._writer.add_hparams(
                    {k: _scalar_hp(v) for k, v in metrics.get("hparams", {}).items()},
                    {"run/final": _scalar_hp(metrics.get("final_greedy", 0.0))})
            except Exception as exc:       # noqa: BLE001
                self.warnings.append(f"hparams_final failed: {exc}")

    def log_cycle(self, cycle, hist_entry, upd_extra=None, group_stats=None,
                  mon_extra=None, system=None, ev=None, elapsed_s=None):
        """One per-cycle line: TB scalars + metrics.jsonl + console + ETA."""
        self.line = cycle
        upd = upd_extra or {}
        gs = group_stats or {}
        mn = mon_extra or {}
        sys = system or {}
        ev = ev or {}
        epoch_last = {}
        if upd.get("epochs"):
            epoch_last = upd["epochs"][-1]

        train_total = float((hist_entry or {}).get("train", 0.0))
        bench_h = float((ev.get("bench_held") or {}).get("total", 0.0))
        real_h = float((ev.get("real_held") or {}).get("total", 0.0))
        syn_h = float((ev.get("syn_held") or {}).get("total", 0.0))
        held_agg = float(np.mean([bench_h, real_h, syn_h])) \
            if ev.get("bench_held") is not None else None

        # --- warnings ----------------------------------------------------
        kl = epoch_last.get("kl_ref_m3", 0.0)
        gn = epoch_last.get("grad_norm", 0.0)
        if kl is not None and kl > self.warn_max_kl:
            self.warnings.append(f"epoch {cycle}: kl_m3={kl:.3f} > warn_max_kl")
        if gn is not None and gn > self.warn_max_gn:
            self.warnings.append(f"epoch {cycle}: grad_norm={gn:.2f} > warn_max_gn")

        # --- TensorBoard scalar sets (§34-41) ----------------------------
        def _tb(prefix, d):
            for k, v in (d or {}).items():
                if v is None:
                    continue
                self._scalar(f"{prefix}/{k}", v)
        _tb("train", {"greedy_gain": train_total})
        if held_agg is not None:
            _tb("held", {"greedy_gain": held_agg})
            _tb("held_benchmark", {"greedy_gain": bench_h})
            _tb("real_held", {"greedy_gain": real_h})
            _tb("syn_held", {"greedy_gain": syn_h})
        _tb("grpo", {"loss": epoch_last.get("loss"),
                     "m3_kl": epoch_last.get("kl_ref_m3"),
                     "m2_kl": epoch_last.get("kl_m2"),
                     "entropy": epoch_last.get("entropy"),
                     "clip_fraction": epoch_last.get("clip_frac_m3"),
                     "ratio_mean": epoch_last.get("ratio_mean"),
                     "ratio_std": epoch_last.get("ratio_std"),
                     "advantage_mean": epoch_last.get("adv_mean"),
                     "advantage_std": epoch_last.get("adv_std"),
                     "grad_norm": epoch_last.get("grad_norm")})
        _tb("group", gs)
        _tb("policy", mn)
        # T2-C residual-scale set (§35-36): norm/ratio/alpha/cap/base-vs-final logit std
        m3tag = {"n_steps": hist_entry.get("n_steps") if hist_entry else None,
                 "n_actions": gs.get("n_actions")}
        _res = (upd.get("_monitor") or {}).get("residual")
        if isinstance(_res, dict):
            m3tag.update({"residual_norm": _res.get("dprop_norm"),
                          "residual_max_abs": _res.get("dprop_max_abs"),
                          "residual_mean_abs": _res.get("dprop_mean_abs"),
                          "residual_alpha_prop": _res.get("alpha_prop"),
                          "residual_cap": _res.get("cap"),
                          "base_logit_std": _res.get("base_std"),
                          "final_logit_std": _res.get("final_std"),
                          "residual_sft_ratio": _res.get("sft_ratio")})
        _tb("m3", m3tag)
        _tb("m2", {"n_s2_trajs": upd.get("n_informative_trajectories_m2")})
        _tb("system", sys)

        # --- metrics.jsonl ------------------------------------------------
        record = {
            "run_id": self.run_id, "config_hash": self.cfg_hash,
            "cycle": cycle, "elapsed_s": float(elapsed_s or 0.0),
            "train_greedy": train_total, "held_greedy_agg": held_agg,
            "bench_held": bench_h, "real_held": real_h, "syn_held": syn_h,
            "grpo": epoch_last, "raw_upd": _jsonable(upd.get("_monitor")),
            "residual": _jsonable((upd.get("_monitor") or {}).get("residual")),
            "group": _jsonable(gs), "policy": _jsonable(mn),
            "system": _jsonable(sys), "ev": _jsonable(ev),
        }
        if self._jsonl is not None:
            with open(self._jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(record), default=str) + "\n")

        # --- console + ETA ------------------------------------------------
        self._recent_walls.append(float(elapsed_s or 0.0))
        self._recent_walls = self._recent_walls[-10:]
        self._print_console(cycle, train_total, held_agg, epoch_last, gs, sys)

    def _print_console(self, cycle, train_total, held_agg, ep, gs, sys):
        pos_rate = gs.get("positive_count_mean")
        pos_rate = float(pos_rate) if isinstance(pos_rate, (int, float)) else None
        div = gs.get("mean_distinct_traj")
        bits = [
            f"[t2c] cycle {cycle}: TRAIN={train_total:.0f}",
            f"held={held_agg:.1f}" if held_agg is not None else None,
            f"pos_rate={pos_rate if pos_rate is not None else 0:.3f}",
            f"kl_m3={ep.get('kl_ref_m3', 0.0):.4f}",
            f"ent={ep.get('entropy', 0.0):.2f}",
            f"clip={ep.get('clip_frac_m3', 0.0):.3f}",
            f"gn={ep.get('grad_norm', 0.0):.2f}",
            f"traj/min={gs.get('traj_per_min', 0.0):.1f}",
            f"mem={sys.get('gpu_memory_used_gb', 'n/a')}GB",
            self._eta(),
        ]
        print(" ".join(str(b) for b in bits if b is not None), flush=True)

    def _eta(self) -> str:
        if not self._recent_walls:
            return ""
        avg = float(np.mean(self._recent_walls))
        remain_cycles = self._total_cycles - self.line
        eta_s = avg * remain_cycles
        h, rem = divmod(int(eta_s), 3600)
        m, s = divmod(rem, 60)
        return f"ETA={h}h{m:02d}m{s:02d}s" if self._total_cycles else ""

    def log_final(self, metrics: dict[str, Any]):
        """Final cycle summary: VAL final-only vs no early stop (§35), all
        final scalars + JSONL + hparams."""
        self.hparams_final(metrics)
        for k, v in (metrics.get("tb", {}) or {}).items():
            self._scalar(f"final/{k}", _flatten(v))
        if self._jsonl is not None:
            with open(self._jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps({"run_id": self.run_id,
                                    "config_hash": self.cfg_hash,
                                    "event": "final",
                                    "metrics": _jsonable(_flatten(metrics))},
                                   default=str) + "\n")
        if self.warnings:
            print(f"[t2c] warnings ({len(self.warnings)}):",
                  flush=True)
            for w in self.warnings[-5:]:
                print(f"  - {w}", flush=True)

    def close(self):
        if self._writer is not None:
            try:
                self._writer.flush()
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass


def _scalar_hp(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (dict, list)):
        return json.dumps(_jsonable(v), default=str)
    return v


def _flatten(d, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(d, dict):
        for k, v in d.items():
            _flatten(v, f"{prefix}{k}/" if prefix else f"{k}/", out)
    else:
        out[prefix.rstrip("/")] = _jsonable(d)
    return out