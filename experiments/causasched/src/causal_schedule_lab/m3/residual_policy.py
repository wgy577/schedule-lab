"""T2-C -- M3 LARGE / CONTROLLED RESIDUAL POLICY (C1 / C2) + cap calibration.

Independent edition (package only; canonical Mac tree untouched).  The frozen-SFT
r6 backbone stays frozen; the ONLY trainable part of M3 is the residual head on top
of the frozen R6 base logits.  Three modes, single source of truth per mode:

    * tiny / C0  -> the CANONICAL M3TemporalEvidencePolicy (zero-init Linear(283,1)
                    + Linear(12,1), delta = alpha * tanh(raw)) = resid_prop 284
                    (3*1+1), resid_stop 13 (12*1+1), M3 = 297, + M2 adapter 161
                    = 458 trainable params TOTAL.  The packed GPU update runs its
                    canonical else-branch, so C0 is bit-identical to T2-B-GPU.
                    The T2-C runner *uses the canonical class directly* for tiny
                    -- nothing new to construct.  (Counts are NOT hand-written:
                    `expected_residual_params(...)` / `param_counts()` derive
                    them programmatically.)
    * large / C1  -> M3ResidualPolicy(mode="large"): the proposal residual head is
                    the PRE-REGISTERED deep MLP
                        LayerNorm(D_resid) -> Linear(D_resid,256) -> GELU
                        -> Linear(256,256) -> GELU -> Linear(256,256) -> GELU
                        -> Linear(256,1)
                    with IDENTITY output activation, so the residual logit shift
                    is NOT capped at +-alpha.  D_resid is read dynamically from
                    D_PROP_IN (= D_FPOOL + D_EVID), never hard-coded.  The final
                    Linear(256,1) is zero-init (E0 anchor).  STOP residual stays
                    the small bounded Linear(12,1)+alpha*tanh in ALL modes
                    (R7 STOP-collapse guard).  alpha_prop warms 0 -> alpha_target
                    monotone-linearly (default `--residual-warmup-frac 0.25`).
                    Param counts come from `expected_residual_params(...)`, never
                    hand-arith: for the default 256/3 build the tally is ~205k
                    M3 (recall the old single-hidden-layer "72,974" number was
                    prop+STOP combined, not the prop net alone -- the corrected
                    tally splits resid_prop vs resid_stop).
    * controlled / C2 -> same pre-registered MLP but output activation =
                    cap*tanh(raw), so the residual shift is bounded in
                    [-alpha*cap, +alpha*cap].  cap defaults to a robust scale of
                    the FROZEN R6 base logits measured on TRAIN-ONLY root states
                    (`residual_scale_calibration.json`); `--residual-cap <float>`
                    overrides (user-supplied => "user" provenance).  Never
                    calibrated on held/val.

All modes keep the canonical zero-init-final rule: delta == 0 at first forward
(pure frozen-SFT base) => the E0/T2-A bit-parity anchor holds at cycle 0.  No new
oracle features: evid stays the R19 6-d memory observations, cap uses ONLY the
frozen base logit scale (no future-reward leakage), reward stays
R_i = Cmax(S_root)-Cmax(S_terminal_i), identified=false, Formal TEST sealed.

Interface mirrors M3TemporalEvidencePolicy exactly (_base_raw, _stop_in,
_stop_base, residual_prop, residual_stop, forward, base_logits, action_logits,
prop_scores, stop_head, snapshot, load_snapshot) so the joint harnesses consume it
unchanged.  `residual_delta_prop(F_ev)` is the ONE extra method the PACKAGE copy of
the packed GPU update dispatches on (hasattr guard; canonical line stays intact).
"""

from __future__ import annotations

import hashlib
import json

import torch
import torch.nn as nn

from causal_schedule_lab.m3 import config as C

D_FPOOL = 277                                  # canonical proposal feature vector
D_EVID = int(C.TO1_R19_EVID_DIM)               # R19 evidence dim (6)
D_STOP = 7 + int(C.TO1_STOP_POOL_STAT_DIM)     # state(7) + pool_stats(5)
D_PROP_IN = D_FPOOL + D_EVID                   # 283


# ---------------------------------------------------------------------------
# cap calibration numerics (pure torch; the STATE sweep lives in the runner)
# ---------------------------------------------------------------------------
def robust_logit_scale(zs: torch.Tensor) -> float:
    """Robust (MAD-based) std of frozen base logits -> default residual cap.

    TRAIN-only scores only (the caller guarantees the source).  Falls back to the
    plain std then 1.0 if the tensor is empty or degenerate -- a cap of <=0 would
    zero the C2 residual entirely."""
    z = zs.detach().float().flatten()
    if z.numel() == 0:
        return 1.0
    med = z.median()
    mad = (z - med).abs().median()
    scale = float(1.4826 * mad)
    if not torch.isfinite(torch.tensor(scale)) or scale <= 1e-6:
        scale = float(z.std().item())
    if not torch.isfinite(torch.tensor(scale)) or scale <= 1e-6:
        scale = 1.0
    return scale


def base_logit_scale_stats(zs: torch.Tensor) -> dict:
    """Summary dict persisted to residual_scale_calibration.json (§cap)."""
    z = zs.detach().float().flatten()
    n = int(z.numel())
    stats = {
        "n_scores": n,
        "mean": float(z.mean().item()) if n else 0.0,
        "mean_abs": float(z.abs().mean().item()) if n else 0.0,
        "std": float(z.std().item()) if n else 0.0,
        "mad_scale": robust_logit_scale(z),
        "p95_abs": float(torch.quantile(z.abs(), 0.95).item()) if n else 0.0,
        "max_abs": float(z.abs().max().item()) if n else 0.0,
    }
    return stats


# ---------------------------------------------------------------------------
# alpha schedule (monotone linear warmup, the directive default)
# ---------------------------------------------------------------------------
def residual_alpha_prop_value(alpha_target: float, warmup_fraction: float,
                              cycle: int, total_cycles: int) -> float:
    """alpha_prop(cycle): 0 at cycle 0 -> alpha_target at warmup_fraction*cycles.

    `warmup_fraction <= 0` means no warmup (alpha == target from the very first
    update).  step is (cycle+1) so cycle 0 (the first update) sees a *tiny* alpha,
    not exactly 0, unless warmup_fraction forces it."""
    total = int(max(1, int(total_cycles)))
    frac = float(warmup_fraction)
    if frac <= 0.0:
        return float(alpha_target)
    warmup = int(max(1, round(frac * total)))
    step = min(total, int(cycle) + 1)
    return float(alpha_target) * min(1.0, step / warmup)


# ---------------------------------------------------------------------------
# programmatic param tallies (the ONLY source for docs/tests -- no hand-arith)
# ---------------------------------------------------------------------------
def expected_residual_param_counts(d_resid=D_PROP_IN, hidden=256, depth=3,
                                   d_stop=D_STOP, m2_feat=8, m2_hidden=16):
    """Exact per-layer param tally for the pre-registered C1/C2 residual head.

    LayerNorm(x) = 2x (affine gamma+beta); Linear(i,o) = i*o + o.  M2 adapter =
    M2RootPolicyAdapter(feat->hidden->1) = (m2_feat*m2_hidden + m2_hidden) +
    (m2_hidden + 1).  Values are DERIVED, not constants."""
    prop = 2 * int(d_resid)                       # LayerNorm(D_resid)
    cur = int(d_resid)
    for _ in range(int(depth)):                   # hidden layers (D->h, h->h xN)
        prop += cur * int(hidden) + int(hidden)
        cur = int(hidden)
    prop += cur * 1 + 1                           # output Linear(hidden,1)
    stop = int(d_stop) * 1 + 1                    # Linear(12,1)
    m3 = int(prop) + int(stop)
    m2 = (int(m2_feat) * int(m2_hidden) + int(m2_hidden)) + (int(m2_hidden) + 1)
    return {"resid_prop": int(prop), "resid_stop": int(stop),
            "m3_residual": int(m3), "m2_adapter": int(m2),
            "total": int(m3) + int(m2), "depth": int(depth),
            "hidden": int(hidden)}


# ---------------------------------------------------------------------------
class M3ResidualPolicy(nn.Module):
    """C1 (large) / C2 (controlled) residual policy on the frozen R6 selector.

    tiny/C0 == canonical M3TemporalEvidencePolicy (this class raises on
    mode="tiny" so a mode typo cannot silently construct the wrong head)."""

    def __init__(self, r6_selector, mode="large", hidden=256, depth=3,
                 cap=None, alpha_target=None, alpha_stop=None,
                 warmup_fraction=0.25):
        super().__init__()
        mode = str(mode)
        if mode not in ("large", "controlled"):
            raise ValueError(
                f"M3ResidualPolicy mode must be 'large'|'controlled', got {mode!r}; "
                f"tiny/C0 == canonical M3TemporalEvidencePolicy (the runner uses it "
                f"directly so C0 is bit-identical to T2-B).")
        if int(depth) < 1 or int(hidden) < 1:
            raise ValueError(f"residual MLP needs hidden>=1, depth>=1 "
                             f"(got hidden={hidden}, depth={depth})")
        self.mode = mode
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.r6 = r6_selector
        self.r6.eval()
        for p in self.r6.parameters():
            p.requires_grad_(False)

        # alpha*: proposal target (warmed), STOP stays constant + bounded (R7 guard)
        self.alpha_target = float(alpha_target if alpha_target is not None
                                  else C.TO1_R19_ALPHA_TEMPORAL)
        self.alpha_stop = float(alpha_stop if alpha_stop is not None
                                else C.TO1_R19_ALPHA_TEMPORAL)
        self.warmup_fraction = float(warmup_fraction)
        self.alpha_prop = float(self.alpha_target)   # pre-warmup value; set_alpha will ramp
        self.total_cycles = 0
        self.cap = None if cap is None else float(cap)

        self.resid_prop = self._build_prop_mlp(D_PROP_IN, self.hidden, self.depth)
        self.resid_stop = nn.Linear(D_STOP, 1)
        self._zero_residual()

    # -- construction ------------------------------------------------------
    def _build_prop_mlp(self, d_in, hidden, depth):
        """Pre-registered C1/C2 residual MLP:
        LayerNorm(D_resid) -> Linear(D_resid,hidden) -> GELU
            -> [Linear(hidden,hidden) -> GELU] x (depth-1)
            -> Linear(hidden,1)                 # final layer ZERO-INIT
        `d_in` = D_PROP_IN (read dynamically from D_FPOOL + D_EVID, never 283)."""
        layers = [nn.LayerNorm(int(d_in))]                 # affine gamma+beta
        cur = int(d_in)
        for _ in range(int(depth)):                        # depth hidden layers
            layers.append(nn.Linear(cur, hidden))
            layers.append(nn.GELU())
            cur = int(hidden)
        layers.append(nn.Linear(cur, 1))                   # final layer (zero-init)
        return nn.Sequential(*layers)

    def _zero_residual(self):
        with torch.no_grad():
            last = self.resid_prop[-1]
            last.weight.zero_()
            last.bias.zero_()
            self.resid_stop.weight.zero_()
            self.resid_stop.bias.zero_()

    # -- schedule / cap ----------------------------------------------------
    def set_total_cycles(self, total_cycles: int) -> None:
        self.total_cycles = int(total_cycles)

    def set_alpha(self, cycle: int) -> None:
        """Monotone warmup default: alpha_prop(cycle) = target*min(1,(cycle+1)/wf)."""
        self.alpha_prop = residual_alpha_prop_value(
            self.alpha_target, self.warmup_fraction, int(cycle), self.total_cycles)

    def set_cap(self, cap: float) -> None:
        self.cap = float(cap)

    def param_counts(self) -> dict:
        """Actual trainable-param tally (dead-simple: sum over state_dict)."""
        tr = {n: p.numel() for n, p in self.named_parameters()
              if p.requires_grad}
        prop = sum(v for n, v in tr.items() if n.startswith("resid_prop."))
        stop = sum(v for n, v in tr.items() if n.startswith("resid_stop."))
        return {"resid_prop": int(prop), "resid_stop": int(stop),
                "m3_residual": int(prop + stop),
                "counted_by": "actual-instantiated-model"}

    # -- frozen base (VERBATIM canonical) ----------------------------------
    def _base_raw(self, F_pool):
        if F_pool is not None and len(F_pool):
            with torch.no_grad():
                raw = self.r6.prop_scores(F_pool)   # frozen, no grad
            return raw, raw, {"scale": 1.0, "selected": "raw"}
        empty = {"center": 0.0, "scale": 0.0, "mad": 0.0, "std": 0.0,
                 "selected": "empty"}
        return torch.zeros(0, dtype=torch.float32), torch.zeros(0,
                                                                dtype=torch.float32), empty

    def _stop_in(self, state_feat, pool_stats):
        return torch.cat([state_feat.detach().reshape(1, -1),
                          pool_stats.detach().reshape(1, C.TO1_STOP_POOL_STAT_DIM)],
                         dim=-1)

    def _stop_base(self, state_feat, pool_stats):
        stop_in = self._stop_in(state_feat, pool_stats)
        with torch.no_grad():
            return self.r6.stop_head(stop_in).reshape(-1)[0]   # float scalar tensor

    # -- mode-specific residual activation --------------------------------
    def _act_prop(self, x):
        if self.mode == "controlled":
            cap = float(self.cap) if (self.cap is not None and self.cap > 0) else 1.0
            return cap * torch.tanh(x)
        return x                                             # large: identity (unbounded)

    def residual_delta_prop(self, F_ev):
        """PACKED GPU path: F_ev is pre-concatenated [F_pool | evid] in [., 283]."""
        return self.alpha_prop * self._act_prop(
            self.resid_prop(F_ev).squeeze(-1))

    # -- canonical interface (CPU path) ------------------------------------
    def residual_prop(self, F_pool, evid=None):
        if F_pool is None or len(F_pool) == 0:
            return torch.zeros(0, dtype=torch.float32)
        F = F_pool.detach().float()
        if evid is not None and len(evid) == len(F_pool):
            F = torch.cat([F, evid.detach().float()], dim=-1)
        else:
            F = torch.cat([F, torch.zeros(len(F_pool), D_EVID,
                                          dtype=torch.float32)], dim=-1)
        return self.residual_delta_prop(F)

    def residual_stop(self, state_feat, pool_stats):
        stop_in = self._stop_in(state_feat, pool_stats)
        return float(self.alpha_stop) * torch.tanh(
            self.resid_stop(stop_in).squeeze(-1)).reshape(1)

    def forward(self, F_pool, state_feat, pool_stats=None, evid=None):
        if pool_stats is None:
            pool_stats = _pool_stats_from(F_pool)
        z, _, _ = self._base_raw(F_pool)
        base_stop = self._stop_base(state_feat, pool_stats)
        return (z + self.residual_prop(F_pool, evid),                     # [M]
                base_stop + self.residual_stop(state_feat, pool_stats))   # [1]

    def base_logits(self, F_pool, state_feat, pool_stats=None, evid=None):
        if pool_stats is None:
            pool_stats = _pool_stats_from(F_pool)
        z, _, _ = self._base_raw(F_pool)
        return torch.cat([z, self._stop_base(state_feat, pool_stats).reshape(1)],
                         dim=-1)

    def action_logits(self, F_pool, state_feat, pool_stats=None, evid=None):
        sp, ss = self.forward(F_pool, state_feat, pool_stats, evid=evid)
        return torch.cat([sp, ss], dim=-1)         # [M+1]

    def prop_scores(self, F, evid=None):
        z, _, _ = self._base_raw(F)
        return z + self.residual_prop(F, evid)

    def stop_head(self, stop_in):
        sf = stop_in[:, :7]
        ps = stop_in[:, 7:12]
        return self._stop_base(sf.reshape(1, -1), ps.reshape(1, -1)).reshape(1, 1)

    def snapshot(self):
        with torch.no_grad():
            return {"params": {name: p.detach().clone()
                               for name, p in self.named_parameters()}}

    def load_snapshot(self, snap):
        params = snap.get("params", snap)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name in params:
                    p.copy_(params[name])

    # -- architecture metadata (checkpoint identity, resume fail-fast) -----
    def arch(self, r6_parent: str | None = None, m3_parent: str | None = None) -> dict:
        return {
            "mode": self.mode,
            "hidden": int(self.hidden),
            "depth": int(self.depth),
            "cap": self.cap,
            "alpha_target": float(self.alpha_target),
            "alpha_stop": float(self.alpha_stop),
            "warmup_fraction": float(self.warmup_fraction),
            "r6_parent": r6_parent,
            "m3_parent": m3_parent,
        }

    def arch_hash(self, **kw) -> str:
        raw = json.dumps({k: str(v) for k, v in sorted(self.arch(**kw).items())},
                         sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


# late import to avoid cycles (top1 defines the pool-stats helper)
def _pool_stats_from(F_pool: torch.Tensor):
    from causal_schedule_lab.m3.top1 import _pool_stats_from as _psf
    return _psf(F_pool)