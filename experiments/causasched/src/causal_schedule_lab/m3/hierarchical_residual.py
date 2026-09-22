"""T2-D SFT-informed hierarchical set-conditioned residual actors.

This module contains only Stage-3 policy architecture.  It does not own candidate
generation, R20 tier selection, Memory, rewards, or the executor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn


ARCH_P0 = "P0"
ARCH_P1 = "P1"
ARCH_P2 = "P2"
ARCHITECTURES = (ARCH_P0, ARCH_P1, ARCH_P2)


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def frozen_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if not p.requires_grad)


def tensor_state_sha256(module: nn.Module) -> str:
    """Stable digest used by the frozen-parent backward test."""
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        x = value.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(x.shape)).encode("ascii"))
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def freeze_sft(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _require_2d(name: str, x: torch.Tensor) -> None:
    if x.dim() != 2:
        raise ValueError(f"{name} must be rank-2, got {tuple(x.shape)}")


def set_mean_max(x: torch.Tensor, mask: torch.Tensor | None = None):
    """Permutation-invariant mean/max for one set; masked rows never contribute."""
    _require_2d("x", x)
    if mask is None:
        mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    mask = mask.to(device=x.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != x.shape[0]:
        raise ValueError("mask length does not match candidate count")
    if not bool(mask.any()):
        z = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        return z, z.clone()
    valid = x[mask]
    return valid.mean(dim=0), valid.max(dim=0).values


def packed_set_mean_max(x: torch.Tensor, offsets: Sequence[int]):
    """Segment-local pooling.  Explicit offsets make cross-state mixing impossible."""
    _require_2d("x", x)
    offs = [int(v) for v in offsets]
    if len(offs) < 2 or offs[0] != 0 or offs[-1] != len(x):
        raise ValueError("offsets must start at 0 and end at len(x)")
    if any(b < a for a, b in zip(offs, offs[1:])):
        raise ValueError("offsets must be non-decreasing")
    means, maxima = [], []
    for a, b in zip(offs, offs[1:]):
        means.append(set_mean_max(x[a:b])[0])
        maxima.append(set_mean_max(x[a:b])[1])
    return torch.stack(means), torch.stack(maxima)


def padded_set_mean_max(x: torch.Tensor, mask: torch.Tensor):
    """Reference padded GPU pooling for `[B,N,D]` plus boolean `[B,N]` mask."""
    if x.dim() != 3 or mask.shape != x.shape[:2]:
        raise ValueError("expected x[B,N,D] and mask[B,N]")
    rows = [set_mean_max(x[b], mask[b]) for b in range(x.shape[0])]
    return torch.stack([r[0] for r in rows]), torch.stack([r[1] for r in rows])


@dataclass(frozen=True)
class ResidualCaps:
    cap2: float
    cap3: float
    cap_stop: float
    source_split: str = "train"
    statistic: str = "max(1e-3, 1.4826*MAD, robust_top_margin_median)"
    formal_test_access: int = 0
    held_used: bool = False
    val_used: bool = False

    def validate(self) -> None:
        if self.source_split != "train" or self.held_used or self.val_used:
            raise ValueError("residual caps must be calibrated from TRAIN only")
        if self.formal_test_access != 0:
            raise ValueError("Formal TEST access is forbidden")
        if min(self.cap2, self.cap3, self.cap_stop) <= 0:
            raise ValueError("all residual caps must be positive")

    def save(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _robust_scale(rows: Iterable[torch.Tensor]) -> float:
    vals = [torch.as_tensor(x, dtype=torch.float64).reshape(-1) for x in rows]
    vals = [x[torch.isfinite(x)] for x in vals if x.numel()]
    if not vals or not any(x.numel() for x in vals):
        return 1.0
    x = torch.cat(vals)
    med = x.median()
    mad_scale = 1.4826 * (x - med).abs().median()
    margins = []
    for row in vals:
        if row.numel() >= 2:
            top = torch.topk(row, 2).values
            margins.append((top[0] - top[1]).abs())
    margin = torch.stack(margins).median() if margins else x.new_tensor(0.0)
    return float(torch.maximum(torch.maximum(mad_scale, margin), x.new_tensor(1e-3)))


def calibrate_residual_caps(*, split: str, root_logits, proposal_logits, stop_logits):
    """TRAIN-only API by construction; callers cannot pass held/VAL flags silently."""
    if str(split).lower() != "train":
        raise ValueError("held/VAL calibration is forbidden; split must be 'train'")
    caps = ResidualCaps(
        cap2=_robust_scale(root_logits),
        cap3=_robust_scale(proposal_logits),
        cap_stop=_robust_scale(stop_logits),
    )
    caps.validate()
    return caps


def trajectory_context(*, step_index: int, horizon: int, root_makespan: float,
                       current_makespan: float, last_step_gain: float,
                       action_count: int, non_improving_streak: int) -> torch.Tensor:
    """History-only P2 context.  No terminal/future/best-of-N field is accepted."""
    h = max(int(horizon), 1)
    root = max(float(root_makespan), 1.0)
    return torch.tensor([
        min(max(float(step_index) / h, 0.0), 1.0),
        (root - float(current_makespan)) / root,
        float(last_step_gain) / root,
        min(max(float(action_count) / h, 0.0), 1.0),
        min(max(float(non_improving_streak) / h, 0.0), 1.0),
    ], dtype=torch.float32)


class FrozenR6LatentView(nn.Module):
    """Audited access to R6 hidden tensors without hooks or train-mode dropout."""

    def __init__(self, selector: nn.Module):
        super().__init__()
        self.selector = freeze_sft(selector)
        prop = selector.prop_head
        stop = selector.stop_head
        if len(prop) != 7 or not isinstance(prop[0], nn.Linear) or not isinstance(prop[3], nn.Linear):
            raise ValueError("unexpected R6 proposal-head architecture")
        if len(stop) != 3 or not isinstance(stop[0], nn.Linear):
            raise ValueError("unexpected R6 STOP-head architecture")
        self.prop_input_dim = int(prop[0].in_features)
        self.prop_latent_dim = int(prop[3].out_features)
        self.stop_input_dim = int(stop[0].in_features)
        self.stop_latent_dim = int(stop[0].out_features)

    def train(self, mode: bool = True):
        super().train(mode)
        self.selector.eval()
        return self

    def proposal(self, F_pool: torch.Tensor):
        if getattr(self, "end_to_end", False):
            h = self.selector.prop_head[:5](F_pool)
            return h, self.selector.prop_head[5:](h).squeeze(-1)
        with torch.no_grad():
            h = F_pool
            for layer in self.selector.prop_head[:5]:
                h = layer(h)
            base = self.selector.prop_head[5](h)
            base = self.selector.prop_head[6](base).squeeze(-1)
        return h.detach(), base.detach()

    def stop(self, stop_in: torch.Tensor):
        if getattr(self, "end_to_end", False):
            h = self.selector.stop_head[:2](stop_in)
            return h, self.selector.stop_head[2](h).reshape(-1)
        with torch.no_grad():
            h = self.selector.stop_head[1](self.selector.stop_head[0](stop_in))
            base = self.selector.stop_head[2](h).reshape(-1)
        return h.detach(), base.detach()


class M2RootSetResidualActor(nn.Module):
    """SFT-informed root Attention actor; still only allocates probe budget."""

    def __init__(self, root_latent_dim: int, local_dim: int, state_dim: int,
                 memory_dim: int = 3, cap: float = 1.0,
                 target_alpha: float = 0.2, warmup_fraction: float = 0.1,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 1,
                 dim_feedforward: int = 256, relation_dim: int = 0,
                 root_value_scale: float = 3.0):
        super().__init__()
        self.root_latent_dim = int(root_latent_dim)
        self.local_dim = int(local_dim)
        self.state_dim = int(state_dim)
        self.memory_dim = int(memory_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.relation_dim = int(relation_dim)
        self.root_value_scale = float(root_value_scale)
        self.probability_trace = None
        self.cap = float(cap)
        self.target_alpha = float(target_alpha)
        self.warmup_fraction = float(warmup_fraction)
        self.alpha_fraction = 0.0
        token_in = self.root_latent_dim + 1 + self.local_dim + self.memory_dim
        self.root_token = nn.Sequential(
            nn.Linear(token_in, self.d_model), nn.LayerNorm(self.d_model), nn.GELU())
        self.state_token = nn.Sequential(
            nn.Linear(self.state_dim, self.d_model), nn.LayerNorm(self.d_model),
            nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead,
            dim_feedforward=self.dim_feedforward, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        # T2-I: bounded root×appearance attention.  The final context projection
        # is zero-initialized so old checkpoints reproduce their original logits
        # before GRPO learns to use the new branch.
        self.relation_encoder = (nn.Sequential(
            nn.Linear(self.relation_dim, self.d_model), nn.LayerNorm(self.d_model),
            nn.GELU()) if self.relation_dim > 0 else None)
        self.relation_attention = (nn.MultiheadAttention(
            self.d_model, self.nhead, dropout=0.0, batch_first=True)
            if self.relation_dim > 0 else None)
        self.appearance_context = (nn.Linear(self.d_model, self.d_model, bias=False)
                                   if self.relation_dim > 0 else None)
        if self.appearance_context is not None:
            nn.init.zeros_(self.appearance_context.weight)
        # B5 is causal evidence, not intervention utility.  Start almost exactly
        # at trust=1, while allowing RL to reduce root/state-specific trust.
        self.prior_distrust = nn.Linear(self.d_model, 1)
        nn.init.zeros_(self.prior_distrust.weight)
        nn.init.constant_(self.prior_distrust.bias, -12.0)
        # A separate zero-init value path is not confined to the historical
        # ±alpha*cap residual envelope.  KL-to-B5 remains the stability control.
        self.root_value = nn.Sequential(
            nn.Linear(self.d_model, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.root_value[-1].weight)
        nn.init.zeros_(self.root_value[-1].bias)

    def enable_probability_trace(self):
        from .probability_trace import ProbabilityTrace
        if self.probability_trace is None:
            # Do not perturb baseline policy or rollout RNG when adding a branch.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1729)
                self.probability_trace = ProbabilityTrace(
                    self.root_latent_dim, self.state_dim).to(next(self.parameters()).device)
            # Preserve the calibrated frozen B5 prior at startup. Probability
            # propagation is a learnable correction, not a randomly initialized
            # replacement for the SFT policy.

    def trace_bias(self, graph, state_context):
        if self.probability_trace is None:
            return None
        if graph is None:
            raise ValueError("probability trace enabled but decision graph is missing")
        return max(float(self.alpha_fraction), 0.1) * self.probability_trace(
            graph, state_context)

    @property
    def alpha_m2(self):
        return self.target_alpha * self.alpha_fraction

    def set_progress(self, cycle: int, total_cycles: int) -> None:
        warm = max(int(math.ceil(max(total_cycles, 1) * self.warmup_fraction)), 1)
        self.alpha_fraction = max(
            self.alpha_fraction, min(max(float(cycle) / warm, 0.0), 1.0))

    def _apply_appearance_single(self, root, app_raw=None, app_mask=None):
        if self.relation_encoder is None or app_raw is None or app_mask is None:
            return root
        app_raw = app_raw.to(device=root.device, dtype=root.dtype)
        valid = app_mask.to(device=root.device, dtype=torch.bool)
        if app_raw.shape[:2] != valid.shape or app_raw.shape[0] != len(root):
            raise ValueError("root appearance tensor shape mismatch")
        safe = valid.clone()
        empty = ~safe.any(dim=1)
        if bool(empty.any()):
            safe[empty, 0] = True
            app_raw = app_raw.clone()
            app_raw[empty, 0] = 0.0
        rel = self.relation_encoder(app_raw)
        ctx, _ = self.relation_attention(
            root.unsqueeze(1), rel, rel, key_padding_mask=~safe,
            need_weights=False)
        return root + self.appearance_context(ctx[:, 0])

    def _apply_appearance_batched(self, root, app_raw=None, app_mask=None):
        if self.relation_encoder is None or app_raw is None or app_mask is None:
            return root
        B, N, D = root.shape
        if app_raw.shape[:2] != (B, N) or app_mask.shape != app_raw.shape[:3]:
            raise ValueError("batched root appearance tensor shape mismatch")
        K = app_raw.shape[2]
        raw = app_raw.to(device=root.device, dtype=root.dtype).reshape(B*N, K, -1)
        valid = app_mask.to(device=root.device, dtype=torch.bool).reshape(B*N, K)
        safe = valid.clone()
        empty = ~safe.any(dim=1)
        if bool(empty.any()):
            safe[empty, 0] = True
            raw = raw.clone(); raw[empty, 0] = 0.0
        rel = self.relation_encoder(raw)
        query = root.reshape(B*N, 1, D)
        ctx, _ = self.relation_attention(
            query, rel, rel, key_padding_mask=~safe, need_weights=False)
        return root + self.appearance_context(ctx[:, 0]).reshape(B, N, D)

    def _encoded_single(self, root_latent, local, base, state_context=None,
                        memory_evidence=None, mask=None, app_raw=None,
                        app_mask=None):
        root_latent = root_latent.float()
        local = local.float()
        base = base.float().reshape(-1, 1)
        if state_context is None:
            state_context = root_latent.new_zeros(self.state_dim)
        state_context = state_context.detach().float().reshape(1, -1)
        if memory_evidence is None:
            memory_evidence = root_latent.new_zeros((len(root_latent), self.memory_dim))
        memory_evidence = memory_evidence.detach().float()
        root = self.root_token(torch.cat(
            [root_latent, base, local, memory_evidence], dim=-1))
        root = self._apply_appearance_single(root, app_raw, app_mask)
        state = self.state_token(state_context)
        seq = torch.cat([state, root], dim=0).unsqueeze(0)
        key_padding_mask = None
        if mask is not None:
            valid = mask.to(device=root.device, dtype=torch.bool).reshape(-1)
            if valid.numel() != len(root):
                raise ValueError("mask length does not match root count")
            key_padding_mask = torch.cat([
                torch.zeros(1, dtype=torch.bool, device=root.device), ~valid
            ]).unsqueeze(0)
        encoded = self.encoder(seq, src_key_padding_mask=key_padding_mask)[0]
        return encoded[1:]

    def raw_delta(self, root_latent, local, base, state_context=None,
                  memory_evidence=None, mask=None, app_raw=None, app_mask=None):
        encoded = self._encoded_single(
            root_latent, local, base, state_context, memory_evidence, mask,
            app_raw, app_mask)
        return self.head(encoded).squeeze(-1)

    def final_logits_batched(self, root_latent, local, base, state_context,
                             memory_evidence, mask, app_raw=None, app_mask=None,
                             trace_queries=None, trace_bias_override=None,
                             end_to_end=False):
        """Padded `[B,N,*]` equivalent of ``final_logits_from_candidates``.

        Padding is excluded from self-attention.  The actor has no positional
        encoding, so each valid row is numerically the same set computation as
        the legacy batch=1 path (up to ordinary GEMM round-off).
        """
        if root_latent.dim() != 3 or mask.shape != root_latent.shape[:2]:
            raise ValueError("expected root_latent[B,N,D] and mask[B,N]")
        root_latent = (root_latent if end_to_end else root_latent.detach()).float()
        local = (local if end_to_end else local.detach()).float()
        base = (base if end_to_end else base.detach()).float()
        state_context = state_context.detach().float()
        memory_evidence = memory_evidence.detach().float()
        root = self.root_token(torch.cat(
            [root_latent, base.unsqueeze(-1), local, memory_evidence], dim=-1))
        root = self._apply_appearance_batched(root, app_raw, app_mask)
        state = self.state_token(state_context).unsqueeze(1)
        seq = torch.cat([state, root], dim=1)
        valid = mask.to(device=seq.device, dtype=torch.bool)
        key_padding = torch.cat([
            torch.zeros((len(seq), 1), dtype=torch.bool, device=seq.device), ~valid
        ], dim=1)
        encoded = self.encoder(seq, src_key_padding_mask=key_padding)
        roots = encoded[:, 1:]
        raw = self.head(roots).squeeze(-1)
        trust = 1.0 - torch.sigmoid(self.prior_distrust(roots).squeeze(-1))
        value = self.root_value_scale * torch.tanh(
            self.root_value(roots).squeeze(-1))
        # One scalar per node: expected net intervention ability.  Causal
        # propagation, frozen-B5 evidence, root/state context and Memory are
        # components of this single logit, never separately selected targets.
        logits = (trust * base + value +
                  float(self.alpha_m2 * self.cap) * torch.tanh(raw))
        if trace_bias_override is not None:
            logits = logits + trace_bias_override
        elif self.probability_trace is not None:
            if trace_queries is None:
                raise ValueError("batched trace queries missing")
            logits = logits + max(float(self.alpha_fraction), 0.1) * self.probability_trace.batch_bias(
                trace_queries, state_context, logits.shape[1])
        return logits.masked_fill(~valid, -torch.inf)

    def residual(self, root_latent, local, base, state_context=None,
                 memory_evidence=None, mask=None):
        return float(self.alpha_m2 * self.cap) * torch.tanh(
            self.raw_delta(root_latent, local, base, state_context,
                           memory_evidence, mask))

    def score(self, base, local, root_latent, state_context=None,
              memory_evidence=None, mask=None):
        out = base + self.residual(root_latent, local, base, state_context,
                                   memory_evidence, mask)
        if mask is not None:
            out = out.masked_fill(~mask.to(device=out.device, dtype=torch.bool), -torch.inf)
        return out

    def delta_from_candidates(self, cand: dict, ids=None):
        dev = cand["base"].device
        idx = (torch.arange(cand["n_cand"], dtype=torch.long, device=dev)
               if ids is None else torch.as_tensor(ids, dtype=torch.long, device=dev))
        memory = cand.get("memory_evidence")
        return self.raw_delta(cand["latents"][idx], cand["feats"][idx],
                              cand["base"][idx], cand.get("state_context"),
                              None if memory is None else memory[idx],
                              app_raw=(cand.get("app_raw")[idx]
                                       if cand.get("app_raw") is not None else None),
                              app_mask=(cand.get("app_mask")[idx]
                                        if cand.get("app_mask") is not None else None))

    def final_logits_from_candidates(self, cand: dict, ids=None):
        dev = cand["base"].device
        idx = (torch.arange(cand["n_cand"], dtype=torch.long, device=dev)
               if ids is None else torch.as_tensor(ids, dtype=torch.long, device=dev))
        base = cand["base"][idx]
        memory = cand.get("memory_evidence")
        roots = self._encoded_single(
            cand["latents"][idx], cand["feats"][idx], base,
            cand.get("state_context"), None if memory is None else memory[idx],
            app_raw=(cand.get("app_raw")[idx]
                     if cand.get("app_raw") is not None else None),
            app_mask=(cand.get("app_mask")[idx]
                      if cand.get("app_mask") is not None else None))
        trust = 1.0 - torch.sigmoid(self.prior_distrust(roots).squeeze(-1))
        value = self.root_value_scale * torch.tanh(self.root_value(roots).squeeze(-1))
        legacy = float(self.alpha_m2 * self.cap) * torch.tanh(
            self.head(roots).squeeze(-1))
        logits = trust * base + value + legacy
        bias = (cand["_trace_bias"] if "_trace_bias" in cand else
                self.trace_bias(cand.get("trace_graph"), cand.get("state_context")))
        return logits if bias is None else logits + bias[idx]

    def diagnostics_from_candidates(self, cand: dict, ids=None):
        """Detached policy decomposition used only for traces/TensorBoard."""
        dev = cand["base"].device
        idx = (torch.arange(cand["n_cand"], dtype=torch.long, device=dev)
               if ids is None else torch.as_tensor(ids, dtype=torch.long, device=dev))
        memory = cand.get("memory_evidence")
        with torch.no_grad():
            roots = self._encoded_single(
                cand["latents"][idx], cand["feats"][idx], cand["base"][idx],
                cand.get("state_context"), None if memory is None else memory[idx],
                app_raw=(cand.get("app_raw")[idx]
                         if cand.get("app_raw") is not None else None),
                app_mask=(cand.get("app_mask")[idx]
                          if cand.get("app_mask") is not None else None))
            trust = 1.0 - torch.sigmoid(self.prior_distrust(roots).squeeze(-1))
            value = self.root_value_scale * torch.tanh(
                self.root_value(roots).squeeze(-1))
            final = self.final_logits_from_candidates(cand, idx)
            trace_probability = torch.zeros(len(idx), device=final.device)
            appearance_entropy = 0.0
            appearance_count = 0
            expected_depth = 0.0
            if self.probability_trace is not None:
                trace_p, trace_diag = self.probability_trace.distribution(
                    cand.get("trace_graph"), cand.get("state_context"))
                trace_probability = trace_p[idx]
                appearance_entropy = float(
                    trace_diag.get("appearance_entropy", torch.tensor(0.0)))
                appearance_count = int(trace_diag.get("n_appearances", 0))
                depth = trace_diag.get("depth")
                if depth is not None:
                    expected_depth = float((depth * torch.arange(
                        len(depth), device=depth.device, dtype=depth.dtype)).sum())
        return {"ids": idx.detach().cpu(), "prior_trust": trust.detach().cpu(),
                "root_value": value.detach().cpu(), "final": final.detach().cpu(),
                "trace_probability": trace_probability.detach().cpu(),
                "appearance_entropy": appearance_entropy,
                "appearance_count": appearance_count,
                "expected_depth": expected_depth}

    def architecture_metadata(self):
        return {
            "name": "M2UnifiedNetMakespanInterventionActor",
            "probability_trace": self.probability_trace is not None,
            "root_latent_dim": self.root_latent_dim,
            "local_dim": self.local_dim, "state_dim": self.state_dim,
            "memory_dim": self.memory_dim,
            "relation_dim": self.relation_dim,
            "root_value_scale": self.root_value_scale,
            "root_token": [self.root_token[0].in_features, self.d_model],
            "state_token": [self.state_dim, self.d_model],
            "transformer": {"d_model": self.d_model, "nhead": self.nhead,
                            "num_layers": self.num_layers,
                            "dim_feedforward": self.dim_feedforward,
                            "dropout": 0.0, "norm_first": True,
                            "positional_encoding": False},
            "head": [self.d_model, 128, 1],
            "cap2": self.cap, "target_alpha2": self.target_alpha,
            "prior_semantics": (
                "one node net-intervention logit = learned-trust B5 evidence + "
                "appearance-specific total-probability propagation + RL context value"),
            "trainable_params": trainable_parameter_count(self),
        }

    def snapshot(self):
        with torch.no_grad():
            return {"params": {n: p.detach().clone() for n, p in self.named_parameters()},
                    "alpha_fraction": self.alpha_fraction}

    def load_snapshot(self, snap):
        params = snap.get("params", snap)
        if any(n.startswith("probability_trace.") for n in params):
            self.enable_probability_trace()
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name in params:
                    p.copy_(params[name])
        self.alpha_fraction = float(snap.get("alpha_fraction", self.alpha_fraction))


class M3ProposalSetResidualActor(nn.Module):
    """SFT-informed Proposal Attention actor over active-tier proposals + STOP."""

    def __init__(self, r6_selector: nn.Module, evidence_dim: int,
                 trajectory_dim: int = 5, use_trajectory: bool = True,
                 cap_prop: float = 1.0, cap_stop: float = 1.0,
                 target_alpha: float = 1.0, warmup_fraction: float = 0.1,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256):
        super().__init__()
        self.r6_view = FrozenR6LatentView(r6_selector)
        self.r6 = self.r6_view.selector  # compatibility with existing diagnostics
        self.evidence_dim = int(evidence_dim)
        self.trajectory_dim = int(trajectory_dim)
        self.use_trajectory = bool(use_trajectory)
        self.cap_prop = float(cap_prop)
        self.cap_stop = float(cap_stop)
        self.target_alpha = float(target_alpha)
        self.warmup_fraction = float(warmup_fraction)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.alpha_fraction = 0.0
        token_in = self.r6_view.prop_latent_dim + self.evidence_dim + 1
        self.proposal_token = nn.Sequential(
            nn.Linear(token_in, self.d_model), nn.LayerNorm(self.d_model), nn.GELU())
        state_in = self.r6_view.stop_latent_dim + self.r6_view.stop_input_dim + 1
        self.state_token = nn.Sequential(
            nn.Linear(state_in, self.d_model), nn.LayerNorm(self.d_model), nn.GELU())
        self.traj_token = (nn.Sequential(
            nn.Linear(self.trajectory_dim, self.d_model), nn.LayerNorm(self.d_model),
            nn.GELU()) if self.use_trajectory else None)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead,
            dim_feedforward=self.dim_feedforward, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.prop_head = nn.Sequential(
            nn.Linear(self.d_model, 128), nn.GELU(), nn.Linear(128, 1))
        stop_ctx_dim = self.d_model * (4 if self.use_trajectory else 3)
        self.stop_residual_head = nn.Sequential(
            nn.Linear(stop_ctx_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        # T2-M: select action granularity (single vs pair) before selecting the
        # concrete proposal.  The head is state/set conditioned and its zero
        # initialization preserves the SFT ranking inside each group.  Group
        # mass is based on mean evidence, so merely adding more pair candidates
        # cannot increase the probability of choosing a pair.
        self.granularity_head = nn.Sequential(
            nn.Linear(stop_ctx_dim, 64), nn.GELU(), nn.Linear(64, 2))
        for final in (self.prop_head[-1], self.stop_residual_head[-1],
                      self.granularity_head[-1]):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @property
    def alpha_prop(self):
        return self.target_alpha * self.alpha_fraction

    @property
    def alpha_stop(self):
        return self.target_alpha * self.alpha_fraction

    def train(self, mode: bool = True):
        super().train(mode)
        self.r6_view.selector.eval()
        return self

    def set_progress(self, cycle: int, total_cycles: int) -> None:
        warm = max(int(math.ceil(max(total_cycles, 1) * self.warmup_fraction)), 1)
        self.alpha_fraction = max(
            self.alpha_fraction, min(max(float(cycle) / warm, 0.0), 1.0))

    def _evidence(self, F_pool, evid):
        if self.evidence_dim == 0:
            return F_pool.new_zeros((len(F_pool), 0))
        if evid is None:
            return F_pool.new_zeros((len(F_pool), self.evidence_dim))
        if evid.shape != (len(F_pool), self.evidence_dim):
            raise ValueError(f"evidence must be {(len(F_pool), self.evidence_dim)}")
        return evid.detach().to(device=F_pool.device, dtype=F_pool.dtype)

    def _trajectory(self, ref, traj_ctx):
        if not self.use_trajectory:
            return None
        if traj_ctx is None:
            traj_ctx = ref.new_zeros(self.trajectory_dim)
        traj_ctx = traj_ctx.detach().to(device=ref.device, dtype=ref.dtype).reshape(-1)
        if traj_ctx.numel() != self.trajectory_dim:
            raise ValueError(f"trajectory context must have {self.trajectory_dim} fields")
        return self.traj_token(traj_ctx.reshape(1, -1))

    def components(self, F_pool, state_feat, pool_stats, evid=None, traj_ctx=None,
                   mask=None):
        F_pool = F_pool.float()
        stop_in = torch.cat([state_feat.detach().reshape(1, -1),
                             pool_stats.detach().reshape(1, -1)], dim=-1)
        h_prop, base_prop = self.r6_view.proposal(F_pool)
        h_state, base_stop = self.r6_view.stop(stop_in)
        ev = self._evidence(F_pool, evid)
        prop_token = self.proposal_token(torch.cat(
            [h_prop, ev, base_prop.reshape(-1, 1)], dim=-1))
        state_token = self.state_token(torch.cat(
            [h_state, stop_in, base_stop.reshape(1, 1)], dim=-1))
        traj_token = self._trajectory(prop_token, traj_ctx)
        specials = [state_token]
        if traj_token is not None:
            specials.append(traj_token)
        n_special = len(specials)
        seq = torch.cat(specials + [prop_token], dim=0).unsqueeze(0)
        key_padding_mask = None
        if mask is not None:
            valid = mask.to(device=prop_token.device, dtype=torch.bool).reshape(-1)
            if valid.numel() != len(prop_token):
                raise ValueError("mask length does not match proposal count")
            key_padding_mask = torch.cat([
                torch.zeros(n_special, dtype=torch.bool, device=valid.device), ~valid
            ]).unsqueeze(0)
        encoded = self.encoder(seq, src_key_padding_mask=key_padding_mask)[0]
        state_out = encoded[0]
        traj_out = encoded[1] if self.use_trajectory else None
        prop_out = encoded[n_special:]
        mean, maximum = set_mean_max(prop_out, mask)
        raw_prop = self.prop_head(prop_out).squeeze(-1)
        stop_fields = [state_out]
        if traj_out is not None:
            stop_fields.append(traj_out)
        stop_fields.extend([mean, maximum])
        raw_stop = self.stop_residual_head(
            torch.cat(stop_fields).reshape(1, -1)).reshape(-1)
        raw_granularity = self.granularity_head(
            torch.cat(stop_fields).reshape(1, -1)).reshape(-1)
        return {"h_prop_sft": h_prop, "h_state_sft": h_state,
                "base_prop": base_prop, "base_stop": base_stop,
                "token": prop_token, "proposal_output": prop_out,
                "state_output": state_out, "trajectory_output": traj_out,
                "mean": mean, "max": maximum,
                "raw_prop": raw_prop, "raw_stop": raw_stop,
                "raw_granularity": raw_granularity}

    @staticmethod
    def _hierarchical_logits(prop, pair_mask, valid, group_residual):
        """Equivalent flat logits for P(group|state)*P(action|group,state).

        The base group score is log-mean-exp rather than log-sum-exp, removing
        candidate-count bias. ``group_residual`` is the learned two-way
        single/pair preference. Missing groups receive no probability mass.
        """
        if pair_mask is None:
            return prop
        pair_mask = pair_mask.to(device=prop.device, dtype=torch.bool)
        valid = valid.to(device=prop.device, dtype=torch.bool)
        if pair_mask.shape != prop.shape or valid.shape != prop.shape:
            raise ValueError("pair_mask/valid shape must match proposal logits")
        squeeze = prop.dim() == 1
        if squeeze:
            prop = prop.unsqueeze(0)
            pair_mask = pair_mask.unsqueeze(0)
            valid = valid.unsqueeze(0)
            group_residual = group_residual.reshape(1, 2)
        out = torch.full_like(prop, -torch.inf)
        for group_index, is_pair in enumerate((False, True)):
            members = valid & (pair_mask == is_pair)
            count = members.sum(dim=1)
            masked = prop.masked_fill(~members, -torch.inf)
            lse = torch.logsumexp(masked, dim=1)
            safe_lse = torch.where(count > 0, lse, torch.zeros_like(lse))
            group_score = (safe_lse - count.clamp_min(1).to(prop.dtype).log()
                           + group_residual[:, group_index])
            within = prop - safe_lse[:, None]
            values = group_score[:, None] + within
            out = torch.where(members, values, out)
        return out[0] if squeeze else out

    def action_and_base_logits_batched(self, F_pool, state_feat, pool_stats,
                                       evid, traj_ctx, mask, pair_mask=None):
        """Padded set forward for many states in one Transformer invocation.

        Returns proposal logits and one STOP logit separately because STOP sits
        after each state's real proposal count, not after the common pad width.
        """
        if F_pool.dim() != 3 or mask.shape != F_pool.shape[:2]:
            raise ValueError("expected F_pool[B,M,D] and mask[B,M]")
        B, M, _ = F_pool.shape
        F_pool = F_pool.float()
        valid = mask.to(device=F_pool.device, dtype=torch.bool)
        state_feat = state_feat.detach().float().reshape(B, -1)
        pool_stats = pool_stats.detach().float().reshape(B, -1)
        stop_in = torch.cat([state_feat, pool_stats], dim=-1)
        h_prop, base_prop = self.r6_view.proposal(F_pool)
        h_state, base_stop = self.r6_view.stop(stop_in)
        if self.evidence_dim == 0:
            ev = F_pool.new_zeros((B, M, 0))
        else:
            ev = evid.detach().to(device=F_pool.device, dtype=F_pool.dtype)
            if ev.shape != (B, M, self.evidence_dim):
                raise ValueError("batched evidence shape mismatch")
        prop_token = self.proposal_token(torch.cat(
            [h_prop, ev, base_prop.unsqueeze(-1)], dim=-1))
        state_token = self.state_token(torch.cat(
            [h_state, stop_in, base_stop.unsqueeze(-1)], dim=-1)).unsqueeze(1)
        specials = [state_token]
        if self.use_trajectory:
            tc = traj_ctx.detach().to(device=F_pool.device, dtype=F_pool.dtype).reshape(B, -1)
            if tc.shape != (B, self.trajectory_dim):
                raise ValueError("batched trajectory-context shape mismatch")
            specials.append(self.traj_token(tc).unsqueeze(1))
        n_special = len(specials)
        seq = torch.cat(specials + [prop_token], dim=1)
        key_padding = torch.cat([
            torch.zeros((B, n_special), dtype=torch.bool, device=F_pool.device),
            ~valid], dim=1)
        encoded = self.encoder(seq, src_key_padding_mask=key_padding)
        state_out = encoded[:, 0]
        traj_out = encoded[:, 1] if self.use_trajectory else None
        prop_out = encoded[:, n_special:]
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1).to(prop_out.dtype)
        mean = (prop_out * valid.unsqueeze(-1)).sum(dim=1) / denom
        maximum = prop_out.masked_fill(~valid.unsqueeze(-1), -torch.inf).max(dim=1).values
        maximum = torch.where(valid.any(dim=1, keepdim=True), maximum,
                              torch.zeros_like(maximum))
        raw_prop = self.prop_head(prop_out).squeeze(-1)
        stop_fields = [state_out]
        if traj_out is not None:
            stop_fields.append(traj_out)
        stop_fields.extend([mean, maximum])
        raw_stop = self.stop_residual_head(torch.cat(stop_fields, dim=-1)).reshape(-1)
        raw_granularity = self.granularity_head(torch.cat(stop_fields, dim=-1))
        prop = base_prop + float(self.alpha_prop * self.cap_prop) * torch.tanh(raw_prop)
        prop = prop.masked_fill(~valid, -torch.inf)
        stop = base_stop + float(self.alpha_stop * self.cap_stop) * torch.tanh(raw_stop)
        base_prop = base_prop.masked_fill(~valid, -torch.inf)
        if pair_mask is not None:
            residual = float(self.alpha_prop * self.cap_prop) * torch.tanh(
                raw_granularity)
            prop = self._hierarchical_logits(prop, pair_mask, valid, residual)
            base_prop = self._hierarchical_logits(
                base_prop, pair_mask, valid, torch.zeros_like(residual))
        return prop, stop, base_prop, base_stop

    def action_logits(self, F_pool, state_feat, pool_stats=None, evid=None,
                      traj_ctx=None, mask=None, pair_mask=None, **_kw):
        if pool_stats is None:
            from .top1 import _pool_stats_from
            pool_stats = _pool_stats_from(F_pool)
        c = self.components(F_pool, state_feat, pool_stats, evid, traj_ctx, mask)
        prop = c["base_prop"] + float(self.alpha_prop * self.cap_prop) * torch.tanh(c["raw_prop"])
        if mask is not None:
            valid = mask.to(device=prop.device, dtype=torch.bool).reshape(-1)
            if valid.numel() != prop.numel():
                raise ValueError("mask length does not match proposal count")
            prop = prop.masked_fill(~valid, -torch.inf)
        else:
            valid = torch.ones_like(prop, dtype=torch.bool)
        if pair_mask is not None:
            residual = float(self.alpha_prop * self.cap_prop) * torch.tanh(
                c["raw_granularity"])
            prop = self._hierarchical_logits(prop, pair_mask, valid, residual)
        stop = c["base_stop"] + float(self.alpha_stop * self.cap_stop) * torch.tanh(c["raw_stop"])
        return torch.cat([prop, stop.reshape(1)])

    def base_logits(self, F_pool, state_feat, pool_stats=None, evid=None,
                    traj_ctx=None, pair_mask=None, **_kw):
        del evid, traj_ctx
        if pool_stats is None:
            from .top1 import _pool_stats_from
            pool_stats = _pool_stats_from(F_pool)
        stop_in = torch.cat([state_feat.detach().reshape(1, -1),
                             pool_stats.detach().reshape(1, -1)], dim=-1)
        _hp, bp = self.r6_view.proposal(F_pool.float())
        _hs, bs = self.r6_view.stop(stop_in)
        if pair_mask is not None:
            valid = torch.ones_like(bp, dtype=torch.bool)
            bp = self._hierarchical_logits(
                bp, pair_mask, valid, bp.new_zeros(2))
        return torch.cat([bp, bs.reshape(1)])

    def prop_scores(self, F, evid=None, traj_ctx=None):
        # Full-state diagnostics have no canonical state/pool context.  Preserve the
        # frozen R6 score instead of inventing one; actor scoring uses action_logits.
        del evid, traj_ctx
        return self.r6.prop_scores(F)

    def stop_head(self, stop_in):
        return self.r6.stop_head(stop_in)

    def snapshot(self):
        with torch.no_grad():
            return {"params": {n: p.detach().clone() for n, p in self.named_parameters()},
                    "alpha_fraction": self.alpha_fraction}

    def load_snapshot(self, snap):
        params = snap.get("params", snap)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name in params:
                    p.copy_(params[name])
        self.alpha_fraction = float(snap.get("alpha_fraction", self.alpha_fraction))
        self.r6_view.selector.eval()

    def architecture_metadata(self):
        return {
            "name": "M3ProposalAttentionResidualActor",
            "m3_latent_source": "M3Top1Selector.prop_head[4]",
            "m3_latent_dim": self.r6_view.prop_latent_dim,
            "state_latent_source": "M3Top1Selector.stop_head[1]",
            "state_latent_dim": self.r6_view.stop_latent_dim,
            "evidence_dim": self.evidence_dim,
            "trajectory_dim": self.trajectory_dim if self.use_trajectory else 0,
            "proposal_token": [self.proposal_token[0].in_features, self.d_model],
            "state_token": [self.state_token[0].in_features, self.d_model],
            "trajectory_token": ([self.trajectory_dim, self.d_model]
                                 if self.use_trajectory else None),
            "transformer": {"d_model": self.d_model, "nhead": self.nhead,
                            "num_layers": self.num_layers,
                            "dim_feedforward": self.dim_feedforward,
                            "dropout": 0.0, "norm_first": True,
                            "positional_encoding": False},
            "proposal_head": [self.d_model, 128, 1],
            "granularity_head": [self.granularity_head[0].in_features, 64, 2],
            "action_factorization": (
                "P(single_or_pair|state,set)*P(operator|state,set,single_or_pair)"),
            "stop_head": [self.stop_residual_head[0].in_features, 128, 64, 1],
            "cap3": self.cap_prop, "cap_stop": self.cap_stop,
            "target_alpha3": self.target_alpha,
            "trainable_params": trainable_parameter_count(self),
        }


def checkpoint_metadata(architecture: str, m2: nn.Module, m3: nn.Module,
                        *, caps: ResidualCaps, k: int, horizon: int):
    arch = str(architecture).upper()
    if arch not in ARCHITECTURES:
        raise ValueError(arch)
    caps.validate()
    m2_meta = (m2.architecture_metadata() if hasattr(m2, "architecture_metadata")
               else {"name": type(m2).__name__})
    m3_meta = (m3.architecture_metadata() if hasattr(m3, "architecture_metadata")
               else {"name": type(m3).__name__})
    return {
        "architecture": arch,
        "pipeline": "M2_SFT->M3_SFT->M2+M3_MultiPath_Joint_GRPO",
        "m2_latent_source": "SGSCTAttributionModelB5._encode_h_a:h_c[operation_node]",
        "m2_latent_dim": getattr(m2, "root_latent_dim", None),
        "m3_latent_source": "M3Top1Selector.prop_head[4]" if arch != ARCH_P0 else None,
        "m3_latent_dim": (getattr(getattr(m3, "r6_view", None), "prop_latent_dim", None)
                          if arch != ARCH_P0 else None),
        "m2_actor_params": trainable_parameter_count(m2),
        "m3_actor_params": trainable_parameter_count(m3),
        "frozen_actor_params": frozen_parameter_count(m2) + frozen_parameter_count(m3),
        "m2_actor": m2_meta, "m3_actor": m3_meta,
        "root_context_fields": ["B5_operation_latent", "B5_prior",
                                "local_root_features", "memory_evidence",
                                "current_state_features", "root_self_attention"],
        "proposal_context_fields": ["R6_proposal_hidden", "frozen_base_logit",
                                    "R20_runtime_evidence", "R6_STOP_hidden",
                                    "current_state_context",
                                    "active_tier_self_attention"],
        "trajectory_fields": (["step_over_H", "realized_gain_over_root_Cmax",
                               "last_gain_over_root_Cmax", "action_count_over_H",
                               "non_improving_streak_over_H"]
                              if arch == ARCH_P2 else []),
        "caps": asdict(caps),
        "alpha_schedule": "linear_warmup",
        "K": int(k), "H": int(horizon),
        "reward": "R_i=Cmax(S_root)-Cmax(S_terminal); best-of-N diagnostic only",
        "credit": "M2=A2_only; M3=A3_only",
        "active_tier": "PA_else_PB_else_PC_else_STOP",
        "formal_test_access": 0, "identified": False,
    }


__all__ = [
    "ARCH_P0", "ARCH_P1", "ARCH_P2", "ARCHITECTURES", "ResidualCaps",
    "FrozenR6LatentView", "M2RootSetResidualActor", "M3ProposalSetResidualActor",
    "calibrate_residual_caps", "checkpoint_metadata", "freeze_sft",
    "packed_set_mean_max", "padded_set_mean_max", "set_mean_max",
    "tensor_state_sha256", "trainable_parameter_count", "frozen_parameter_count",
    "trajectory_context",
]
