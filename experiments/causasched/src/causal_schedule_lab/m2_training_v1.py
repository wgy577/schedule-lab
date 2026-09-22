"""V5 M2 -- supervised training loop (Phase 9, spec §44).

Multi-task supervision over the V5 heads (spec §16):

* per-block **node relevance** BCE, *masked* to each block's finite window
  (non-window entries are `-inf` and never contribute).
* per-block **edge relevance** BCE, window-masked.
* **edit relevance** BCE against the Phase-8 weak label in {-1,0,+1}.
* optional **decision-root / deviation** regression (V4-compat head as an
  auxiliary).

Training honesty contract (project doctrine):

* This round wires the loop + metrics + checkpoint and proves it **smokes and
  overfits** on a tiny single-sample target-reconstruction task.  It does NOT
  run full-scale training (SAFE_TO_TRAIN stays NO) and never claims validation.
* Checkpoint save/load is **case-bound**: the stored ``dataset_hash`` must match
  on load, so an old checkpoint cannot be silently applied to a new instance.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .sg_sct_model_v5 import M2V5Output, MODEL_SCHEMA_V5

# V5-injected modules (the twin-stream + relevance supervision heads).  These
# are the parameters the controlled overfit smoke optimises with the backbone
# frozen -- a faithful analogue of "fit M2 supervision onto a fixed M1 feature".
V5_HEAD_PREFIXES = (
    "dual_fusion",
    "relation_gate_mlp",
    "node_relevance_head",
    "edge_relevance_head",
    "edit_relevance_head",
    "edit_dependency_head",  # Patch §4 -- P(e_ij=1): editor ENABLES dependent
    "decision_type_embedding",
    "decision_reference_head",
)


def v5_head_parameters(model: nn.Module) -> Iterable[Tensor]:
    for name, p in model.named_parameters():
        if p.requires_grad and name.split(".")[0] in V5_HEAD_PREFIXES:
            yield p


@dataclass
class M2Targets:
    """Supervision bundle aligned to the V5 heads.

    Entries are optional; a missing head defaults to no gradient on that term.
    """

    node_relevance: Tensor | None = None  # [N_total]
    edge_relevance: Tensor | None = None  # [E]
    edit_weak: Tensor | None = None  # [n_edit] in {-1,0,+1}
    relation_gate: Tensor | None = None  # [B, R]
    root_logits: Tensor | None = None  # [B, N] (V4-compat auxiliary)
    edit_dependency: Tensor | None = None  # [n_pairs] in {0,1} (Patch §4)
    edit_dependency_feats: Tensor | None = None  # [n_pairs, 2*D] (Patch §4)
    decision_root: Tensor | None = None  # [B,N_site], one/multi-hot roots

    def validate(self, out: M2V5Output) -> None:
        if self.node_relevance is not None:
            assert self.node_relevance.ndim == 1
        if out.per_block_node_logits is not None and self.node_relevance is not None:
            assert out.per_block_node_logits.shape[-1] == self.node_relevance.shape[0]


def _masked_bce(logits_or_probs: Tensor, target: Tensor) -> Tensor:
    """BCE over finite (window-active) entries only; -inf are ignored."""
    mask = torch.isfinite(logits_or_probs)
    if not mask.any():
        return logits_or_probs.sum() * 0.0 + logits_or_probs.sum().detach() * 0.0
    x = logits_or_probs[mask]
    y = target.expand_as(logits_or_probs)[mask]
    if x.min() < 0.0 or x.max() > 1.0:
        return F.binary_cross_entropy_with_logits(x, y)
    return F.binary_cross_entropy(x, y)


def _edit_loss(edit_logits: Tensor | None, edit_weak: Tensor | None) -> Tensor:
    if edit_logits is None or edit_weak is None:
        return None
    # Weak labels {-1,0,+1}; zero means unknown and must not be silently turned
    # into a negative action label.
    labels = edit_weak.expand_as(edit_logits)
    known = labels != 0
    if not known.any():
        return edit_logits.sum() * 0.0
    pos = (labels[known] > 0).float()
    return F.binary_cross_entropy_with_logits(edit_logits[known], pos)


def compute_m2_loss(
    out: M2V5Output,
    targets: M2Targets,
    *,
    w_node: float = 1.0,
    w_edge: float = 1.0,
    w_edit: float = 1.0,
    w_rel: float = 0.5,
    w_dep: float = 1.0,
    w_root: float = 1.0,
) -> dict[str, Tensor]:
    """Multi-task V5 loss; returns a {name: tensor} dict (caller sums).

    ``w_dep`` weights the Patch §4 dependency term :math:`L_{dep}=BCE(y,\\hat y)`
    (editor ENABLES dependent edit) -- the third M2 loss in Stage-1
    ``L_M2 = L_root + λ1·L_edit + λ2·L_dep``.
    """
    losses: dict[str, Tensor] = {}
    if out.per_block_decision_root_logits is not None and targets.decision_root is not None:
        losses["root"] = w_root * F.binary_cross_entropy_with_logits(
            out.per_block_decision_root_logits,
            targets.decision_root.expand_as(out.per_block_decision_root_logits),
        )
    if out.per_block_node_logits is not None and targets.node_relevance is not None:
        losses["node_relevance"] = w_node * _masked_bce(
            out.per_block_node_logits, targets.node_relevance
        )
    if out.per_block_edge_logits is not None and targets.edge_relevance is not None:
        losses["edge_relevance"] = w_edge * _masked_bce(
            out.per_block_edge_logits, targets.edge_relevance
        )
    if targets.edit_weak is not None and out.per_block_edit_logits is not None:
        losses["edit_weak"] = w_edit * _edit_loss(
            out.per_block_edit_logits, targets.edit_weak
        )
    if out.per_block_relation_gates is not None and targets.relation_gate is not None:
        losses["relation_gate"] = w_rel * F.binary_cross_entropy(
            out.per_block_relation_gates, targets.relation_gate
        )
    if out.edit_dependency_logits is not None and targets.edit_dependency is not None:
        losses["dependency"] = w_dep * F.binary_cross_entropy_with_logits(
            out.edit_dependency_logits, targets.edit_dependency
        )
    return losses


def _node_window_mask(out: M2V5Output) -> Tensor:
    return torch.isfinite(out.per_block_node_logits)


def per_appearance_metrics(out: M2V5Output, targets: M2Targets) -> dict[str, float]:
    """Per-block accuracy + coverage of the windowed node-relevance head.

    ``accuracy`` = fraction of window-active nodes whose predicted relevance is
    on the right side of 0.5 vs the (binarised) target.  Sliced per block so a
    user can audit one Appearance at a time; the caller maps block index ->
    appearance via ``source_block_id``/``appearance_id``.
    """
    metrics: dict[str, float] = {}
    if out.per_block_node_logits is None or targets.node_relevance is None:
        return metrics
    mask = _node_window_mask(out)
    probs = out.per_block_node_logits
    y = targets.node_relevance.expand_as(probs)
    B, N = probs.shape
    total_acc = 0.0
    count = 0
    for b in range(B):
        m = mask[b]
        if not m.any():
            continue
        acc = float(((probs[b][m] > 0.5) == (y[b][m] > 0.5)).float().mean())
        total_acc += acc
        count += 1
        metrics[f"block_{b}_node_acc"] = acc
    if count:
        metrics["per_appearance_mean_acc"] = total_acc / count
    return metrics


class M2SupervisedTrainer:
    """AdamW multi-task trainer for the V5 M2 model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        clip_grad_norm: float = 1.0,
        device: str = "cpu",
        dropout_off: bool = False,
        optimize_heads_only: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.dropout_off = dropout_off
        params = v5_head_parameters(model) if optimize_heads_only else model.parameters()
        self.opt = torch.optim.AdamW(
            list(params), lr=lr, weight_decay=weight_decay
        )
        self.clip_grad_norm = clip_grad_norm

    def train_step(self, batch, targets: M2Targets, loss_weights: Mapping[str, float] | None = None) -> dict[str, float]:
        if not self.dropout_off:
            self.model.train()
        else:
            self.model.eval()  # deterministic overfit smoke (dropout off)
        self.opt.zero_grad()
        out = self.model(batch)
        # Patch §4 -- bridge edit-pair features through the dependency head so the
        # graph forward (which carries no edit pairs) still yields differentiable
        # dependency logits :math:`P(e_{ij}=1)` for :math:`L_{dep}`.
        if targets.edit_dependency is not None and targets.edit_dependency_feats is not None:
            dep_logits = self.model.edit_dependency_head(
                targets.edit_dependency_feats
            ).squeeze(-1)
            out.edit_dependency_logits = dep_logits
        losses = compute_m2_loss(out, targets, **({} if loss_weights is None else loss_weights))
        if not losses:
            raise ValueError("no supervised targets matched any enabled head")
        total = sum(losses.values())
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.opt.step()
        self.model.eval()
        return {k: float(v.item()) for k, v in {**losses, "total": total}.items()}

    # -- checkpoint (case-bound, schema-checked) ------------------------------

    def save_checkpoint(self, path: str | Path, dataset_hash: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_schema_version": MODEL_SCHEMA_V5,
                "dataset_hash": dataset_hash,
                "model_state": self.model.state_dict(),
                "identified": False,
                "training_status": "code_implemented_not_trained",
            },
            path,
        )

    def load_checkpoint(self, path: str | Path, dataset_hash: str) -> dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get("model_schema_version") != MODEL_SCHEMA_V5:
            raise ValueError(
                f"checkpoint schema {ckpt.get('model_schema_version')!r} != {MODEL_SCHEMA_V5!r}"
            )
        if ckpt.get("dataset_hash") != dataset_hash:
            raise ValueError(
                f"checkpoint bound to dataset_hash {ckpt.get('dataset_hash')!r}, got {dataset_hash!r}"
            )
        self.model.load_state_dict(ckpt["model_state"])
        return ckpt


def _smoke_targets() -> M2Targets:
    """Reachable strong weak targets: drive window node relevance to 0.9 and
    edge relevance to 0.1 (far from the head initialisation) so the controlled
    overfit has a clear direction to learn."""
    return M2Targets(
        node_relevance=torch.tensor(0.9),
        edge_relevance=torch.tensor(0.1),
    )


def _finite_grad_norm(model: nn.Module) -> float:
    gn = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None)
    return float(gn ** 0.5)


def run_smoke(
    bundle,
    *,
    steps: int = 40,
    device: str = "cpu",
) -> dict[str, Any]:
    """Controlled overfit smoke on a real Mk9 bundle, 1 sample.

    The backbone is frozen and only the small V5 supervision heads are trained
    (dropout off) toward a reachable strong target, so the BCE loss moves
    deterministically down to near its floor -- proof that forward, backward,
    step and the relevance heads all work on a real graph.  This is a chain
    smoke / head overfit, NOT full-scale training (identified stays False).
    """
    from .sg_sct_data_v1_3 import to_sg_sct_batch_v1_3
    from .sg_sct_model_v5 import from_manifest_v5

    model = from_manifest_v5(bundle).to(device)
    model.eval()
    trainer = M2SupervisedTrainer(
        model, lr=1e-2, dropout_off=True, optimize_heads_only=True, device=device
    )
    batch = to_sg_sct_batch_v1_3(bundle)
    targets = _smoke_targets()
    history: list[float] = []
    for _ in range(steps):
        losses = trainer.train_step(batch, targets)
        history.append(float(losses["total"]))
    return {
        "loss_history": history,
        "final_loss": history[-1],
        "first_loss": history[0],
        "decreased": history[-1] < history[0],
        "converged_to_floor": history[-1] < 0.7,  # both heads near their BCE floors
        "grad_norm": _finite_grad_norm(model),
        "identified": False,
    }


__all__ = [
    "M2Targets",
    "compute_m2_loss",
    "per_appearance_metrics",
    "M2SupervisedTrainer",
    "run_smoke",
    "MODEL_SCHEMA_V5",
]
