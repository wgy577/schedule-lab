"""V5 M2 -- training-data generator (Phase 8, spec §45-1..§45-7).

Produces decision-level supervision examples per appearance/root block:

* **Weak labels** -- :math:`y \\in \\{-1,0,+1\\}` from a *decision-time-only*
  proxy: moving onto a relatively-lighter receiver is weakly positive, onto a
  heavier receiver weakly negative, ambiguity (equal) is 0.  The label NEVER
  sees the post-perturbation makespan or any "after" estimate (no leakage).
* **Positive / unknown / hard-negative taxonomy** -- legal relief edits that
  improve the proxy are *positive*; legal-but-ambiguous edits are *unknown*
  (soft-positive, NOT all unlabelled forced to hard-negative); edits targeting
  an **illegal** machine (not in the operation's eligible set) are
  *hard-negative* -- the network must learn never to score an infeasible route.
* **Hard negatives** are explicitly synthesised, never silently inferred.

Label kinds (spec §45-7, no "unlabeled = hard-negative by default"):

* ``positive``    -- legal relief, weak_label +1
* ``positive_weak``-- legal relief, weak_label +1 (equal-magnitude tie broken)
* ``unknown``     -- legal edit, weak_label 0 (ambiguous or not decidable)
* ``negative_weak``-- legal edit, weak_label -1 (moving onto a heavier receiver)
* ``hard_negative``-- ILLEGAL machine target (explicit)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .ir import Problem, Schedule
from .legal_edit_enumerator import LegalEditEnumerator, enumerate_legal_edits
from .m2_v5_schema_v1 import LegalEdit


@dataclass
class DataSample:
    """One decision-level supervision example (no after-state leakage)."""

    sample_id: str
    block_id: str
    root_op: str
    edit: LegalEdit | None
    weak_label: int  # +1 / 0 / -1
    label_kind: str  # positive|positive_weak|unknown|negative_weak|hard_negative
    source_machine: str | None
    target_machine: str | None
    decision_time_features: tuple[tuple[str, float], ...] = ()


class DecisionTimeLabeler:
    """Weak label from decision-time-visible quantities only (spec §45-11)."""

    def _weak_label(self, edit: LegalEdit, loads, makespan, eligible) -> int:
        f = dict(edit.features)
        s_rel = f.get("source_relative_load", 0.0)
        t_rel = f.get("target_relative_load", 0.0)
        gap = (s_rel - t_rel)
        if abs(gap) < 1e-9:
            return 0
        return 1 if gap > 0 else -1


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass
class RootPerturbationGenerator:
    """Deterministic weak-supervision sampler over legal edits per root."""

    problem: Problem
    schedule: Schedule
    labels: DecisionTimeLabeler = field(default_factory=DecisionTimeLabeler)
    seeds: tuple[int, ...] = (0,)

    def __post_init__(self):
        self.enum = LegalEditEnumerator(self.problem, self.schedule)

    # -- sampling ------------------------------------------------------------

    def _legal_relief_samples(self, root_op: str, block_id: str) -> list[DataSample]:
        edits = enumerate_legal_edits(
            self.problem, self.schedule, [root_op],
            request_seq_insert=False, request_seq_swap=False,
        )
        out: list[DataSample] = []
        f = dict(self.enum.eligible.get(root_op, {}))
        for e in edits:
            lab = self.labels._weak_label(e, self.enum.loads, self.enum.makespan, self.enum.eligible)
            kind = "positive" if lab == 1 else ("negative_weak" if lab == -1 else "unknown")
            out.append(DataSample(
                sample_id=f"{block_id}|{e.edit_id}",
                block_id=block_id, root_op=root_op, edit=e,
                weak_label=lab, label_kind=kind,
                source_machine=e.source_machine, target_machine=e.target_machine,
                decision_time_features=e.features,
            ))
        return out

    def _illegal_hard_negatives(self, root_op: str, block_id: str) -> list[DataSample]:
        """Explicit hard negatives: a machine NOT in the op's eligible set."""
        eligible_machines = set(self.enum.eligible.get(root_op, {}))
        all_machines = set(self.enum.loads.keys()) | {
            m for od in self.enum.eligible.values() for m in od
        }
        illegal = sorted(all_machines - eligible_machines)
        out: list[DataSample] = []
        for m in illegal[:2]:
            out.append(DataSample(
                sample_id=f"{block_id}|HARD:{root_op}->{m}",
                block_id=block_id, root_op=root_op, edit=None,
                weak_label=-1, label_kind="hard_negative",
                source_machine=None, target_machine=m,
                decision_time_features=(),
            ))
        return out

    # -- public --------------------------------------------------------------

    def generate(self, blocks: Sequence[tuple[str, Sequence[str]]]) -> list[DataSample]:
        """Yield samples for every block/root; positive, unknown and explicit
        hard-negative examples are all represented."""
        out: list[DataSample] = []
        from .m2_root_localizer_v1 import RootDecisionLocalizer

        loc = RootDecisionLocalizer(self.problem, self.schedule)
        for block_id, members in blocks:
            res = loc.localize(block_id, members)
            if res.unresolved:
                continue
            for root in res.root_sites:
                oid = root.operation_id
                out.extend(self._legal_relief_samples(oid, block_id))
                out.extend(self._illegal_hard_negatives(oid, block_id))
        return out

    def label_distribution(self, samples: Sequence[DataSample]) -> dict[str, int]:
        dist: dict[str, int] = {}
        for s in samples:
            dist[s.label_kind] = dist.get(s.label_kind, 0) + 1
        return dist


def generate_training_samples(
    problem: Problem,
    schedule: Schedule,
    blocks: Sequence[tuple[str, Sequence[str]]],
) -> list[DataSample]:
    """Convenience wrapper (deterministic)."""
    return RootPerturbationGenerator(problem, schedule).generate(blocks)


__all__ = [
    "DecisionTimeLabeler",
    "RootPerturbationGenerator",
    "DataSample",
    "generate_training_samples",
]