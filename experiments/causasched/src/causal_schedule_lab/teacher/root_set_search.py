"""Root Set Search — greedy + beam, no brute force (教师模型.md §17/§19/§20).

A Root Cause Set ``B_R`` is the *minimal effective intervention set* that jointly
explains the current appearance; it is not a neighbourhood.  We search it with:

* atomic CE top atoms (``K_joint``),
* joint probes ``CE_A(B)``,
* beam search over growing sets (``beam_width``, ``max_size`` = ``K_R``),

scoring each candidate set ``B`` by ``CE_A(B) - lambda_B * |B|``.  We return the
smallest set meeting ``CE_A(B) >= tau_R``, else the best-scoring set within
``|B| <= K_R``.  Marginal gain ``MG_A(r|B)`` is also reported for the final set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .atom_generator import DecisionAtom

TAU_ROOT = 0.70
TAU_MG = 0.05
K_R = 3
BEAM_WIDTH = 4
K_JOINT = 6
LAMBDA_B = 0.10

# Joint CE oracle: takes a tuple of atoms, returns (ce, feasible).
JointProbeFn = Callable[[tuple[DecisionAtom, ...]], tuple[float, bool]]


@dataclass(frozen=True)
class RootSet:
    atoms: tuple[DecisionAtom, ...]
    ce: float
    feasible: bool = False

    @property
    def size(self) -> int:
        return len(self.atoms)

    @property
    def atom_ids(self) -> tuple[str, ...]:
        return tuple(a.atom_id for a in self.atoms)


def _score(ce: float, size: int, lambda_b: float = LAMBDA_B) -> float:
    return ce - lambda_b * size


def build_root_set(
    top_atoms: tuple[DecisionAtom, ...],
    atomic_ce: dict[str, float],
    joint_probe: JointProbeFn,
    *,
    max_size: int = K_R,
    beam_width: int = BEAM_WIDTH,
    tau_root: float = TAU_ROOT,
    tau_mg: float = TAU_MG,
    lambda_b: float = LAMBDA_B,
) -> RootSet:
    """Greedy + beam search for the minimal effective root set (§20)."""
    if not top_atoms:
        return RootSet(atoms=(), ce=0.0, feasible=False)
    width = min(beam_width, len(top_atoms))

    beams: list[RootSet] = [
        RootSet(atoms=(atom,), ce=atomic_ce.get(atom.atom_id, 0.0), feasible=atomic_ce.get(atom.atom_id, 0.0) > 0.0)
        for atom in top_atoms[:width]
    ]
    all_tested = list(beams)

    for size in range(2, max_size + 1):
        candidates: list[RootSet] = []
        for base in beams:
            for atom in top_atoms:
                if any(atom.atom_id == base_atom.atom_id for base_atom in base.atoms):
                    continue
                combined = base.atoms + (atom,)
                ce, feasible = joint_probe(combined)
                candidate = RootSet(atoms=combined, ce=ce, feasible=feasible)
                candidates.append(candidate)
                all_tested.append(candidate)
        if not candidates:
            break
        candidates.sort(
            key=lambda item: (-_score(item.ce, item.size, lambda_b), item.atom_ids)
        )
        beams = candidates[:width]
        valid = sorted(
            (b for b in beams if b.ce >= tau_root and b.feasible),
            key=lambda b: (b.size, -b.ce),
        )
        if valid:
            return valid[0]

    best = max(all_tested, key=lambda b: (_score(b.ce, b.size, lambda_b), -b.size, b.atom_ids))
    return best


def marginal_gain(
    base: tuple[DecisionAtom, ...],
    atom: DecisionAtom,
    joint_probe: JointProbeFn,
) -> float:
    """``MG_A(r|B) = CE_A(B ∪ {r}) - CE_A(B)`` (§19)."""
    ce_base, _ = joint_probe(base)
    ce_union, _ = joint_probe(base + (atom,))
    return ce_union - ce_base