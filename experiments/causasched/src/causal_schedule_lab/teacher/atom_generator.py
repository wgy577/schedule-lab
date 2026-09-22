"""Root Atoms and the rule probe (教师模型.md §10/§12/§14).

Propagation yields an operation relevance ``R_A(v)``.  The root-cause unit is
not the operation but the *intervenable decision atom*: a routing decision
(``routing:op@machine``), a sequencing decision (``sequencing:machine:left<right``),
or an intermediate operation atom.  Each atom maps onto the operator/parameter
candidates the V1 executor can actually probe (the rule probe), so the atom is
both a ranked candidate and directly executable.

Generating an atom for an operation is opportunistic: a routing atom is legal
only when the operation has more than one eligible machine (``|E_o| > 1``), and
a sequencing atom only when the operation sits on a realized machine sequence
with a partner.  ``generate_operator_candidates`` is reused as the rule probe to
produce the concrete probes each atom can run.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from ..ir import Problem, Schedule
from ..operator_registry_v1 import generate_operator_candidates
from .propagation_graph import RealizedPropagationGraph

ATOM_SOURCE_TEACHER = "teacher"


@dataclass(frozen=True)
class DecisionAtom:
    """A root-cause candidate unit: an intervenable routing/sequencing decision."""

    atom_id: str
    atom_type: str  # "routing" | "sequencing" | "operation"
    operation: str
    machine: str | None
    partner_operation: str | None
    source: str = ATOM_SOURCE_TEACHER
    relevance: float = 0.0
    probe_operator_id: str = ""
    probe_parameters: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.probe_parameters is None:
            object.__setattr__(self, "probe_parameters", {})

    @property
    def operations(self) -> tuple[str, ...]:
        """The operations this atom's effect depends on (for ``Rel_A``)."""
        # routing atoms: operation = the single op id.
        # sequencing pair atoms (critical_block_resequence): operation = (L, R),
        # unpack so downstream consumers get flat op ids, not a nested tuple.
        base = tuple(self.operation) if isinstance(self.operation, tuple) else (self.operation,)
        if self.partner_operation is not None:
            return base + (self.partner_operation,)
        return base

    def as_record(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "atom_type": self.atom_type,
            "operation": self.operation,
            "machine": self.machine,
            "partner_operation": self.partner_operation,
            "source": self.source,
            "relevance": self.relevance,
            "probe_operator_id": self.probe_operator_id,
            "probe_parameters": self.probe_parameters,
        }


def _operation_relevance(
    operation: str,
    R: dict[str, float],
) -> float:
    return float(R.get(operation, 0.0))


def generate_root_atoms(
    problem: Problem,
    schedule: Schedule,
    graph: RealizedPropagationGraph,
    R: dict[str, float],
    *,
    top_k: int = 24,
    relevance_threshold: float = 1e-6,
    maximum_per_operator: int = 8,
    neighborhood_radius: int = 0,
) -> tuple[DecisionAtom, ...]:
    """Map high-relevance operations to executable decision atoms (rule probe).

    Only operations with at least one legal probe become atoms.  The atom's
    ``probe_operator_id`` / ``probe_parameters`` are the first legal candidate
    the V1 executor can run, so the atom double-checks as a probe.
    """
    ranked = sorted(
        (op for op in graph.nodes if _operation_relevance(op, R) >= relevance_threshold),
        key=lambda op: (-_operation_relevance(op, R), op),
    )
    operation_map = problem.operation_map()
    mode_map = problem.mode_map()
    assignment_map = schedule.assignment_map()

    chosen = ranked[:top_k]
    atoms: list[DecisionAtom] = []

    for operation in chosen:
        relevance = _operation_relevance(operation, R)
        assignment = assignment_map.get(operation)
        if assignment is None or operation not in operation_map:
            continue
        operation_obj = operation_map[operation]
        current_mode = mode_map[assignment.mode_id][1]
        current_machine = current_mode.resources[0] if current_mode.resources else None

        # Routing atom: legal only when >1 eligible machine.
        candidates = generate_operator_candidates(
            problem,
            schedule,
            root_operations=(operation,),
            maximum_per_operator=maximum_per_operator,
            neighborhood_radius=neighborhood_radius,
        )
        legal = [c for c in candidates if c.legal]
        routing = [c for c in legal if c.operator_id in {
            "machine_reassignment", "stage_machine_reassignment",
        }]
        relevant_modes = [m for m in operation_obj.modes if m.id != assignment.mode_id]
        if relevant_modes and len(operation_obj.modes) > 1:
            target_machine = relevant_modes[0].resources[0] if relevant_modes[0].resources else None
            probe = routing[0] if routing else None
            atoms.append(
                DecisionAtom(
                    atom_id=f"routing:{operation}@{current_machine}",
                    atom_type="routing",
                    operation=operation,
                    machine=current_machine,
                    partner_operation=None,
                    source=ATOM_SOURCE_TEACHER,
                    relevance=relevance,
                    probe_operator_id=probe.operator_id if probe else "",
                    probe_parameters=dict(probe.parameters) if probe else {},
                )
            )

        # Sequencing atom: realized machine partner (left/right neighbour).
        for score in legal:
            if score.operator_id not in {
                "adjacent_resource_swap", "resource_sequence_insertion",
                "critical_block_resequence",
            }:
                continue
            params = score.parameters
            partner = params.get(
                "left_operation_id", params.get("right_operation_id",
                params.get("predecessor_id", params.get("successor_id")))
            )
            if partner is None:
                continue
            partner = str(partner)
            atoms.append(
                DecisionAtom(
                    atom_id=f"sequencing:{current_machine}:{operation}<{partner}",
                    atom_type="sequencing",
                    operation=operation,
                    machine=current_machine,
                    partner_operation=partner,
                    source=ATOM_SOURCE_TEACHER,
                    relevance=max(relevance, _operation_relevance(partner, R)),
                    probe_operator_id=score.operator_id,
                    probe_parameters=dict(score.parameters),
                )
            )

    return tuple(atoms)


def atom_to_candidate(
    atom: DecisionAtom,
    problem: Problem,
    schedule: Schedule,
    *,
    maximum_per_operator: int = 8,
    neighborhood_radius: int = 0,
):
    """Return the concrete ``OperatorParameterCandidate`` for ``atom`` (rule probe).

    Rebuilds the candidate set for the atom's operation and returns the legal
    candidate whose ``candidate_id`` matches the atom's stored probe.  Returns
    ``None`` if the stored probe is no longer legal (stale atom).
    """
    from ..training_v1 import OperatorParameterCandidate

    # sequencing pair-atoms carry operation as a (left, right) tuple; a
    # multi-op root must be UNPACKED (each root is an op id), wrapping again
    # would send `(('L','R'),)` and trip generate_operator_candidates' unknown-root check.
    operation_tuple = atom.operation if isinstance(atom.operation, tuple) else (atom.operation,)
    # Rebuild under the SAME generation signature the probe was created with
    # (decision_time=0 + subject_operations_only=True). Without these, e.g. a
    # critical_block_resequence pair-atom regenerates a different pair candidate
    # set and the stored probe never matches -> poison "unsupported" for a
    # perfectly legal resequence.
    candidates = generate_operator_candidates(
        problem,
        schedule,
        root_operations=operation_tuple,
        maximum_per_operator=maximum_per_operator,
        neighborhood_radius=neighborhood_radius,
        decision_time=0,
        subject_operations_only=True,
    )
    for candidate in candidates:
        if candidate.operator_id == atom.probe_operator_id and candidate.parameters == atom.probe_parameters:
            return candidate
    return None


def _atom_json(atom: DecisionAtom) -> str:
    return json.dumps(atom.as_record(), sort_keys=True, ensure_ascii=False)