"""Experience memory for interventions and their future trajectories.

An :class:`InterventionExperience` couples the *state* (``S``), the *proposal*
(``P``: root cause + intervention actions + dependency edges) and the *outcome*
(``S'``: delta Cmax / gap / load imbalance + structural changes + outcome class).
The outcome may be ``None`` while an intervention is pending evaluation -- the
record is the unit of memory, not the action.

The immediate ``(S,P,S')`` transition is retained, and a record may additionally
carry ``S0 -> P1 -> S1 -> ... -> Pn -> Sn``.  This lets a neutral first action
receive future-success evidence only when a later action actually lowers Cmax.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .state_encoder import StateFeatures, state_vector

# Patch §7 / §6 outcome classes produced by the counterfactual evaluator.
DIRECT_SUCCESS = "direct_success"
DELAYED_SUCCESS = "delayed_success"
FAILURE = "failure"
PENDING = "pending"


@dataclass(frozen=True)
class ProposalRecord:
    """The P of (S,P,S'): root cause + intervention graph + dependency graph.

    All fields are optional with defaults; the patch V2 spec (§3) additionally
    lets M2 attach a ``target_region`` and a *pre-hoc prediction*
    (``predicted_delta_cmax`` / ``predicted_future_value`` / ``confidence``).
    These are surface metadata -- not the deterministic
    ``intervention_actions``/``dependency_edges`` the evaluator executes.
    """

    appearance_type: str = ""
    root_nodes: tuple[str, ...] = ()
    intervention_actions: tuple[str, ...] = ()  # edit ids (routing/sequence/insert)
    dependency_edges: tuple[tuple[str, str], ...] = ()  # (editor, dependent) edit ids
    affected_region: tuple[str, ...] = ()
    proposal_id: str = ""
    # Patch V2 §3 optional M2 metadata (prediction layer, not the executed graph).
    target_region: str = ""
    predicted_delta_cmax: float = 0.0
    predicted_future_value: float = 0.0
    confidence: float = 0.0
    # Causal Explorer / Operator Reasoning provenance.
    causal_chain: tuple[str, ...] = ()
    root_decision_id: str = ""
    operator_type: str = ""
    causal_search_trace: tuple[str, ...] = ()
    causal_relations: tuple[str, ...] = ()
    causal_chain_depth: int = 0
    causal_explanation_gain: float = 0.0
    causal_root_position: float = 0.0
    estimated_action_complexity: float = 0.0


@dataclass(frozen=True)
class Outcome:
    """The S' of (S,P,S'): post-intervention deltas + structural changes."""

    delta_cmax: float = 0.0
    delta_gap: float = 0.0
    delta_load_imbalance: float = 0.0
    delta_processing_excess: float = 0.0
    collateral_damage: float = 0.0
    ready_tightness: float = 0.0
    critical_block_change: float = 0.0
    critical_block_worsening: float = 0.0
    feasibility_degradation: float = 0.0
    new_anomaly_rate: float = 0.0
    risk: float = 0.0
    structural_changes: tuple[str, ...] = ()  # e.g. "critical_block_reduced","rebalanced"
    classification: str = PENDING  # DIRECT_SUCCESS | DELAYED_SUCCESS | FAILURE | PENDING

    def validate(self) -> None:
        if self.classification not in (DIRECT_SUCCESS, DELAYED_SUCCESS, FAILURE, PENDING):
            raise ValueError(f"unknown outcome class: {self.classification!r}")
        if self.classification != PENDING:
            if not (self.delta_cmax == self.delta_cmax):  # finite guard
                raise ValueError("delta_cmax must be finite")


@dataclass(frozen=True)
class TrajectoryStep:
    """One real transition in a continued intervention trajectory."""

    step_index: int
    proposal: ProposalRecord
    state_before: StateFeatures
    state_after: StateFeatures | None
    delta_cmax: float
    continued: bool = False
    outcome: Outcome | None = None
    validation_metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryRecord:
    """Normalized Effect-Predictor view of one persisted trajectory.

    Successful, failed and delayed/partial records use the same schema so
    training retains contrastive evidence rather than a success-only library.
    """

    key: str
    before_state: StateFeatures
    causal_chain: tuple[str, ...]
    root_decision: str
    operator: str
    proposal: ProposalRecord
    after_state: StateFeatures | None
    delta_cmax: float
    fiv: float
    success: str
    risk: float


@dataclass(frozen=True)
class InterventionExperience:
    """One stored (S, P, S') transition."""

    key: str
    state: StateFeatures
    proposal: ProposalRecord
    outcome: Outcome | None = None
    after_state: StateFeatures | None = None
    trajectory: tuple[TrajectoryStep, ...] = ()
    trajectory_final_delta_cmax: float | None = None
    trajectory_future_success: bool | None = None
    trajectory_final_gain: float = 0.0
    trajectory_future_steps: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def with_outcome(
        self, outcome: Outcome, after_state: StateFeatures | None = None
    ) -> "InterventionExperience":
        trajectory = self.trajectory
        if not trajectory:
            trajectory = (TrajectoryStep(
                step_index=0,
                proposal=self.proposal,
                state_before=self.state,
                state_after=after_state,
                delta_cmax=float(outcome.delta_cmax),
            ),)
        final_delta = (
            float(after_state.cmax - self.state.cmax)
            if after_state is not None else float(outcome.delta_cmax)
        )
        return InterventionExperience(
            key=self.key, state=self.state, proposal=self.proposal,
            outcome=outcome, after_state=after_state,
            trajectory=trajectory,
            trajectory_final_delta_cmax=final_delta,
            trajectory_future_success=final_delta < -1e-9,
            trajectory_final_gain=max(0.0, -final_delta),
            trajectory_future_steps=0,
            metadata=self.metadata,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def experience_key(
    state: StateFeatures | Sequence[float],
    proposal: ProposalRecord | str = "",
) -> str:
    """Content key over the full ``(S,P)`` structure, never proposal_id alone."""
    state_payload = asdict(state) if isinstance(state, StateFeatures) else {
        "legacy_vector": [round(float(x), 6) for x in state]
    }
    if isinstance(proposal, ProposalRecord):
        proposal_payload = asdict(proposal)
        proposal_payload.pop("proposal_id", None)
    else:
        # Backward-compatible call shape; the string is deliberately ignored.
        proposal_payload = {}
    raw = json.dumps(
        {"state": state_payload, "proposal": proposal_payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def experience_future_success(experience: InterventionExperience) -> bool:
    """Whether the observed trajectory eventually lowered Cmax."""
    if experience.trajectory_future_success is not None:
        return bool(experience.trajectory_future_success)
    return bool(
        experience.outcome is not None
        and experience.outcome.delta_cmax < -1e-9
    )


def experience_final_gain(experience: InterventionExperience) -> float:
    """Final observed Cmax gain relative to the trajectory's initial state."""
    if experience.trajectory_final_delta_cmax is not None:
        return max(0.0, -float(experience.trajectory_final_delta_cmax))
    if experience.outcome is None:
        return 0.0
    return max(0.0, -float(experience.outcome.delta_cmax))


def experience_training_eligible(experience: InterventionExperience) -> bool:
    """Strict label firewall: only attribution-clean frozen-local rows train."""
    metadata = experience.metadata
    return bool(
        metadata.get("counterfactual_mode") == "frozen_local"
        and metadata.get("training_eligible") is True
        and metadata.get("proposal_legal") is True
        and metadata.get("actions_executed") is True
        and metadata.get("before_feasible") is True
        and metadata.get("after_feasible") is True
        and metadata.get("delta_cmax_verified") is True
        and metadata.get("validator_passed") is True
        and not metadata.get("outside_closure_changes")
    )


class ExperienceStore:
    """Atomic, optionally-persisted store of (S,P,S').

    In-memory ``dict`` keyed by :func:`experience_key`; :meth:`append` coalesces
    on duplicate keys; :meth:`snapshot` returns a JSON-serialisable form.
    """

    def __init__(self, *, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._records: dict[str, InterventionExperience] = {}
        if self.path and self.path.exists():
            self._load()

    # -- mutation -------------------------------------------------------------

    def append(
        self,
        state: StateFeatures,
        proposal: ProposalRecord,
        *,
        outcome: Outcome | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        base_key = experience_key(state, proposal)
        existing = self._records.get(base_key)
        if existing is not None and existing.outcome is None and outcome is None:
            return base_key
        key = base_key
        suffix = 1
        # Repeated observations are evidence, not duplicates to overwrite.
        while key in self._records:
            key = f"{base_key}-{suffix}"
            suffix += 1
        rec = InterventionExperience(
            key=key, state=state, proposal=proposal,
            outcome=outcome, metadata=metadata or {},
        )
        self._records[key] = rec
        if self.path:
            self._write()
        return key

    def record_outcome(
        self,
        key: str,
        outcome: Outcome,
        *,
        after_state: StateFeatures | None = None,
        metadata_update: Mapping[str, object] | None = None,
    ) -> None:
        """Attach the evaluated (S,P,S') outcome to a stored experience."""
        cur = self._records.get(key)
        if cur is None:
            raise KeyError(f"no experience with key {key!r}")
        updated = cur.with_outcome(outcome, after_state)
        if metadata_update:
            trajectory = list(updated.trajectory)
            if trajectory:
                trajectory[0] = replace(
                    trajectory[0], outcome=outcome,
                    validation_metadata=dict(metadata_update),
                )
            updated = replace(
                updated,
                trajectory=tuple(trajectory),
                metadata={**dict(cur.metadata), **dict(metadata_update)},
            )
        self._records[key] = updated
        if self.path:
            self._write()

    def append_trajectory_step(
        self,
        key: str,
        proposal: ProposalRecord,
        outcome: Outcome,
        *,
        after_state: StateFeatures,
        validation_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Continue an existing trajectory without replacing its first outcome.

        The trajectory endpoint is always measured against the original ``S0``.
        A neutral P1 therefore becomes future-success evidence only after a real
        later state has ``Cmax(Sn) < Cmax(S0)``.
        """
        cur = self._records.get(key)
        if cur is None:
            raise KeyError(f"no experience with key {key!r}")
        before = cur.trajectory[-1].state_after if cur.trajectory else cur.after_state
        before = before or cur.state
        prior = list(cur.trajectory)
        if prior:
            prior[-1] = replace(prior[-1], continued=True)
        prior.append(TrajectoryStep(
            step_index=len(prior), proposal=proposal, state_before=before,
            state_after=after_state, delta_cmax=float(outcome.delta_cmax),
            continued=False, outcome=outcome,
            validation_metadata=validation_metadata or {},
        ))
        final_delta = float(after_state.cmax - cur.state.cmax)
        self._records[key] = replace(
            cur,
            trajectory=tuple(prior),
            trajectory_final_delta_cmax=final_delta,
            trajectory_future_success=final_delta < -1e-9,
            trajectory_final_gain=max(0.0, -final_delta),
            trajectory_future_steps=max(len(prior) - 1, 0),
        )
        if self.path:
            self._write()

    # -- query ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, key: str) -> bool:
        return key in self._records

    def get(self, key: str) -> InterventionExperience | None:
        return self._records.get(key)

    def experiences(self) -> tuple[InterventionExperience, ...]:
        return tuple(self._records.values())

    def resolved(self) -> tuple[InterventionExperience, ...]:
        """Only experiences whose outcome has been evaluated."""
        return tuple(e for e in self._records.values() if e.outcome is not None)

    def trajectory_records(self) -> tuple[TrajectoryRecord, ...]:
        """Return resolved rows in the frozen Effect-Predictor schema."""
        records: list[TrajectoryRecord] = []
        for experience in self.resolved():
            outcome = experience.outcome
            if outcome is None:
                continue
            future_success = experience_future_success(experience)
            immediate_success = float(outcome.delta_cmax) < -1e-9
            status = (
                "success" if immediate_success
                else "partial_success" if future_success
                else "failure"
            )
            gain = experience_final_gain(experience)
            records.append(TrajectoryRecord(
                key=experience.key,
                before_state=experience.state,
                causal_chain=experience.proposal.causal_chain,
                root_decision=experience.proposal.root_decision_id,
                operator=experience.proposal.operator_type,
                proposal=experience.proposal,
                after_state=experience.after_state,
                delta_cmax=float(outcome.delta_cmax),
                fiv=float(gain if future_success else 0.0),
                success=status,
                risk=float(outcome.risk),
            ))
        return tuple(records)

    # -- persistence ----------------------------------------------------------

    def _record_dict(self) -> dict:
        return {k: json.loads(e.to_json()) for k, e in self._records.items()}

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._record_dict(), sort_keys=True, indent=1))
        tmp.replace(self.path)

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        from dataclasses import fields
        from typing import get_origin, get_type_hints

        def _coerce(ftype, raw):
            # JSON serialises tuples as lists; coerce tuple-typed fields back.
            if ftype is not None and get_origin(ftype) is tuple and isinstance(raw, list):
                return tuple(raw)
            return raw

        def _ff(t, raw):
            hints = get_type_hints(t)
            return t(**{f.name: _coerce(hints.get(f.name), raw.get(f.name, f.default))
                        for f in fields(t)})

        self._records = {}
        for k, v in data.items():
            try:
                trajectory = tuple(
                    TrajectoryStep(
                        step_index=int(item.get("step_index", index)),
                        proposal=_ff(ProposalRecord, item.get("proposal", {})),
                        state_before=_ff(StateFeatures, item.get("state_before", {})),
                        state_after=(_ff(StateFeatures, item["state_after"])
                                     if item.get("state_after") else None),
                        delta_cmax=float(item.get("delta_cmax", 0.0)),
                        continued=bool(item.get("continued", False)),
                        outcome=(_ff(Outcome, item["outcome"])
                                 if item.get("outcome") else None),
                        validation_metadata=item.get("validation_metadata", {}),
                    )
                    for index, item in enumerate(v.get("trajectory", ()))
                )
                self._records[k] = InterventionExperience(
                    key=str(v.get("key", k)),
                    state=_ff(StateFeatures, v.get("state", {})),
                    proposal=_ff(ProposalRecord, v.get("proposal", {})),
                    outcome=(_ff(Outcome, v["outcome"]) if v.get("outcome") else None),
                    after_state=(_ff(StateFeatures, v["after_state"])
                                 if v.get("after_state") else None),
                    trajectory=trajectory,
                    trajectory_final_delta_cmax=v.get("trajectory_final_delta_cmax"),
                    trajectory_future_success=v.get("trajectory_future_success"),
                    trajectory_final_gain=float(v.get("trajectory_final_gain", 0.0)),
                    trajectory_future_steps=int(v.get(
                        "trajectory_future_steps", max(len(trajectory) - 1, 0)
                    )),
                    metadata=v.get("metadata", {}),
                )
            except (TypeError, ValueError):
                continue  # skip corrupt rows, never crash the whole store

    def snapshot(self) -> Mapping[str, object]:
        return self._record_dict()


__all__ = [
    "DIRECT_SUCCESS",
    "DELAYED_SUCCESS",
    "FAILURE",
    "PENDING",
    "ProposalRecord",
    "Outcome",
    "TrajectoryStep",
    "TrajectoryRecord",
    "InterventionExperience",
    "ExperienceStore",
    "experience_key",
    "experience_future_success",
    "experience_final_gain",
    "experience_training_eligible",
]
