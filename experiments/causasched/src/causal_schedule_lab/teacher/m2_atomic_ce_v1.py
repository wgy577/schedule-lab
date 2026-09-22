"""Appearance-conditioned atomic signed-CE learnability contracts.

This module deliberately keeps the causal target separate from optimization
utility.  One sample is exactly ``(schedule, appearance block, atomic
candidate) -> signed CE``.  Negative, zero, and positive targets are retained.

The compact models here are Phase-2.14 learnability probes.  They do not claim
causal identification and they do not replace the full SG-SCT graph/Gantt
backbone.  ``SignedAtomicCEHead`` is intentionally unbounded (no sigmoid) and
can be attached to that backbone after the learnability gate passes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


SCHEMA_VERSION = "appearance_atomic_signed_ce_sample_v1"
MODEL_SCHEMA_VERSION = "appearance_conditioned_atomic_m2_learnability_v1"
EPS = 1e-12

SCHEDULE_FEATURES = (
    "log_num_jobs",
    "log_num_machines",
    "log_schedule_makespan",
    "quality_gap",
)
APPEARANCE_FEATURES = (
    "appearance_ordinal_scaled",
)
ATOM_FEATURES = (
    "max_plus_score",
    "page_rank_score",
    "teacher_score",
    "primary_job_scaled",
    "primary_operation_scaled",
    "secondary_job_scaled",
    "secondary_operation_scaled",
    "machine_scaled",
    "target_machine_scaled",
    "same_job",
    "operation_distance_scaled",
    "feasible",
)

_ROUTING = re.compile(r"^routing:J(?P<job>\d+)\.O(?P<op>\d+)@M(?P<machine>\d+)$")
_SEQUENCING = re.compile(
    r"^sequencing:M(?P<machine>\d+):J(?P<job1>\d+)\.O(?P<op1>\d+)"
    r"<J(?P<job2>\d+)\.O(?P<op2>\d+)$"
)


def content_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_bucket(value: str, buckets: int = 128) -> int:
    if buckets < 2:
        raise ValueError("buckets must be at least two")
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (buckets - 1)


def _finite(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _scaled(index: int, total: int) -> float:
    return float(index) / max(1, int(total))


@dataclass(frozen=True)
class AtomicCESample:
    sample_id: str
    instance_uid: str
    schedule_id: str
    schedule_fingerprint: str
    split: str
    benchmark_family: str
    schedule_role: str
    appearance_id: str
    appearance_type: str
    atom_id: str
    atom_type: str
    sampling_layer: str
    schedule_features: tuple[float, ...]
    appearance_features: tuple[float, ...]
    atom_features: tuple[float, ...]
    signed_ce: float

    def validate(self) -> None:
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split: {self.split}")
        if self.atom_type not in {"routing", "sequencing"}:
            raise ValueError(f"invalid atom type: {self.atom_type}")
        if len(self.schedule_features) != len(SCHEDULE_FEATURES):
            raise ValueError("schedule feature dimension mismatch")
        if len(self.appearance_features) != len(APPEARANCE_FEATURES):
            raise ValueError("appearance feature dimension mismatch")
        if len(self.atom_features) != len(ATOM_FEATURES):
            raise ValueError("atom feature dimension mismatch")
        values = self.schedule_features + self.appearance_features + self.atom_features + (self.signed_ce,)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("sample contains NaN or Inf")


def sample_from_row(row: Mapping[str, Any]) -> AtomicCESample:
    """Convert one frozen Phase-2.11 row without filtering target sign."""

    required = {
        "instance_uid", "schedule_id", "schedule_fingerprint", "split",
        "benchmark_family", "schedule_role", "appearance_id", "appearance_type",
        "atom_id", "atom_type", "sampling_layer", "ce_atomic", "num_jobs",
        "num_machines", "schedule_makespan", "max_plus_score", "page_rank_score",
        "teacher_score",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"atomic CE row misses {sorted(missing)}")
    jobs = max(1, int(row["num_jobs"]))
    machines = max(1, int(row["num_machines"]))
    atom_id = str(row["atom_id"])
    atom_type = str(row["atom_type"])
    primary_job = primary_op = secondary_job = secondary_op = machine = target_machine = 0
    same_job = 0.0
    if atom_type == "routing":
        match = _ROUTING.fullmatch(atom_id)
        if match is None:
            raise ValueError(f"unparseable routing atom: {atom_id}")
        primary_job = int(match["job"])
        primary_op = int(match["op"])
        machine = int(match["machine"])
        # The frozen Phase-2.11 row does not contain the canonical target mode.
        # Zero is an explicit unavailable sentinel; Phase-2.13 sidecars retain it.
        target_machine = 0
    elif atom_type == "sequencing":
        match = _SEQUENCING.fullmatch(atom_id)
        if match is None:
            raise ValueError(f"unparseable sequencing atom: {atom_id}")
        machine = int(match["machine"])
        primary_job, primary_op = int(match["job1"]), int(match["op1"])
        secondary_job, secondary_op = int(match["job2"]), int(match["op2"])
        same_job = float(primary_job == secondary_job)
    else:
        raise ValueError(f"unsupported atom type: {atom_type}")
    appearance_match = re.search(r"(\d+)$", str(row["appearance_id"]))
    appearance_ordinal = int(appearance_match.group(1)) if appearance_match else 0
    sample_key = "|".join(
        str(row[key]) for key in ("instance_uid", "schedule_id", "appearance_id", "atom_id")
    )
    sample = AtomicCESample(
        sample_id="m2ce:" + hashlib.sha256(sample_key.encode("utf-8")).hexdigest()[:20],
        instance_uid=str(row["instance_uid"]),
        schedule_id=str(row["schedule_id"]),
        schedule_fingerprint=str(row["schedule_fingerprint"]),
        split=str(row["split"]),
        benchmark_family=str(row["benchmark_family"]),
        schedule_role=str(row["schedule_role"]),
        appearance_id=str(row["appearance_id"]),
        appearance_type=str(row["appearance_type"]),
        atom_id=atom_id,
        atom_type=atom_type,
        sampling_layer=str(row["sampling_layer"]),
        schedule_features=(
            math.log1p(jobs),
            math.log1p(machines),
            math.log1p(max(0.0, _finite(row["schedule_makespan"]))),
            _finite(row.get("quality_gap")),
        ),
        appearance_features=(math.log1p(appearance_ordinal) / 8.0,),
        atom_features=(
            _finite(row["max_plus_score"]),
            _finite(row["page_rank_score"]),
            _finite(row["teacher_score"]),
            _scaled(primary_job, jobs),
            _scaled(primary_op, jobs),
            _scaled(secondary_job, jobs),
            _scaled(secondary_op, jobs),
            _scaled(machine, machines),
            _scaled(target_machine, machines),
            same_job,
            _scaled(abs(primary_op - secondary_op), jobs),
            float(bool(row.get("feasible", False))),
        ),
        signed_ce=_finite(row["ce_atomic"]),
    )
    sample.validate()
    return sample


@dataclass(frozen=True)
class FeatureScaler:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, rows: np.ndarray) -> "FeatureScaler":
        if rows.ndim != 2 or rows.shape[0] == 0:
            raise ValueError("cannot fit scaler on an empty/non-matrix input")
        mean = rows.mean(axis=0)
        scale = rows.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(tuple(float(x) for x in mean), tuple(float(x) for x in scale))

    def transform(self, rows: np.ndarray) -> np.ndarray:
        return (rows - np.asarray(self.mean)) / np.asarray(self.scale)


@dataclass(frozen=True)
class AtomicCETensors:
    dense: Tensor
    schedule: Tensor
    appearance: Tensor
    atom: Tensor
    appearance_type: Tensor
    atom_type: Tensor
    family: Tensor
    schedule_role: Tensor
    sampling_layer: Tensor
    target: Tensor
    row_indices: Tensor

    def select(self, indices: Tensor) -> "AtomicCETensors":
        return AtomicCETensors(**{
            name: getattr(self, name)[indices]
            for name in self.__dataclass_fields__
        })


def tensorize_samples(
    samples: Sequence[AtomicCESample], *, scaler: FeatureScaler | None = None,
) -> tuple[AtomicCETensors, FeatureScaler]:
    if not samples:
        raise ValueError("no atomic CE samples")
    schedule = np.asarray([s.schedule_features for s in samples], dtype=np.float32)
    appearance = np.asarray([s.appearance_features for s in samples], dtype=np.float32)
    atom = np.asarray([s.atom_features for s in samples], dtype=np.float32)
    dense = np.concatenate([schedule, appearance, atom], axis=1)
    fitted = scaler or FeatureScaler.fit(dense)
    scaled = fitted.transform(dense).astype(np.float32)
    s_end = schedule.shape[1]
    a_end = s_end + appearance.shape[1]
    return AtomicCETensors(
        dense=torch.from_numpy(scaled),
        schedule=torch.from_numpy(scaled[:, :s_end]),
        appearance=torch.from_numpy(scaled[:, s_end:a_end]),
        atom=torch.from_numpy(scaled[:, a_end:]),
        appearance_type=torch.tensor([stable_bucket(s.appearance_type) for s in samples]),
        atom_type=torch.tensor([1 if s.atom_type == "routing" else 2 for s in samples]),
        family=torch.tensor([stable_bucket(s.benchmark_family) for s in samples]),
        schedule_role=torch.tensor([stable_bucket(s.schedule_role) for s in samples]),
        sampling_layer=torch.tensor([stable_bucket(s.sampling_layer) for s in samples]),
        target=torch.tensor([s.signed_ce for s in samples], dtype=torch.float32),
        row_indices=torch.arange(len(samples), dtype=torch.long),
    ), fitted


class SignedAtomicCEHead(nn.Module):
    """Unbounded final linear head for signed CE; never applies sigmoid."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, value: Tensor) -> Tensor:
        return self.linear(value).squeeze(-1)


class AtomicFeatureMLP(nn.Module):
    """Simple dense baseline without separate appearance/atom encoders."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = SignedAtomicCEHead(hidden_dim)

    def forward(self, batch: AtomicCETensors) -> Tensor:
        return self.head(self.network(batch.dense))


class AppearanceConditionedAtomicM2(nn.Module):
    """Minimal query/candidate M2 used only for the learnability gate.

    ``schedule_encoder`` is a compact schedule-summary proxy in this runner.
    The full SG-SCT graph/Gantt states can replace its output through
    ``forward_encoded`` without changing the appearance query, candidate
    encoder, or signed head.
    """

    model_schema_version = MODEL_SCHEMA_VERSION

    def __init__(self, schedule_dim: int, appearance_dim: int, atom_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.schedule_encoder = nn.Sequential(nn.Linear(schedule_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.appearance_numeric = nn.Sequential(nn.Linear(appearance_dim, hidden_dim), nn.GELU())
        self.atom_numeric = nn.Sequential(nn.Linear(atom_dim, hidden_dim), nn.GELU())
        self.appearance_embedding = nn.Embedding(128, hidden_dim)
        self.atom_embedding = nn.Embedding(3, hidden_dim)
        self.family_embedding = nn.Embedding(128, hidden_dim)
        self.role_embedding = nn.Embedding(128, hidden_dim)
        self.layer_embedding = nn.Embedding(128, hidden_dim)
        self.appearance_query = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.atomic_candidate_encoder = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.signed_ce_head = SignedAtomicCEHead(hidden_dim)

    def forward_encoded(self, schedule_state: Tensor, appearance_query: Tensor, candidate_state: Tensor) -> Tensor:
        interaction = appearance_query * candidate_state
        fused = self.fusion(torch.cat([schedule_state, appearance_query, candidate_state, interaction], dim=-1))
        return self.signed_ce_head(fused)

    def forward(self, batch: AtomicCETensors) -> Tensor:
        schedule = self.schedule_encoder(batch.schedule)
        schedule = schedule + self.family_embedding(batch.family) + self.role_embedding(batch.schedule_role)
        appearance = self.appearance_query(torch.cat([
            self.appearance_numeric(batch.appearance), self.appearance_embedding(batch.appearance_type)
        ], dim=-1))
        candidate = self.atomic_candidate_encoder(torch.cat([
            self.atom_numeric(batch.atom), self.atom_embedding(batch.atom_type), self.layer_embedding(batch.sampling_layer)
        ], dim=-1))
        return self.forward_encoded(schedule, appearance, candidate)


@dataclass(frozen=True)
class GraphGanttAtomicCandidateBatch:
    """Atomic candidates attached to an already encoded SG-SCT schedule."""

    appearance_block_index: Tensor  # [A]
    graph_index: Tensor  # [A]
    atom_type: Tensor  # [A], routing=1/sequencing=2
    atom_numeric: Tensor  # [A,F]
    primary_node_index: Tensor  # [A]
    secondary_node_index: Tensor  # [A], -1 for routing
    appearance_numeric: Tensor | None = None  # [A,F_A], frozen block measurements
    source_resource_node_index: Tensor | None = None  # [A], optional old-adapter compatibility
    target_resource_node_index: Tensor | None = None  # [A], routing required by target-conditioned adapter
    target_mode_node_index: Tensor | None = None  # [A], routing required by target-conditioned adapter
    target_machine_index: Tensor | None = None  # [A], instance-local selector, never embedded as identity

    def validate(self, *, block_count: int, graph_count: int, node_count: int) -> None:
        count = self.atom_numeric.shape[0]
        if self.atom_numeric.ndim != 2:
            raise ValueError("atom_numeric must have shape [A,F]")
        if self.appearance_numeric is not None and (
            self.appearance_numeric.ndim != 2 or self.appearance_numeric.shape[0] != count
        ):
            raise ValueError("appearance_numeric must have shape [A,F_A]")
        for name in ("appearance_block_index", "graph_index", "atom_type", "primary_node_index", "secondary_node_index"):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} must have shape [A]")
        if count and (int(self.appearance_block_index.min()) < 0 or int(self.appearance_block_index.max()) >= block_count):
            raise ValueError("appearance_block_index is out of range")
        if count and (int(self.graph_index.min()) < 0 or int(self.graph_index.max()) >= graph_count):
            raise ValueError("graph_index is out of range")
        if count and (int(self.primary_node_index.min()) < 0 or int(self.primary_node_index.max()) >= node_count):
            raise ValueError("primary_node_index is out of range")
        secondary = self.secondary_node_index >= 0
        if torch.any(secondary) and int(self.secondary_node_index[secondary].max()) >= node_count:
            raise ValueError("secondary_node_index is out of range")
        if count and not torch.all((self.atom_type == 1) | (self.atom_type == 2)):
            raise ValueError("atom_type must be routing=1 or sequencing=2")
        for name in (
            "source_resource_node_index", "target_resource_node_index",
            "target_mode_node_index", "target_machine_index",
        ):
            value = getattr(self, name)
            if value is not None and value.shape != (count,):
                raise ValueError(f"{name} must have shape [A]")
        for name in ("source_resource_node_index", "target_resource_node_index", "target_mode_node_index"):
            value = getattr(self, name)
            if value is not None:
                present = value >= 0
                if torch.any(present) and int(value[present].max()) >= node_count:
                    raise ValueError(f"{name} is out of range")


class GraphGanttAtomicCEAdapter(nn.Module):
    """Attach the atomic signed head to frozen/live SG-SCT graph+Gantt states.

    The SG-SCT output's ``reverse_embeddings`` already fuse sparse graph and
    local Gantt encodings.  This adapter conditions those states on one
    appearance query and one routing/sequencing atom.  It is implemented and
    smoke-tested, but Phase-2.14 does not train it because the frozen Phase-2.11
    rows do not carry the required per-schedule SG-SCT tensors.
    """

    def __init__(self, hidden_dim: int, atom_numeric_dim: int, appearance_numeric_dim: int = 0) -> None:
        super().__init__()
        self.atom_type_embedding = nn.Embedding(3, hidden_dim)
        self.atom_numeric_encoder = nn.Sequential(
            nn.Linear(atom_numeric_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.appearance_numeric_encoder = (
            nn.Sequential(nn.Linear(appearance_numeric_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
            if appearance_numeric_dim > 0 else None
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
        )
        self.signed_ce_head = SignedAtomicCEHead(hidden_dim)

    def forward(self, output: Any, candidates: GraphGanttAtomicCandidateBatch) -> Tensor:
        block_state = output.symptom_block_state
        if block_state is None:
            raise ValueError("SG-SCT output has no appearance-conditioned block state")
        graph_state = output.state_embeddings
        node_state = output.reverse_embeddings
        candidates.validate(
            block_count=block_state.shape[0], graph_count=graph_state.shape[0], node_count=node_state.shape[0]
        )
        primary = node_state[candidates.primary_node_index]
        secondary_valid = candidates.secondary_node_index >= 0
        secondary = torch.zeros_like(primary)
        if torch.any(secondary_valid):
            secondary[secondary_valid] = node_state[candidates.secondary_node_index[secondary_valid]]
        atom = self.candidate_encoder(torch.cat([
            primary, secondary,
            self.atom_type_embedding(candidates.atom_type),
            self.atom_numeric_encoder(candidates.atom_numeric),
        ], dim=-1))
        schedule = graph_state[candidates.graph_index]
        appearance = block_state[candidates.appearance_block_index]
        if self.appearance_numeric_encoder is not None:
            if candidates.appearance_numeric is None:
                raise ValueError("adapter requires frozen appearance numeric features")
            appearance = appearance + self.appearance_numeric_encoder(candidates.appearance_numeric)
        interaction = appearance * atom
        return self.signed_ce_head(self.fusion(torch.cat([schedule, appearance, atom, interaction], dim=-1)))


class TargetConditionedGraphGanttAtomicCEAdapter(GraphGanttAtomicCEAdapter):
    """Routing-only repair that gathers existing target Graph/Gantt states.

    Sequencing retains the original candidate encoder.  Routing drops the
    instance-local source/target/mode ordinal columns and instead reads the
    contextual source resource, target resource, target mode and target-machine
    Gantt tokens already produced by the frozen SG-SCT backbone.
    """

    def __init__(self, hidden_dim: int, atom_numeric_dim: int, appearance_numeric_dim: int = 0) -> None:
        super().__init__(hidden_dim, atom_numeric_dim, appearance_numeric_dim)
        self.target_gantt_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.routing_candidate_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.hidden_dim = hidden_dim

    @staticmethod
    def _require_context(candidates: GraphGanttAtomicCandidateBatch) -> None:
        required = (
            "source_resource_node_index", "target_resource_node_index",
            "target_mode_node_index", "target_machine_index",
        )
        if any(getattr(candidates, name) is None for name in required):
            raise ValueError("target-conditioned adapter requires canonical Graph/Gantt context indices")
        routing = candidates.atom_type == 1
        for name in required:
            value = getattr(candidates, name)
            assert value is not None
            if torch.any(routing & (value < 0)):
                raise ValueError(f"routing candidate misses {name}")

    def _target_gantt_readout(
        self,
        operation: Tensor,
        candidates: GraphGanttAtomicCandidateBatch,
        output: Any,
        gantt_machine_index: Tensor,
        gantt_graph: Tensor,
        empty_machine_fallback: Tensor,
    ) -> Tensor:
        target_machine = candidates.target_machine_index
        assert target_machine is not None
        token = output.gantt_embeddings
        if gantt_machine_index.shape != (token.shape[0],) or gantt_graph.shape != (token.shape[0],):
            raise ValueError("Gantt selector shape differs from SG-SCT output")
        query = self.target_gantt_query(operation)
        scores = query @ token.transpose(0, 1) / math.sqrt(self.hidden_dim)
        mask = (
            (gantt_machine_index.unsqueeze(0) == target_machine.unsqueeze(1))
            & (gantt_graph.unsqueeze(0) == candidates.graph_index.unsqueeze(1))
        )
        routing = candidates.atom_type == 1
        has_tokens = mask.any(dim=1)
        # A legal alternative machine can be empty in the incumbent schedule.
        # Its contextual graph resource state is the deterministic no-queue fallback.
        safe_mask = mask | (~routing | ~has_tokens).unsqueeze(1)
        attention = torch.softmax(scores.masked_fill(~safe_mask, -torch.inf), dim=1)
        readout = attention @ token
        readout = torch.where(has_tokens.unsqueeze(1), readout, empty_machine_fallback)
        return torch.where(routing.unsqueeze(1), readout, torch.zeros_like(readout))

    def forward(
        self,
        output: Any,
        candidates: GraphGanttAtomicCandidateBatch,
        *,
        gantt_machine_index: Tensor,
        gantt_graph: Tensor,
        return_fusion: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Signed-CE readout.

        ``return_fusion`` is a backward-compatible extension for Phase 2.28: when
        ``True`` the adapter also returns the pre-head fusion state ``[A,H]`` that
        feeds ``signed_ce_head``, so a parallel auxiliary head (effect trace) can
        share the exact CE trunk without altering the CE path.  The default
        (``False``) is byte-identical to the historical Phase 2.25 behaviour.
        """
        block_state = output.symptom_block_state
        if block_state is None:
            raise ValueError("SG-SCT output has no appearance-conditioned block state")
        graph_state = output.state_embeddings
        node_state = output.reverse_embeddings
        candidates.validate(
            block_count=block_state.shape[0], graph_count=graph_state.shape[0], node_count=node_state.shape[0]
        )
        self._require_context(candidates)
        primary = node_state[candidates.primary_node_index]
        secondary_valid = candidates.secondary_node_index >= 0
        secondary = torch.zeros_like(primary)
        if torch.any(secondary_valid):
            secondary[secondary_valid] = node_state[candidates.secondary_node_index[secondary_valid]]
        base_atom = self.candidate_encoder(torch.cat([
            primary, secondary,
            self.atom_type_embedding(candidates.atom_type),
            self.atom_numeric_encoder(candidates.atom_numeric),
        ], dim=-1))

        routing = candidates.atom_type == 1
        numeric = candidates.atom_numeric.clone()
        numeric[routing, 2:5] = 0.0  # remove source/target/mode ordinal identity
        source_index = candidates.source_resource_node_index
        target_index = candidates.target_resource_node_index
        mode_index = candidates.target_mode_node_index
        assert source_index is not None and target_index is not None and mode_index is not None
        source = torch.zeros_like(primary)
        target = torch.zeros_like(primary)
        mode = torch.zeros_like(primary)
        source[routing] = node_state[source_index[routing]]
        target[routing] = node_state[target_index[routing]]
        mode[routing] = node_state[mode_index[routing]]
        target_gantt = self._target_gantt_readout(
            primary, candidates, output, gantt_machine_index, gantt_graph, target
        )
        routing_atom = self.routing_candidate_encoder(torch.cat([
            primary, source, target, mode, target_gantt,
            self.atom_type_embedding(candidates.atom_type),
            self.atom_numeric_encoder(numeric),
        ], dim=-1))
        atom = torch.where(routing.unsqueeze(1), routing_atom, base_atom)
        schedule = graph_state[candidates.graph_index]
        appearance = block_state[candidates.appearance_block_index]
        if self.appearance_numeric_encoder is not None:
            if candidates.appearance_numeric is None:
                raise ValueError("adapter requires frozen appearance numeric features")
            appearance = appearance + self.appearance_numeric_encoder(candidates.appearance_numeric)
        interaction = appearance * atom
        fusion_state = self.fusion(torch.cat([schedule, appearance, atom, interaction], dim=-1))
        ce = self.signed_ce_head(fusion_state)
        if return_fusion:
            return ce, fusion_state
        return ce


class RoutingTargetEffectAtomicCEAdapter(TargetConditionedGraphGanttAtomicCEAdapter):
    """Identity-free routing readout expressed as source→target effects.

    Machine IDs remain selectors only.  The routing encoder receives existing
    contextual Graph/Gantt states through comparative terms rather than a
    target ordinal or a replacement identity embedding.  Sequencing keeps the
    Phase 2.25 path so Phase 2.27 changes only routing representation.
    """

    def __init__(self, hidden_dim: int, atom_numeric_dim: int, appearance_numeric_dim: int = 0) -> None:
        super().__init__(hidden_dim, atom_numeric_dim, appearance_numeric_dim)
        # Reuse the exact Phase 2.25 7H→H parameter budget.  Only the meaning
        # of the seven H-wide inputs changes; model scale does not grow.
        self.routing_target_effect_output = nn.Identity()

    def forward(
        self,
        output: Any,
        candidates: GraphGanttAtomicCandidateBatch,
        *,
        gantt_machine_index: Tensor,
        gantt_graph: Tensor,
    ) -> Tensor:
        block_state = output.symptom_block_state
        if block_state is None:
            raise ValueError("SG-SCT output has no appearance-conditioned block state")
        graph_state = output.state_embeddings
        node_state = output.reverse_embeddings
        candidates.validate(
            block_count=block_state.shape[0],
            graph_count=graph_state.shape[0],
            node_count=node_state.shape[0],
        )
        self._require_context(candidates)
        primary = node_state[candidates.primary_node_index]
        secondary_valid = candidates.secondary_node_index >= 0
        secondary = torch.zeros_like(primary)
        if torch.any(secondary_valid):
            secondary[secondary_valid] = node_state[
                candidates.secondary_node_index[secondary_valid]
            ]
        base_atom = self.candidate_encoder(torch.cat([
            primary,
            secondary,
            self.atom_type_embedding(candidates.atom_type),
            self.atom_numeric_encoder(candidates.atom_numeric),
        ], dim=-1))

        schedule = graph_state[candidates.graph_index]
        appearance = block_state[candidates.appearance_block_index]
        if self.appearance_numeric_encoder is not None:
            if candidates.appearance_numeric is None:
                raise ValueError("adapter requires frozen appearance numeric features")
            appearance = appearance + self.appearance_numeric_encoder(
                candidates.appearance_numeric
            )

        routing = candidates.atom_type == 1
        numeric = candidates.atom_numeric.clone()
        numeric[routing, 2:5] = 0.0
        source_index = candidates.source_resource_node_index
        target_index = candidates.target_resource_node_index
        mode_index = candidates.target_mode_node_index
        assert source_index is not None and target_index is not None and mode_index is not None
        source = torch.zeros_like(primary)
        target = torch.zeros_like(primary)
        mode = torch.zeros_like(primary)
        source[routing] = node_state[source_index[routing]]
        target[routing] = node_state[target_index[routing]]
        mode[routing] = node_state[mode_index[routing]]
        target_gantt = self._target_gantt_readout(
            primary, candidates, output, gantt_machine_index, gantt_graph, target
        )

        target_minus_source = target - source
        effect_input = torch.cat([
            target_minus_source,
            target * source,
            primary * target,
            appearance * target,
            target_gantt * primary,
            mode,
            self.atom_numeric_encoder(numeric),
        ], dim=-1)
        target_effect = self.routing_target_effect_output(
            torch.nn.functional.layer_norm(
                self.routing_candidate_encoder(effect_input) + target_minus_source,
                (self.hidden_dim,),
            )
        )
        atom = torch.where(routing.unsqueeze(1), target_effect, base_atom)
        interaction = appearance * atom
        return self.signed_ce_head(
            self.fusion(torch.cat([schedule, appearance, atom, interaction], dim=-1))
        )


@dataclass(frozen=True)
class RegressionMetrics:
    count: int
    mae: float
    rmse: float
    pearson: float | None
    spearman: float | None
    sign_accuracy: float
    sign_macro_accuracy: float


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2.0
        start = end + 1
    return ranks


def regression_metrics(target: Sequence[float], prediction: Sequence[float], sign_epsilon: float = 1e-8) -> RegressionMetrics:
    y = np.asarray(target, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("target/prediction must be same-length non-empty vectors")
    error = p - y
    pearson = float(np.corrcoef(y, p)[0, 1]) if y.std() > EPS and p.std() > EPS else None
    yr, pr = _average_ranks(y), _average_ranks(p)
    spearman = float(np.corrcoef(yr, pr)[0, 1]) if yr.std() > EPS and pr.std() > EPS else None
    sign = lambda x: np.where(x > sign_epsilon, 1, np.where(x < -sign_epsilon, -1, 0))
    ys, ps = sign(y), sign(p)
    per_class = [float(np.mean(ps[ys == label] == label)) for label in (-1, 0, 1) if np.any(ys == label)]
    return RegressionMetrics(
        count=len(y), mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error ** 2))),
        pearson=pearson, spearman=spearman, sign_accuracy=float(np.mean(ys == ps)),
        sign_macro_accuracy=float(np.mean(per_class)),
    )


def grouped_ranking_metrics(
    samples: Sequence[AtomicCESample], prediction: Sequence[float], *, ks: tuple[int, ...] = (1, 3, 5, 10),
    sequencing_ties_only: bool = False,
) -> dict[str, Any]:
    """Block-level CE ranking metrics; relevance is positive signed CE only."""

    if len(samples) != len(prediction):
        raise ValueError("sample/prediction length mismatch")
    groups: dict[tuple[str, str, str, float] | tuple[str, str, str], list[int]] = {}
    for index, sample in enumerate(samples):
        if sequencing_ties_only:
            if sample.atom_type != "sequencing":
                continue
            # Exact frozen Max-Plus tie group within one appearance block.
            key = (sample.instance_uid, sample.schedule_id, sample.appearance_id, round(sample.atom_features[0], 12))
        else:
            key = (sample.instance_uid, sample.schedule_id, sample.appearance_id)
        groups.setdefault(key, []).append(index)
    if sequencing_ties_only:
        groups = {key: ids for key, ids in groups.items() if len(ids) >= 2}
    result: dict[str, Any] = {"group_count": len(groups), "candidate_count": sum(map(len, groups.values()))}
    for k in ks:
        recalls: list[float] = []
        best: list[float] = []
        ndcg: list[float] = []
        eligible_ndcg: list[float] = []
        hits: list[float] = []
        for ids in groups.values():
            ordered = sorted(ids, key=lambda i: (-float(prediction[i]), samples[i].atom_id))
            top = ordered[:k]
            positive = {i for i in ids if samples[i].signed_ce > 0.0}
            recalls.append(len(set(top) & positive) / len(positive) if positive else 0.0)
            best.append(max((samples[i].signed_ce for i in top), default=0.0))
            hits.append(float(bool(set(top) & positive)))
            gains = [max(0.0, samples[i].signed_ce) for i in ordered[:k]]
            ideal = sorted((max(0.0, samples[i].signed_ce) for i in ids), reverse=True)[:k]
            dcg = sum((2.0 ** gain - 1.0) / math.log2(rank + 2) for rank, gain in enumerate(gains))
            idcg = sum((2.0 ** gain - 1.0) / math.log2(rank + 2) for rank, gain in enumerate(ideal))
            ndcg.append(dcg / idcg if idcg > 0 else 0.0)
            if idcg > 0:
                eligible_ndcg.append(dcg / idcg)
        result[str(k)] = {
            "recall_positive_ce": float(np.mean(recalls)) if recalls else None,
            "best_signed_ce": float(np.mean(best)) if best else None,
            "ndcg_positive_ce": float(np.mean(ndcg)) if ndcg else None,
            "eligible_positive_group_count": len(eligible_ndcg),
            "eligible_ndcg_positive_ce": float(np.mean(eligible_ndcg)) if eligible_ndcg else None,
            "hit_positive_ce": float(np.mean(hits)) if hits else None,
        }
    return result


__all__ = [
    "APPEARANCE_FEATURES", "ATOM_FEATURES", "MODEL_SCHEMA_VERSION", "SCHEMA_VERSION",
    "SCHEDULE_FEATURES", "AppearanceConditionedAtomicM2", "AtomicCESample",
    "AtomicCETensors", "AtomicFeatureMLP", "FeatureScaler", "RegressionMetrics",
    "GraphGanttAtomicCandidateBatch", "GraphGanttAtomicCEAdapter", "SignedAtomicCEHead",
    "RoutingTargetEffectAtomicCEAdapter",
    "TargetConditionedGraphGanttAtomicCEAdapter",
    "content_hash", "grouped_ranking_metrics",
    "regression_metrics", "sample_from_row", "stable_bucket", "tensorize_samples",
]
