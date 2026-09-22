"""Domain-independent project-conditioned schedule improvement laboratory."""

from .models import (
    AgentAction,
    CausalInterventionPoint,
    ExperimentRecord,
    Fidelity,
    ProjectSemantics,
)
from .unified_representation import (
    CapabilitySignature,
    FamilyProfile,
    FeasibilityMaskSnapshot,
    OperatorMaskSnapshot,
    UnifiedSchedulingRepresentation,
    build_unified_representation,
    order_schedule_for_serialization,
)

__all__ = [
    "AgentAction",
    "CausalInterventionPoint",
    "ExperimentRecord",
    "Fidelity",
    "ProjectSemantics",
    "CapabilitySignature",
    "FamilyProfile",
    "FeasibilityMaskSnapshot",
    "OperatorMaskSnapshot",
    "UnifiedSchedulingRepresentation",
    "build_unified_representation",
    "order_schedule_for_serialization",
]

__version__ = "0.10.0"
