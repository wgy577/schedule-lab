"""Compatibility facade for the Causal Explorer V1 call signature.

The implementation is V2's real bounded beam search. Existing callers may
continue passing the six positional V1 arguments.
"""

from __future__ import annotations

from typing import Sequence

from ..m2_v5_schema_v1 import LegalEdit
from .causal_explorer_v2 import (
    ActionableRoot,
    ActionableRootSelectorConfig,
    ActionableRootSelectorV2,
    AppearanceContext,
    CauseEdge,
    CausalExplanationChain,
    CausalExplorerConfig,
    CausalExplorerV2,
    CausalSearchTrace,
    DecisionCandidate,
    load_actionable_root_selector_config,
    load_causal_explorer_config,
)
from .transition_reasoner_v1 import ScheduleGraphView


class CausalExplorer(CausalExplorerV2):
    """V1-compatible adapter backed by :class:`CausalExplorerV2`."""

    def explore(
        self,
        appearance_id: str,
        appearance_members: Sequence[str],
        decision_sites: Sequence[object],
        m2_root_scores: Sequence[float],
        legal_edits: Sequence[LegalEdit],
        schedule_graph: ScheduleGraphView,
    ) -> tuple[CausalExplanationChain, ...]:
        if len(decision_sites) != len(m2_root_scores):
            raise ValueError("decision sites and M2 scores must align")
        return super().explore(
            AppearanceContext(
                appearance_id=str(appearance_id),
                members=tuple(str(item) for item in appearance_members),
            ),
            tuple(DecisionCandidate(site, float(score))
                  for site, score in zip(decision_sites, m2_root_scores)),
            schedule_graph,
            legal_edits,
        )


class ActionableRootSelector(ActionableRootSelectorV2):
    """V1 name retained for callers; selection behavior is V2."""


__all__ = [
    "ActionableRoot",
    "ActionableRootSelector",
    "ActionableRootSelectorConfig",
    "AppearanceContext",
    "CauseEdge",
    "CausalExplanationChain",
    "CausalExplorer",
    "CausalExplorerConfig",
    "CausalExplorerV2",
    "CausalSearchTrace",
    "DecisionCandidate",
    "load_actionable_root_selector_config",
    "load_causal_explorer_config",
]
