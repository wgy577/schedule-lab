"""Deterministic VNS/ALNS/tabu search controller around an incumbent schedule."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .models import CausalInterventionPoint, ExperimentRecord
from .posterior import MultiFidelityPosterior


@dataclass(frozen=True)
class SearchProposal:
    cip_id: str
    operator: str
    closure_level: int
    strategy: str
    signature: str
    acquisition: float


class TabuMemory:
    def __init__(self, tenure: int = 32) -> None:
        self._values: deque[str] = deque(maxlen=tenure)

    def add(self, signature: str) -> None:
        self._values.append(signature)

    def __contains__(self, signature: str) -> bool:
        return signature in self._values


class DeterministicNeighborhoodPortfolio:
    """Ordered VNS first, bounded ALNS after plateau, posterior exploitation last."""

    def __init__(
        self,
        *,
        closure_levels: tuple[int, ...] = (1, 2, 3),
        tabu_tenure: int = 32,
        alns_plateau: int = 2,
    ) -> None:
        self.closure_levels = closure_levels
        self.tabu = TabuMemory(tabu_tenure)
        self.alns_plateau = alns_plateau
        self.posterior = MultiFidelityPosterior()

    @staticmethod
    def _signature(cip: CausalInterventionPoint, operator: str, level: int) -> str:
        return f"{cip.id}|{operator}|L{level}"

    def update(self, records: Iterable[ExperimentRecord]) -> None:
        for record in records:
            self.posterior.update(record)
            self.tabu.add(record.proposal_signature)

    def propose(
        self,
        candidates: Iterable[CausalInterventionPoint],
        *,
        failed_rounds: int = 0,
        limit: int = 16,
    ) -> tuple[SearchProposal, ...]:
        proposals = []
        strategy = "vns" if failed_rounds < self.alns_plateau else "bounded-alns"
        estimates = {
            item.key: item for item in self.posterior.rank()
        }
        for cip in candidates:
            for level in self.closure_levels:
                if level < cip.closure.level:
                    continue
                operators = list(cip.recommended_operators)
                if strategy == "bounded-alns":
                    operators.sort(
                        key=lambda item: (
                            item not in {
                                "critical_block_resequence",
                                "blocking_chain_repair",
                                "stage_resequence",
                            },
                            item,
                        )
                    )
                for operator in operators:
                    signature = self._signature(cip, operator, level)
                    if signature in self.tabu:
                        continue
                    posterior_key = f"{cip.diagnostic.type}|{operator}|L{level}"
                    estimate = estimates.get(posterior_key)
                    acquisition = (
                        estimate.acquisition()
                        if estimate is not None
                        else (
                            cip.score
                            / max(1.0, len(cip.closure.operation_ids))
                        )
                    )
                    proposals.append(
                        SearchProposal(
                            cip_id=cip.id,
                            operator=operator,
                            closure_level=level,
                            strategy=strategy,
                            signature=signature,
                            acquisition=acquisition,
                        )
                    )
        proposals.sort(
            key=lambda item: (
                -item.acquisition,
                item.closure_level,
                item.cip_id,
                item.operator,
            )
        )
        return tuple(proposals[:limit])
