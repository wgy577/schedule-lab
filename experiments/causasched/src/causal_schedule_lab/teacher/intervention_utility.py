"""Intervention Utility (教师模型.md Phase2加强_2 §3.2/§6/§7).

The routing / sequencing priors are renamed to *probe utilities*:

    D(r) = InterventionUtility(r)

They answer only "how easy / cheap is it to probe this candidate, and how much
relief would a probe plausibly give" — they are NOT root causality.  They never
enter ``causal_root_score``; they only re-order probes inside the causal Top-L
via :func:`probe_ranker.probe_priority`.
"""

from __future__ import annotations

from .teacher_ranker import (
    routing_probe_utility,
    sequencing_probe_utility,
)

__all__ = ["routing_probe_utility", "sequencing_probe_utility"]