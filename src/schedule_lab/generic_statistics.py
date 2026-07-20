from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_generic_evidence(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    observations: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted({str(Path(item).expanduser().resolve()) for item in paths}):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for family_key, result in sorted(payload.items()):
            if not isinstance(result, dict) or "experiments" not in result:
                continue
            baseline = int(result["baselineMakespan"])
            family = str(result.get("family", family_key)).upper()
            total_operations = max(
                (int(item["releasedCount"]) + int(item["frozenCount"]) for item in result["experiments"]),
                default=1,
            )
            for experiment in result["experiments"]:
                candidate = experiment.get("candidateMakespan")
                if candidate is None:
                    continue
                signature = str(experiment["signature"])
                observation = {
                    "source": path,
                    "family": family,
                    "signature": signature,
                    "bottleneckKind": str(experiment["bottleneckKind"]),
                    "moveDepth": int(experiment.get("moveDepth", 1)),
                    "radius": int(experiment.get("radius", 1)),
                    "releasedFraction": int(experiment["releasedCount"]) / total_operations,
                    "baselineMakespan": baseline,
                    "candidateMakespan": int(candidate),
                    "gain": baseline - int(candidate),
                    "improved": bool(experiment["accepted"]) and int(candidate) < baseline,
                }
                observations[(family, signature)] = observation
    return sorted(observations.values(), key=lambda item: (item["family"], item["signature"]))


def _posterior(successes: int, trials: int, alpha: float, beta: float) -> dict[str, float]:
    a, b = alpha + successes, beta + trials - successes
    total = a + b
    mean = a / total
    variance = a * b / (total * total * (total + 1))
    return {"alpha": a, "beta": b, "mean": mean, "standardDeviation": math.sqrt(variance)}


def rank_generic_evidence(
    observations: list[dict[str, Any]],
    *,
    pool_strength: float = 2.0,
    exploration_weight: float = 0.5,
    release_penalty: float = 0.1,
    retire_after_zero_improvements: int = 6,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("no generic neighborhood evidence found")
    global_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    family_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        global_groups[(item["bottleneckKind"], item["moveDepth"])].append(item)
        family_groups[(item["family"], item["bottleneckKind"], item["moveDepth"])].append(item)
    positive_gains = [float(item["gain"]) for item in observations if item["gain"] > 0]
    global_gain = statistics.mean(positive_gains) if positive_gains else 1.0
    arms = []
    for (family, bottleneck_kind, move_depth), records in sorted(family_groups.items()):
        pooled = global_groups[(bottleneck_kind, move_depth)]
        pooled_rate = (1 + sum(int(item["improved"]) for item in pooled)) / (2 + len(pooled))
        alpha = 1.0 + pool_strength * pooled_rate
        beta = 1.0 + pool_strength * (1.0 - pooled_rate)
        successes = sum(int(item["improved"]) for item in records)
        posterior = _posterior(successes, len(records), alpha, beta)
        gains = [max(0.0, float(item["gain"])) for item in records]
        gain_mean = (2.0 * global_gain + sum(gains)) / (2.0 + len(gains))
        release_mean = statistics.mean(float(item["releasedFraction"]) for item in records)
        retired = len(records) >= retire_after_zero_improvements and successes == 0
        score = (
            -1_000_000_000.0
            if retired
            else posterior["mean"] * gain_mean
            + exploration_weight * posterior["standardDeviation"] * gain_mean
            - release_penalty * release_mean
        )
        arms.append(
            {
                "family": family,
                "bottleneckKind": bottleneck_kind,
                "moveDepth": move_depth,
                "observations": len(records),
                "improvements": successes,
                "pooledPriorRate": pooled_rate,
                "improvementPosterior": posterior,
                "gainPosteriorMean": gain_mean,
                "meanReleasedFraction": release_mean,
                "retired": retired,
                "priorityScore": score,
            }
        )
    arms.sort(key=lambda item: (item["family"], -item["priorityScore"], item["bottleneckKind"], item["moveDepth"]))
    recommendations = {}
    for arm in arms:
        recommendations.setdefault(arm["family"], arm)
    return {
        "model": {
            "kind": "family-posterior-with-cross-family-empirical-prior",
            "poolStrength": pool_strength,
            "explorationWeight": exploration_weight,
            "releasePenalty": release_penalty,
            "purpose": "rank bounded-neighborhood experiments; never certify feasibility",
        },
        "observationCount": len(observations),
        "recommendations": recommendations,
        "arms": arms,
    }
