from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def proposal_key(
    *,
    incumbent_hash: str,
    neighborhood_signature: str,
    operator: str,
    dispatch_swaps: Iterable[dict[str, Any]] = (),
    seed: int = 0,
) -> str:
    """Identify one deterministic proposal without generalizing beyond it."""

    return _stable_hash(
        {
            "incumbentHash": incumbent_hash,
            "neighborhoodSignature": neighborhood_signature,
            "operator": operator,
            "dispatchSwaps": list(dispatch_swaps),
            "seed": int(seed),
        }
    )


def _reason(candidate: dict[str, Any]) -> str:
    if candidate.get("domainErrors"):
        return "domain-construction-failed"
    if candidate.get("requiredExpansionJobs"):
        return "propagation-not-closed"
    if not candidate.get("domainConstructed", False):
        return "domain-construction-failed"
    return "not-a-strict-improvement"


def extract_oracle_cuts(history_paths: Iterable[str | Path]) -> dict[str, Any]:
    """Extract conservative no-goods from deterministic carrier search histories.

    A cut can skip only the exact proposal on the same incumbent, neighborhood,
    operator and seed. Expansion jobs are advice and are never generalized into
    a global feasibility claim.
    """

    sources: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    observation_count = 0
    accepted_count = 0
    for source_value in history_paths:
        source = Path(source_value).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        sources.append(str(source))
        incumbent_hash = str(payload.get("incumbent", {}).get("scheduleHash", ""))
        seed = int(payload.get("meta", {}).get("seed", 0))
        if not incumbent_hash:
            raise ValueError(f"history lacks incumbent.scheduleHash: {source}")
        for candidate in payload.get("candidates", []):
            observation_count += 1
            if candidate.get("accepted", False):
                accepted_count += 1
                continue
            key = proposal_key(
                incumbent_hash=incumbent_hash,
                neighborhood_signature=str(candidate["neighborhoodSignature"]),
                operator=str(candidate["operator"]),
                dispatch_swaps=candidate.get("dispatchSwaps", ()),
                seed=seed,
            )
            cut_payload = {
                "kind": "exact-no-good",
                "proposalKey": key,
                "incumbentHash": incumbent_hash,
                "neighborhoodSignature": str(candidate["neighborhoodSignature"]),
                "operator": str(candidate["operator"]),
                "seed": seed,
                "reason": _reason(candidate),
                "requiredExpansionJobs": sorted(
                    set(map(int, candidate.get("requiredExpansionJobs", ())))
                ),
                "scheduleHash": candidate.get("scheduleHash"),
                "trueMakespan": candidate.get("trueMakespan"),
                "domainConstructed": bool(candidate.get("domainConstructed", False)),
                "propagationClosed": bool(candidate.get("propagationClosed", False)),
            }
            cut_id = _stable_hash(cut_payload)
            cut_payload["id"] = cut_id
            cut_payload["evidence"] = [
                {
                    "source": str(source),
                    "candidateIndex": int(candidate.get("candidateIndex", -1)),
                }
            ]
            if cut_id in by_id:
                by_id[cut_id]["evidence"].extend(cut_payload["evidence"])
            else:
                by_id[cut_id] = cut_payload

    cuts = sorted(by_id.values(), key=lambda item: (item["proposalKey"], item["id"]))
    return {
        "schemaVersion": SCHEMA_VERSION,
        "policy": {
            "scope": "exact deterministic proposal on the same incumbent",
            "effect": "skip repeated domain-Oracle evaluation",
            "generalization": "none",
            "acceptedCandidatesBecomeCuts": False,
        },
        "sources": sorted(set(sources)),
        "statistics": {
            "observations": observation_count,
            "acceptedObservations": accepted_count,
            "uniqueCuts": len(cuts),
            "propagationCuts": sum(
                item["reason"] == "propagation-not-closed" for item in cuts
            ),
        },
        "cuts": cuts,
    }


def load_oracle_cut_store(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"schemaVersion": SCHEMA_VERSION, "cuts": []}
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if int(payload.get("schemaVersion", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported carrier Oracle-cut schema")
    return payload


def matching_exact_cuts(store: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [
        cut
        for cut in store.get("cuts", [])
        if cut.get("kind") == "exact-no-good" and cut.get("proposalKey") == key
    ]
