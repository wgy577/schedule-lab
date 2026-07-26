# Secondary Metric Knowledge

This directory stores proposed metric-discovery knowledge. It is not runtime
project evidence and does not activate candidates in the optimizer.  The catalog
is now consumed by the optional constraint-impact Critic through deterministic,
token-bounded retrieval; selection still remains `pending_human_review`.

## Layout

- `round3/`: immutable source catalogs used by the Round 4 materializer;
- `round4/`: capability groups, 623 relationship recommendations, 55
  computability contracts, 21 design-reference cases, and the materialization
  audit;
- `round5/`: 12 metric, 5 diagnostic and 171 relationship additions for batch,
  lot-streaming, WIP, multi-resource, charging and assembly contexts;
- `round6/`: 14 metric, 6 diagnostic and 257 relationship additions for dynamic
  and distributed scheduling;
- `compatibility/`: four non-equivalent canonical additions plus an audited map
  proving that all 11 original active choices are represented once;
- `../../secondary_metric_knowledge.py`: read-only validated loader;
- `../../../../../scripts/build_round4_knowledge.py`: deterministic rebuild.

## Evidence boundary

- All canonical candidates remain `proposed`.
- Reference cases are design labels, not independent truth.
- `currently_computable` remains unknown until a real project's IR is checked.
- Deterministic recall uses family as a hard gate, explicit variants as activation
  gates, and mechanism/decision/evidence views for ranking. Missing IR fields are
  reported as `needs_adapter_or_evidence`, never silently inferred.
- At most 12 compact options enter one Critic prompt; returned candidate IDs are
  checked against that exact whitelist.
- The optimizer still does not consume proposed candidates automatically.
