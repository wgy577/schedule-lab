# Family-aware Agent strategy routing

## Contract

The Agent diagnoses and ranks methods. It does not fabricate assignments, start times, routes, or feasibility. The deterministic repair engine constructs candidates; the generic validator and optional domain Oracle decide acceptance.

```text
canonical problem + incumbent
  → evidence-backed diagnosis
  → family strategy pack
  → deterministic ranked method route
  → bounded repair
  → generic validation
  → optional domain Oracle
  → strict acceptance or exact incumbent fallback
```

## Diagnostic record

Record family, problem scale, job/operation/resource counts, flexible-operation fraction, dominant bottleneck, bottleneck counts and concentration, plateau count, Oracle failure count, validated evidence count, and validated elite count. Stable inputs must produce byte-equivalent routing output.

## Strategy packs

- JSP: critical resource blocks, shifting bottleneck, critical-block VNS, local branching, and bounded decision diagrams.
- FSP: sink-stage gaps, permutation insertion/NEH proposals, shifting bottleneck, downstream causal repair, and rolling horizon.
- FJSP: flexible-resource imbalance, assignment-sequence alternation, dual-price release, local branching, and Logic-Based Benders when repeated Oracle failures exist.
- HFSP: stage starvation/blocking, sink gaps, parallel-machine reassignment, shifting bottleneck, causal closure, and decomposition.
- Carrier/domain hybrid: use the closest HFSP/FJSP structure for proposals, but retain route, vehicle-continuity, time-window, collision, and real-duration authority in the original Oracle.

## Routing rules

1. Run sequence-preserving compaction before structural changes.
2. Score family-primary methods, then add evidence from the dominant bottleneck.
3. Add exact bounded repair before plateau escape.
4. Enable deterministic ALNS only after the declared no-improvement count.
5. Enable Oracle-cut Benders only when the family supports assignment/decomposition and repeated Oracle failures provide evidence.
6. Enable Bayesian allocation only after at least 12 deduplicated validated experiments.
7. Enable path relinking only with at least two structurally distinct validated elites.
8. Filter every ranked method by its declared compatible families.

## Cost-aware shortlist

Rank structural methods by diagnostic suitability divided by a versioned estimated-cost unit. Cost is used only to order compatible experiments; it never changes feasibility or acceptance. Use three profiles:

- `fast`: at most 3 methods, 2 bottlenecks, 4 candidates, radius 2, 0.25 deterministic repair units, and 1 Oracle survivor;
- `balanced`: at most 4 methods, 3 bottlenecks, 8 candidates, radii 2–3, and 2 Oracle survivors;
- `thorough`: at most 6 methods, 4 bottlenecks, 16 candidates, radii 2–4, and 4 Oracle survivors.

Apply cheap gates in this order: duplicate/Cut cache, lower-bound or potential-improvement check, canonical feasibility and frozen-region check, objective comparison, then domain Oracle. Stop the fast profile on the first strict validated improvement. If none exists, return the exact incumbent and optionally escalate one profile; do not silently widen into a global reschedule.

## Meaning of “enhancement”

The controller guarantees monotonic acceptance, not universal strict improvement. A family is enhanced when it has multiple compatible diagnostic and repair lanes, deterministic routing, full validation, and exact incumbent fallback. No candidate may replace the incumbent unless it is feasible and lexicographically better. Some instances are already optimal or have no improvement inside the configured budget.

## Regression matrix

Require deterministic tests for JSP, FSP, FJSP, and HFSP that verify:

- expected family-primary methods are selected;
- every ranked strategy declares compatibility with that family;
- plateau and evidence gates activate only at their thresholds;
- the fallback retains the incumbent;
- frozen-region and feasibility tests still pass;
- domain-required candidates remain provisional until Oracle replay.
