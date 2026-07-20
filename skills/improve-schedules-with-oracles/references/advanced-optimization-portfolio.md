# Advanced optimization portfolio

## Purpose

Use this reference when radius-based VNS/ALNS has plateaued or when another scheduling family needs a different improvement mechanism. Preserve the incumbent and keep the same validation and Oracle gates.

## Methods beyond ordinary neighborhoods

### 1. Sequence-preserving compaction

Fix every selected mode and resource-order arc. Recompute the earliest feasible schedule with difference constraints or CP-SAT. This detects avoidable timing slack without changing the dispatch policy. Run it before structural search.

### 2. Shifting bottleneck decomposition

Select the machine or stage with the strongest critical-path contribution, optimize that single-machine subproblem, add its sequence arcs, and recompute the next bottleneck. Use first for JSP and for HFSP/carrier instances dominated by one sink channel. Do not assume that the currently most utilized machine is the causal bottleneck.

### 3. Local branching, fix-and-optimize, and proximity search

Bound Hamming distance from the incumbent in machine assignments and sequence arcs. This searches “at most k structural changes” even when no geometric neighborhood is known. Increase k deterministically and penalize disruption after operational objectives.

### 4. Rolling horizon and relax-and-fix

Freeze the completed prefix, solve an overlapping time window, fix only its stable part, and advance. Use for large suffixes or online rescheduling. Keep overlap consistency and revalidate any cross-window resource or route dependency.

### 5. Logic-based Benders decomposition

Split the problem into:

- master: job order, eligible machine, stage, lane, or vehicle assignment;
- subproblem: exact timing, setup, route, collision, or simulator feasibility.

When the subproblem fails, return a no-good, precedence, resource-conflict, time-window, or path-incompatibility cut to the master. Cache cuts by normalized context. This is especially suitable when the master is easy for CP-SAT but the domain Oracle is expensive.

For the carrier, let the master choose tractor/preparation/catapult bindings and launch order. Let trajectory replay check real path duration, vehicle continuity, and collision. Convert repeated Oracle failures into reusable cuts rather than merely enlarging a neighborhood.

### 6. Lagrangian or dual-price guidance

Relax capacity, precedence, or transport coupling constraints and interpret their multipliers as congestion prices. Use prices to rank which resource-time arcs or assignments should be released. Do not accept the relaxed solution; repair it and run the normal validators.

### 7. Column generation or pattern generation

Represent a legal machine sequence, batch, vehicle route, or stage pattern as a column. Solve a restricted master and generate negative-reduced-cost patterns. Use only when the instance has repeated route/sequence structure and a tractable pricing subproblem.

### 8. Critical-path decision diagrams or deterministic beam repair

Encode partial sequences as states and keep a fixed number of states under stable dominance rules. This provides a reproducible middle ground between enumeration and heuristics. Use for small critical blocks or when CP-SAT repeatedly spends effort rediscovering the same partial orders.

### 9. Elite path relinking

Require two generically and domain-validated schedules. Move from one toward the other by adopting one ranked machine choice or sequence arc at a time, repairing every intermediate schedule. This exploits structural diversity without restarting from scratch.

### 10. Robust and scenario-based polishing

After nominal feasibility, replay a fixed scenario set for processing, travel, setup, or collision delays. Optimize lexicographically:

1. hard feasibility in every required scenario;
2. service/tardiness target;
3. nominal objective;
4. tail degradation such as worst-case or CVaR;
5. change cost.

Do not call a schedule robust merely because it has a smaller nominal makespan.

### 11. Learning and statistical guidance

Use Bayesian posteriors, contextual bandits, ranking models, or graph models to select the next method configuration. Features may include bottleneck kind, criticality, slack, route family, release fraction, propagation expansion, previous gain, and Oracle cost. The learned layer may rank experiments; it may not fabricate feasible start times or bypass repair.

## Project-specific new mechanisms

### Causal-closure repair

Trace backward from the objective bottleneck through precedence, resource, setup, route, continuity, time-window, and collision-propagation arcs. Release the smallest evidence-backed propagation closure and freeze its complement. This generalizes the current O7 gap idea without depending on an operation number or radius.

### Oracle-cut memory

Promote repeated domain failures into reusable constraints:

- assignment pair cannot share a route/time window;
- one vehicle/job order requires a minimum separation;
- one lane binding forces another linked choice;
- one path conflict requires a precedence disjunction;
- one outside-mode propagation identifies the missing closure edge.

Key each cut by problem, geometry, route library, and Oracle version. Expire it when any of those hashes change.

The carrier implementation begins with an intentionally narrower cut: the tuple of incumbent hash, neighborhood signature, repair operator, dispatch swaps, and seed. It only prevents an identical deterministic replay. A propagation failure may additionally return `requiredExpansionJobs`; use those jobs to construct a new closure, not as proof that every related neighborhood is infeasible. Broader precedence, separation, route-incompatibility, or machine-binding cuts require direct evidence from the trajectory/collision Oracle and version hashes for their validity domain.

### Counterfactual bottleneck influence

For each upstream arc, estimate the earliest theoretical shift of the target bottleneck if that arc were relaxed alone. Rank operations by estimated objective improvement divided by expected repair and Oracle cost. Use this score to seed causal closure, local branching, or Benders cuts.

### Two-timescale controller

Keep structural decisions slow and conservative while timing is repaired frequently:

- inner loop: compact timings and insert legal waits with fixed modes/sequences;
- outer loop: change a small number of machine, route, or sequence decisions;
- Oracle loop: validate only generically feasible, nonduplicate outer-loop candidates.

This avoids spending an Oracle call on candidates whose only problem is deterministic timing slack.

## Recommended staged combination

```text
incumbent audit and compaction
  → family-specific bottleneck diagnosis
  → causal closure or local branching exact repair
  → generic validator
  → domain Oracle and reusable cut extraction
  → strict acceptance and restart

plateau A
  → shifting bottleneck or assignment/sequence alternation
  → rolling horizon / relax-and-fix

plateau B
  → logic-based Benders with Oracle cuts
  → deterministic ALNS or elite path relinking

enough evidence
  → Bayesian allocation of method/operator/budget
  → fixed scenario robustness polishing
```

## Current carrier priority

1. Run sequence-preserving compaction on `627.8` to prove that remaining gaps are structural.
2. Implement causal influence scores around the largest remaining O7 gaps.
3. Add local branching over tractor, preparation, catapult, and O7 order changes with a strict change budget.
4. Extract reusable cuts from propagation and collision failures.
5. Introduce a logic-based Benders loop with the existing trajectory environment as subproblem Oracle.
6. Test path relinking between structurally different validated elites.
7. Add fixed delay scenarios and accept only improvements that preserve a declared robustness floor.

## Primary sources

- Google OR-Tools job-shop scheduling model: https://developers.google.com/optimization/scheduling/job_shop
- PyJobShop supported scheduling constraints: https://pyjobshop.org/stable/
- Adams, Balas, and Zawack, *The Shifting Bottleneck Procedure for Job Shop Scheduling*: https://doi.org/10.1287/mnsc.34.3.391
- Fischetti and Lodi, *Local Branching*: https://publications.polymtl.ca/25914/
- Naderi and Roshanaei, *Critical-Path-Search Logic-Based Benders Decomposition Approaches for Flexible Job Shop Scheduling*: https://doi.org/10.1287/ijoo.2021.0056
- Peng, Lu, and Cheng, *A Tabu Search/Path Relinking Algorithm to Solve the Job Shop Scheduling Problem*: https://arxiv.org/abs/1402.5613
