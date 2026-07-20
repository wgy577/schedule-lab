# Agentic RL and joint schedule-trajectory optimization

## Scope

Use this design only after an incumbent, deterministic validator, domain Oracle, rollback path, and versioned evidence store exist. The learning layer allocates experiments; it does not replace feasibility checking.

## Current carrier evidence

The legacy carrier implementation contains 220 fixed MAT paths:

- 100 initial tractor movements;
- 60 coupled tractor-aircraft towing movements;
- 60 later transfer movements.

Each exact job/machine/phase binding has one route, but every job has 3–5 existing alternatives across preparation/tractor bindings. Scheduling selects these paths only indirectly through state-dependent machine and spot choices, then the collision layer inserts delays in 0.1-second increments. This is collision-aware replay with opaque route binding, not an explicit joint route-column master. Run `carrier-route-catalog` to reproduce the inventory and version every source path.

## Where Agentic RL belongs

Treat Agentic RL as a shielded experiment controller:

1. select a diagnosed bottleneck or collision cluster;
2. select one compatible exact/heuristic method;
3. select a bounded release set, route columns, and solver budget;
4. decide whether a failed proposal needs a Cut, a causal expansion, or retirement;
5. in online rescheduling, choose which local suffix to repair after a disruption.

Do not let the policy emit unconstrained start times, bypass machine/route masks, declare collision freedom, or replace the incumbent without validation. DR-ALNS demonstrates that RL can control destroy/repair operators, parameters, and acceptance, while learning-guided LNS demonstrates learned neighborhood selection; both still rely on a solver or repair procedure for the actual combinatorial decisions.

## Encoder decision

Do not train a neural encoder merely because the problem is graph-shaped. Start with explicit, inspectable diagnostic features and a deterministic contextual posterior while data are sparse.

Recommend a learned encoder only after at least 500 deduplicated post-Oracle experiments and a held-out evaluation split. The first useful model is a heterogeneous graph encoder:

- schedule nodes: operation, job, machine/stage, tractor/vehicle, lane/catapult;
- schedule edges: precedence, eligible mode, incumbent assignment, resource order, shared capacity, binding choice;
- route nodes: waypoint or route segment, vehicle state, conflict event;
- route edges: spatial adjacency, route membership, time overlap, collision conflict;
- fusion: cross-attention from transport operations and assigned vehicles to space-time route segments.

This follows the useful part of graph-based dispatching research: encode the disjunctive scheduling graph, but keep the exact solver and Oracle outside the network. If the validated data never reach the threshold, the explicit-feature controller remains the production path.

## Learning stages

| Stage | Controller | Admission gate |
|---|---|---|
| 0 | Deterministic posterior/contextual bandit | Default |
| 1 | Offline imitation of exact-solver and accepted-search choices | 500 deduplicated validated experiments |
| 2 | Conservative offline actor-critic for method/neighborhood control | Held-out feasibility, regret, and calibration gates pass |
| 3 | Shielded online fine-tuning in simulator | Deterministic repair, Oracle, and rollback all active |

Use post-Oracle reward only. Rank lexicographically by hard feasibility, operational objective, makespan, route/collision delay, and disruption. Divide gain by deterministic solver plus Oracle cost when allocating experiments. Apply a hard negative reward to invalid proposals, but retain the incumbent rather than learning from an executed invalid schedule.

## Joint master-subproblem model

Use logic-based Benders decomposition (LBBD), because scheduling choices are compact while conflict-free movement is expensive.

### Master problem

Use CP-SAT or MILP to choose:

- operation-machine and tractor assignment;
- machine, preparation, catapult, and global-launch orders;
- tractor continuity;
- transport release/deadline windows;
- one candidate route column for every movement;
- a bounded Hamming distance from the incumbent during improvement.

The master must contain valid travel-time lower bounds. It supplies a global lower bound only for the declared finite model.

### Trajectory subproblem

Given the master decisions, solve conflict-free timed movement with one of:

- Conflict-Based Search for small discrete optimal multi-agent pathfinding cases;
- a time-expanded CP-SAT/MILP network for exact finite space-time routing;
- safe-interval/shortest-path pricing to generate additional route columns;
- continuous smoothing with speed, acceleration, curvature, and separation constraints after the discrete plan is safe.

Return either timed trajectories and a feasible upper bound or a minimal conflict explanation. Never return only `infeasible` when the conflicting assignment, route, shared segment, and time windows can be identified.

### Cuts

Feed explanations back as:

- route-assignment no-goods;
- pairwise conflict precedence disjunctions;
- minimum temporal separation constraints;
- shortest feasible travel-time lower bounds;
- tractor continuity/reachability cuts;
- causal-closure expansion requirements.

Cache every Cut under problem, geometry, route-library, Oracle, time-resolution, vehicle-geometry, and incumbent-context hashes.

## Route-column requirement

The normalized catalog contains 60 job/phase routing groups and all 60 have 3–5 alternatives across machine/spot bindings. Expose those existing columns and their state-dependent compatibility first. In particular, an O1 path depends on the tractor depot for its first task and on the previous aircraft's preparation spot thereafter. Do not model that sequence-dependent travel as a fixed operation duration.

Only after the existing alternatives are explicit should the system generate additional same-binding spatial routes with deterministic k-shortest/safe-interval search, visibility or lane graphs, and different conflict-precedence patterns. Keep every original MAT route as a known fallback.

## First fixed-route exact experiment

The first finite-grid CP-SAT subproblem uses half-open 0.1-second occupancy, fixed MAT geometry, fixed incumbent bindings, tractor O1–O4 locks, catapult O5–O8 locks, and the 10-second launch-channel cooldown. It finds a mathematical optimum of 619.5 seconds for that abstraction. The target-order domain replay is deterministic but returns 803.9 seconds and 40 binding changes because several requested bindings are unreachable under the proposed dispatch order. Therefore the candidate is rejected and converted into a context-exact reachability no-good Cut affecting jobs 3, 4, 9–13, and 17 in one-based numbering. The validated 627.8-second incumbent remains unchanged.

This experiment is a required LBBD behavior, not a failed implementation: an exact master optimum is provisional until state-dependent domain reachability and collision replay pass.

## Exactness boundary

The system may claim a global optimum only when:

1. the master is solved exactly;
2. every trajectory subproblem is solved exactly;
3. all valid conflict explanations needed for convergence are added;
4. time, space, geometry, kinematics, and objectives are a declared finite model;
5. the feasible upper bound equals the master lower bound.

At zero gap, report “globally optimal for the declared discretized joint model.” Otherwise report the best validated solution, lower bound, and certified gap. Do not claim global optimality for unmodeled continuous physics, uncertain motion, or arbitrary real-world deck operations.

## Reproduction

From the comparison repository root:

```bash
schedule_lab/.venv/bin/python schedule_lab/run_schedule_lab.py \
  adaptive-improvement-benchmark --baseline-rule lpt \
  --output schedule_lab/outputs/adaptive_multifamily_improvement_benchmark.json

schedule_lab/.venv/bin/python schedule_lab/run_schedule_lab.py \
  carrier-route-catalog --legacy-root . --max-points 24 \
  --output schedule_lab/outputs/carrier_route_catalog.json

schedule_lab/.venv/bin/python schedule_lab/run_schedule_lab.py \
  carrier-joint-schedule-trajectory-plan --legacy-root . \
  --schedule schedule_lab/outputs/carrier_alns_best_iter3_gap6_closed_630_5.json \
  --validated-experiment-count 62 \
  --output schedule_lab/outputs/carrier_joint_schedule_trajectory_plan.json
```

## Primary sources

- Zhang et al., *Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning*: https://papers.nips.cc/paper_files/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html
- Reijnen et al., *An exact decomposition-based approach to the conflict-free-transportation-constrained flexible job-shop scheduling problem*: https://doi.org/10.1016/j.cor.2025.107342
- Sharon et al., *Conflict-based search for optimal multi-agent pathfinding*: https://doi.org/10.1016/j.artint.2014.11.006
- Mihoubi et al., *Deep reinforcement learning-based adaptive large neighborhood search*: https://doi.org/10.1609/icaps.v34i1.31507
- Song et al., *Learning large neighborhood search policy for integer programming*: https://arxiv.org/abs/2111.03466
