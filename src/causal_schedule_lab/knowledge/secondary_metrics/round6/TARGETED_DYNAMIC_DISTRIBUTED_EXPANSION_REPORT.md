# Dynamic and Distributed Incremental Expansion Report

## Scope

This is an additive-only package. It does not rewrite the existing 67 canonical metrics, does not create separate catalogs such as `dynamic_FJSP`, and does not promote any entity beyond `proposed`.

The added knowledge remains organized through the existing orthogonal axes:

- `problem_family`: JSP / FSP / FJSP / HFSP;
- `variant_head`: dynamic, rescheduling, distributed, multi_factory, interfactory_transport;
- `mechanism`: waiting, workload, transport, event_response, disruption_propagation;
- `decision`: factory_assignment, interfactory_transfer, rescheduling_trigger, rescheduling_scope.

A project described as dynamic distributed FJSP is therefore a query profile formed by the union of these views, not a duplicated candidate store.

## Additions

- New literature records: 16
- New metric candidates: 14
- New diagnostic candidates: 6
- New candidate-view memberships: 257
- New retrieval bundles: 5

## New distributed mechanisms

1. `interfactory_transfer_wait_time`
2. `critical_interfactory_transport_time`
3. `factory_workload_cv`
4. `factory_assignment_concentration_hhi`
5. `factory_release_delay_total`
6. `cross_factory_synchronization_wait`
7. `interfactory_transfer_count`

## New dynamic mechanisms and diagnostics

1. `new_job_response_delay`
2. `disruption_detection_delay`
3. `rescheduling_trigger_latency`
4. `rescheduling_computation_delay`
5. `disruption_propagation_ratio`
6. `rescheduling_scope_ratio`
7. `machine_breakdown_capacity_loss`

## Evidence limitations

All new metric formulas are marked `project_canonical_formula` and `project_formula_defined`. This package does not claim source-level formula verification. Publisher abstracts and bibliographic records were used to confirm problem semantics, decisions, and disturbance types. Full-text equation and page-level auditing remains a separate evidence task.

No candidate is marked active, validated, causal, variation_verified, or intervention_supported.
