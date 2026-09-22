# Targeted Evidence and Metric Expansion — Incremental Report

**Generated:** 2026-07-26

## Scope

This package is additive. It does not reproduce or replace the 55 Round-2 canonical metric candidates or 25 diagnostics. It adds targeted evidence and candidates identified from batching, lot streaming, reentrant/WIP, dual-resource, AGV charging, assembly, stochastic and rescheduling literature.

## Increment counts

- New literature records: **18**
- New canonical metric/feature/objective/diagnostic candidates: **12**
- New specialized diagnostics: **5**
- New candidate–view memberships: **171**
- New retrieval bundles: **8**

## Main additions

1. Batch formation waiting and unused batch capacity.
2. Lot-streaming sublot synchronization.
3. Reentrant revisit queues and WIP congestion.
4. Worker transfer and worker workload imbalance.
5. Source-defined fixture loading/unloading decomposition.
6. Charging-attributed AGV waiting.
7. Component synchronization before assembly.
8. Scenario infeasibility and rescheduling frequency with non-secondary roles.

## Guardrails

- Every new canonical entity has a unique ID and remains `proposed`.
- Existing candidate bodies are not copied. New relations to existing entities are stored separately.
- Project-normalized formulas are labeled `project_formula_defined`; they are not presented as source equations.
- No claim of `code_backed`, `variation_verified`, `intervention_supported`, validation, or causality is made.
- Abstract-only sources are used for taxonomy and applicability, not formula verification.
