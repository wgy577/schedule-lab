# Unresolved Questions Additions

1. **Batch formation versus productive waiting** — fuller batches may reduce the number of cycles while increasing formation waiting. Direction must be tested by batch capacity and load regime.
2. **Lot-streaming merge semantics** — some lot-splitting models allow sublots to remain independent; `sublot_merge_wait_time` is not applicable without a real merge/joint gate.
3. **WIP definition** — decide whether jobs count from release, shop admission, first processing start, or physical arrival. These definitions are not interchangeable.
4. **Reentrant visit identity** — the IR needs stable `(job, operation, visit_index, stage)` identity before revisit-specific waiting can be computed.
5. **Worker transfer attribution** — worker travel time, worker occupancy and operation waiting must not be summed without an overlap policy.
6. **Fixture setup equivalence** — the exact source formula applies to fixture loading/unloading; it must not automatically replace generic sequence-dependent setup.
7. **AGV charging counterfactual** — charging-attributed waiting requires an event-cause label or a defensible counterfactual vehicle-ready time.
8. **Assembly readiness** — distinguish component processing completion, delivery completion, buffer admission and inspection release.
9. **Scenario infeasibility** — solver timeout, decoder failure and truly infeasible realized schedules require separate result codes.
10. **Rescheduling event semantics** — scheduled periodic epochs, event-triggered checks and actual schedule revisions must be represented separately.

All additions remain `proposed`; no causal or active claims are made.
