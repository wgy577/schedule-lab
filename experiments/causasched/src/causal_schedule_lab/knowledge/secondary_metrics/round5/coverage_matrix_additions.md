# Coverage Matrix Additions — Targeted Journal Expansion

Only incremental views and candidates are listed. Existing 55 entities are not repeated.

| New/expanded view | JSP | FSP | FJSP | HFSP | Added canonical candidates |
|---|---:|---:|---:|---:|---|
| batch_processing | ✓ | ✓ | ✓ | ✓ | batch_formation_wait_time; batch_capacity_slack_ratio |
| lot_streaming |  | ✓ | ✓ | ✓ | sublot_merge_wait_time |
| reentrant |  | ✓ | ✓ | ✓ | work_in_process_area; reentrant_revisit_queue_wait |
| worker transfer/workload | ✓ | ✓ | ✓ | ✓ | worker_transfer_delay_total; worker_workload_cv |
| fixture/tool setup | ✓ |  | ✓ |  | fixture_loading_unloading_time_total |
| AGV charging | ✓ |  | ✓ | ✓ | agv_charging_wait_time |
| assembly |  |  | ✓ | ✓ | assembly_component_sync_wait |
| stochastic feasibility | ✓ | ✓ | ✓ | ✓ | scenario_infeasibility_rate |
| rescheduling policy | ✓ | ✓ | ✓ | ✓ | rescheduling_event_count |

## Evidence note

Only `fixture_loading_unloading_time_total` is added at `source_definition_verified`. The other new formulas are explicitly project-canonical definitions grounded in source problem/event semantics. None is `code_backed`, `variation_verified`, or `intervention_supported`.
