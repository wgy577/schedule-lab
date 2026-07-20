# Carrier domain and comparison video

## Stable schedule artifacts

- Greedy incumbent: `schedule_lab/outputs/carrier_greedy_baseline_675_5.json`
  - true makespan: `675.5`
  - normalized schedule SHA-256 starts with `4100e0d42a977fdf`
- Current validated candidate: `schedule_lab/outputs/carrier_alns_best_iter3_gap6_closed_630_5.json`
  - true makespan: `627.8`
  - normalized schedule SHA-256 starts with `e8e67033c5dd257c`
- The `637.5` artifact is a validated intermediate candidate and may be the incumbent for the next incremental comparison; it is not the original greedy baseline.

The carrier adapter must preserve tractor, preparation-spot, catapult/lane, and path identity. Different machines at the same operation can imply different MAT trajectories and time windows. Require the legacy trajectory/collision replay after generic validation.

## Carrier diagnostic specialization

Use O7 (`op == 6`) on the single global launch channel only as the first diagnostic surface. Rank gaps, then trace their upstream causes through O1–O6 and the downstream O8 suffix. Do not accept merely because O7 appears denser.

## Video workflow

Run from the repository root:

```bash
python3 schedule_lab/workflows/video/render_schedule_comparison.py
```

The fixed contract is 1920×816, 30 fps, 60 seconds, H.264/yuv420p. Both 960×816 panels use the same layout, shared simulated clock, and Gantt x-axis maximum. The shorter schedule finishes first and holds its completed state until the longer schedule ends; never normalize each makespan independently to 60 seconds. The renderer hashes both schedules and refuses identical inputs. It writes an adjacent manifest containing input paths, schedule hashes, makespans, machine-binding differences, timing differences, shared-clock parameters, and encoding parameters.
