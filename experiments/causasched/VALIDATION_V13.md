# V13 validation and limitations

- Manifest: 128 unique IDs and structural fingerprints; 26 retained + 102 selected
  from the old 200 TRAIN snapshot. Every problem file has a verified SHA256.
  Selection uses kind round-robin and ID hash ordering, not performance results.
- All 128 included schedules passed feasibility validation during bank creation.
- Clock unit test: three complete all-128 episodes, each with 200 allocated steps.
- Console logging test: detailed lines go to detail.log, summaries and OOM notices
  remain visible. Per-trajectory data and reward components remain separate JSONL.
- CPU integration: migration from a real 26-instance V12 checkpoint to 128,
  retaining weights/optimizer, explicitly resetting the episode. Two shortened
  episodes exercise all 128 instances, each with two one-step trajectories.
- These are correctness smoke checks, not a 128-instance x 16-trajectory CUDA
  benchmark or evidence that load shaping improves performance. Hardware memory
  and full 200-step run time require server validation.
- This training bank must not be presented as unseen evaluation data.
