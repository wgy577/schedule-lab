"""One-time trusted-checkpoint export; never called by the search loop."""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import torch
from causal_schedule_lab.m3 import config as C
from causal_schedule_lab.m3 import hierarchical_residual as HR
import run_m3_canonical_training as RUN


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'inference_assets/runtime.pt')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    start = time.perf_counter()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    env = RUN.build_env(SimpleNamespace(quick=False, ckpt=str(ROOT / 'checkpoints/b5_1_shared.pt')))
    replay = RUN.build_replay_env(env)
    phase1 = torch.load(C.SFT_CKPT, map_location='cpu', weights_only=False)
    scorer = phase1['state']['p1']['scorer_mem']
    caps_data = json.loads((ROOT / 'outputs/t2d/train_residual_caps.json').read_text())
    caps = HR.ResidualCaps(**{f.name: caps_data[f.name] for f in fields(HR.ResidualCaps)
                             if f.name in caps_data})
    caps.validate()
    policy = RUN._t2d_policy('P2', RUN._load_r6_parent(None), caps, env)
    policy.m2.enable_probability_trace()
    policy.load_snapshot(ck['state']['policy_snapshot'])
    modules = [policy.m2, policy.m3, scorer, env['model_b5'], env['single_head'], env['direct_head']]
    for module in modules:
        module.eval()
        module.requires_grad_(False)
    # Do not carry optimizer, cached training states, or RL frontier solutions.
    runtime = {k: env[k] for k in ('executor', 'model_b5', 'single_head', 'direct_head')}
    runtime.update(policy=policy, scorer=scorer, memory=replay['progmem'])
    runtime['meta'] = {
        'format': 'causasched-fast-runtime-v1',
        'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'optimizer_update': ck['meta']['optimizer_update'],
        'train_ids': [row[0] for row in ck['state']['frontier_state']['active_definitions']],
        'memory_source': 'original TRAIN replay only; no evaluated-bank outcomes',
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix('.tmp')
    torch.save(runtime, tmp)
    tmp.replace(a.output)
    a.output.with_suffix('.json').write_text(json.dumps(runtime['meta'], indent=2))
    print(json.dumps({**runtime['meta'], 'seconds': time.perf_counter()-start,
                      'output': str(a.output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
