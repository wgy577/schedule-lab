"""Export all trained active networks, without optimizer/training frontiers."""
import argparse
from pathlib import Path
import torch
import train_e2e_single as train
from causal_schedule_lab.m3.end_to_end import modules, reset_rl_actors

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--runtime',type=Path,default=train.ROOT/'inference_assets/runtime.pt')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    train.base.initialize(a.runtime,6)
    r=train.base.RUNTIME
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck['format']!='causasched-e2e-single-v1': raise ValueError('Wrong checkpoint format')
    if ck['config']['initialization']=='sft':
        reset_rl_actors(r)
        r['policy'].m2.set_progress(ck['cycle'],ck['config']['cycles'])
        r['policy'].m3.set_progress(ck['cycle'],ck['config']['cycles'])
    hops=int(ck['weights']['target']['probability_trace.hop_embedding.weight'].shape[0])
    train.configure_trace_depth(r,hops)
    for k,m in modules(r).items():
        m.load_state_dict(ck['weights'][k],strict=True)
        m.eval().requires_grad_(False)
    train.restore_progress(r,ck['weights'])
    r['model_b5'].e2e_capture=False
    r['meta'].update(optimizer_update=ck['cycle'],training=ck['config'].get('training_variant','e2e'),
                     trace_hops=hops, pair_mode='bounded-relations',
                     train_ids=sorted(ck['arena']),regression_weight=ck['config']['regression_weight'],
                     recommended_inference='single+pair; --single-only is an optional ablation')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    train.atomic_save(r,a.output)
    train.base.write_json(a.output.with_suffix('.json'),r['meta'])
    print(a.output)

if __name__=='__main__': main()
