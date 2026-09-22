"""Numerical forward/backward and timing check, including real pair candidates.

Uses synthetic test advantages ONLY for numerical gradient tests (not training).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import train_e2e_single as t
from causal_schedule_lab.m3.end_to_end import recompute,configure,clipped_loss
from causal_schedule_lab.m3.e2e_batch import replay_batch,loss_batch
from causal_schedule_lab.m3.e2e_encoder import encode_many

def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cpu'); p.add_argument('--repeat',type=int,default=1)
    a=p.parse_args(); torch.manual_seed(9162026)
    t.worker_init(ROOT/'inference_assets/runtime.pt',6,0.,'sft',9162026)
    r=t.base.RUNTIME; records=[]
    bank=ROOT/'data/train128'
    files={row['instance_id']:row['file'] for row in json.loads((bank/'protocol.json').read_text())['entries']}
    for i,iid in enumerate(('Behnke4','Behnke7','BrandimarteMk6')):
        x=json.loads((bank/files[iid]).read_text())
        problem=t.Problem.model_validate(x['problem']); schedule=t.solve_dispatching(problem,rule='earliest_finish')
        cache=t.FastAnalyzeCache(r['model_b5'],r['single_head'],r['direct_head'],max_entries=3)
        ast=cache.ast(problem,schedule,iid)
        _,metas,_=cache.proposals(problem,schedule,iid)
        assert sum(m['kind']=='single' for m in metas)==len(ast['pool'])
        paired_atoms=[idx for m in metas if m['kind']=='pair' for idx in (m['i'],m['j'])]
        assert len(paired_atoms)==len(set(paired_atoms)), 'An atom has multiple partners'
        with torch.no_grad():
            tr=t.JG.collect_trajectory_r14(r['policy'],r['scorer'],r['executor'],r['model_b5'],
                r['single_head'],r['direct_head'],problem,schedule,schedule.makespan,iid,900000000,
                copy.deepcopy(r['memory']),9162026+i,i,horizon=2,step_offset=0,stop_on_negative=False,
                action_space='policy_sampled',analyze_cache=cache,allow_policy_stop=False,
                feasible_fallback=True,anchor_trajectories=0)
        records.extend(tr['steps'])
    records=(records*a.repeat)[:32]
    configure(r,True)
    for m in t.modules(r).values(): m.to(a.device)
    named=t.parameters(r)
    for i,s in enumerate(records):
        s['adv2']=.7 if i%2 else -.4; s['adv3']=.3 if i%2 else -.8
        s['_w2']=s['_w3']=1/len(records)
    def sync():
        if a.device=='cuda': torch.cuda.synchronize()
    r['model_b5'].e2e_fast_encoder=False
    sync(); start=time.perf_counter(); refs=[]
    for s in records:
        p2,p3,_,aux=recompute(r,s,a.device)
        l2=sum(clipped_loss(lp,old,s['adv2']) for lp,old in aux['draws'])/len(aux['draws'])
        loss=l2*s['_w2']+clipped_loss(p3,s['logp_old'],s['adv3'])*s['_w3']
        loss.backward(); refs.append((p2.detach(),p3.detach(),aux['F'].detach()))
    sync(); serial=time.perf_counter()-start
    grads={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in named}
    for _,p in named: p.grad=None
    r['model_b5'].e2e_fast_encoder=True
    sync(); start=time.perf_counter(); out=replay_batch(r,records,a.device); loss_batch(out,records).backward(); sync()
    batched=time.perf_counter()-start
    errs=dict(lp2=0.,lp3=0.,features=0.,gradient=0.)
    for i,(p2,p3,f) in enumerate(refs):
        for name,x,y in [('lp2',out['lp2'][i],p2),('lp3',out['lp3'][i],p3),('features',out['features'][i],f)]:
            errs[name]=max(errs[name],float((x-y).detach().abs().max()))
            torch.testing.assert_close(x,y,atol=3e-4,rtol=3e-4)
    for n,p in named:
        before=grads[n]; after=None if p.grad is None else p.grad.detach().cpu()
        if before is None and after is None: continue
        if before is None: before=torch.zeros_like(after)
        if after is None: after=torch.zeros_like(before)
        errs['gradient']=max(errs['gradient'],float((before-after).abs().max()))
        torch.testing.assert_close(before,after,atol=5e-4,rtol=5e-3,msg=lambda msg:n+' '+msg)
    summary=dict(verdict='PASS',device=a.device,decisions=len(records),
        pair_candidates=sum(int(s['pair_mask'].sum()) for s in records),errors=errs,
        serial_seconds=serial,batched_seconds=batched,speedup=serial/batched,
        gradient_modules=sorted({n.split('.')[0] for n,p in named if p.grad is not None and float(p.grad.norm())>0}))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
