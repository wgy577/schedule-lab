import itertools
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import torch
from causal_schedule_lab.m3.step20_training import batch_logp, ordered_sample, build_collector


def main():
    p = torch.tensor([[.2,.3,.5]], dtype=torch.float64, requires_grad=True)
    records = [dict(M=3,a=2,m3_candidate_draws=[2,0])]
    lp = batch_logp(p, records)
    torch.testing.assert_close(lp.exp(), torch.tensor([.5*.2/.5],dtype=p.dtype))
    lp.sum().backward()
    assert torch.isfinite(p.grad).all()
    # Every possible ordered set sums to probability one, including K=N.
    for k in (1,2,3):
        rows = [dict(M=3,a=q[0],m3_candidate_draws=list(q))
                for q in itertools.permutations(range(3),k)]
        total = batch_logp(p.detach().expand(len(rows),-1),rows).exp().sum()
        torch.testing.assert_close(total, torch.tensor(1.,dtype=p.dtype))
    # K=1 equals the original categorical likelihood; heterogeneous padded batches.
    probs = torch.tensor([[.2,.3,.5,0],[.4,.6,0,0]],dtype=torch.float64)
    rows = [dict(M=3,a=2,m3_candidate_draws=[2,0]),dict(M=2,a=0)]
    out = batch_logp(probs,rows)
    torch.testing.assert_close(out.exp(), torch.tensor([.2,.4],dtype=probs.dtype))
    for seed in range(100):
        draws = ordered_sample([.2,.3,.5,0],20,random.Random(seed))
        assert len(draws)==3 and len(set(draws))==3 and 3 not in draws
    # Check gradients against an explicit sequential reference for variable K.
    logits = torch.randn(2,5,dtype=torch.float64,requires_grad=True)
    probs = .95*logits.softmax(-1)+.01
    rows = [dict(M=5,a=1,m3_candidate_draws=[1,3,2]),
            dict(M=5,a=4,m3_candidate_draws=[4,2])]
    fast = batch_logp(probs,rows).sum()
    slow = 0
    for i,s in enumerate(rows):
        remain=list(range(5))
        for j in s['m3_candidate_draws']:
            slow=slow+probs[i,j].log()-probs[i,remain].sum().log()
            remain.remove(j)
    torch.testing.assert_close(fast,slow)
    torch.testing.assert_close(torch.autograd.grad(fast,logits,retain_graph=True)[0],
                               torch.autograd.grad(slow,logits)[0])
    collector=build_collector(20)
    compare=collector.__globals__['_compare_train']
    from causal_schedule_lab.m3 import joint_grpo as JG
    from types import SimpleNamespace
    saved={name:getattr(JG,name) for name in ('schedule_hash','_edits_for','_execute_step_with_reason')}
    calls=[]
    class Cache:
        def execute(self, fn, *args):
            result,reason=fn(*args)
            return result,reason,False
    try:
        JG.schedule_hash=lambda schedule: schedule['tag']
        JG._edits_for=lambda ast,meta: (meta,None)
        def execute(executor,problem,schedule,edits,makespan,state_hash):
            assert schedule=={'tag':'unchanged'}
            calls.append(edits['value'])
            return (None if edits['value'] is None else
                    {'schedule':SimpleNamespace(makespan=edits['value'])}), 'test'
        JG._execute_step_with_reason=execute
        metas=[dict(kind='single',value=120),dict(kind='pair',value=110),
               dict(kind='single',value=None)]
        with torch.no_grad():
            result=compare(torch.zeros(3),1.,.05,random.Random(4),list(range(3)),None,
                metas,None,None,{'tag':'unchanged'},100,'unchanged',Cache())
        assert result[0]==1 and result[1]['schedule'].makespan==110  # worsening permitted
        assert len(calls)==3 and result[4]['trial_pairs']==1 and result[4]['feasible']==2
        assert len(set(result[2]))==3
        metas=[dict(kind='single',value=None)]
        with torch.no_grad():
            result=compare(torch.zeros(1),1.,.05,random.Random(4),[0],None,
                metas,None,None,{'tag':'unchanged'},100,'unchanged',Cache())
        assert result[0]==0 and result[1] is None and result[4]['feasible']==0
    finally:
        for name,value in saved.items():setattr(JG,name,value)
    print('[PASS] likelihood, K=1/K=N, padding, sampling, gradients, collector integration, '
          'best feasible selection, worsening allowed, all-infeasible handling')
    print('[NOTE] Full AutoDL rollout/GPU parity still runs during the first training cycle.')


if __name__=='__main__':
    main()
