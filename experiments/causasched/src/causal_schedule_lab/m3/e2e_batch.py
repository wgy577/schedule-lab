"""End-to-end replay with disjoint graph and padded M2/M3 decision batches."""
import torch
from torch.nn.utils.rnn import pad_sequence
from . import config as C
from .end_to_end import prepare, move
from .e2e_encoder import encode_many


def replay_batch(r, records, device):
    encoded=encode_many(r['model_b5'],[s['e2e']['input']['batch'] for s in records],device)
    prepared=[prepare(r,s,device,e) for s,e in zip(records,encoded)]
    cands=[p['cand'] for p in prepared]
    states=torch.stack([c['state_context'].reshape(-1) for c in cands])
    m2=r['policy'].m2
    width=max(c['n_cand'] for c in cands)
    if m2.probability_trace is not None:
        biases=max(float(m2.alpha_fraction),.1)*m2.probability_trace.batch_bias(
            [(c['trace_graph'],list(range(c['n_cand']))) for c in cands],
            states,width,end_to_end=True)
    else:
        biases=states.new_zeros((len(records),width))
    queries=[]; lat=[]; loc=[]; bas=[]; mem=[]; apps=[]; masks=[]; biases_q=[]; state_q=[]
    for i,(rec,c) in enumerate(zip(records,cands)):
        for draw in rec['m2_rec']['draws']:
            ids=list(draw['rem_ids'])
            queries.append((i,ids.index(draw['idx']),draw['logp'],len(ids)))
            lat.append(c['latents'][ids]); loc.append(c['feats'][ids]); bas.append(c['base'][ids])
            mem.append(c['memory_evidence'][ids]); apps.append(c['app_raw'][ids]); masks.append(c['app_mask'][ids])
            biases_q.append(biases[i,ids]); state_q.append(states[i])
    def pad(xs): return pad_sequence(xs,batch_first=True)
    pbase=pad(bas)
    valid=torch.arange(pbase.shape[1],device=device)[None]<torch.tensor([q[3] for q in queries],device=device)[:,None]
    scores=m2.final_logits_batched(pad(lat),pad(loc),pbase,torch.stack(state_q),pad(mem),valid,
        app_raw=pad(apps),app_mask=pad(masks),trace_bias_override=pad(biases_q),end_to_end=True)
    exploration=[]
    for rec in records:
        mr=rec['m2_rec']
        full=mr.get('exploration_bias',torch.zeros(mr['n_cand'])).to(device)
        exploration.extend(full[list(d['rem_ids'])] for d in mr['draws'])
    scores=scores+pad(exploration)
    probs=torch.softmax(scores/float(C.TO1_R13_TEMP_M2),dim=1)
    eps=float(C.TO1_R13_MIX_EPS)
    probs=(1-eps)*probs+eps*valid/probs.new_tensor([q[3] for q in queries])[:,None]
    positions=torch.tensor([q[1] for q in queries],device=device)
    draw_lp=probs.gather(1,positions[:,None]).squeeze(1).clamp_min(1e-12).log()
    owners=torch.tensor([q[0] for q in queries],device=device)
    lp2=draw_lp.new_zeros(len(records)).index_add(0,owners,draw_lp)
    pf=torch.cat([p['pf'] for p in prepared])
    lengths=[len(p['pf']) for p in prepared]
    sf=torch.stack([s['sf_t'].to(device) for s in records])
    mem=torch.cat([s['e2e']['memory'] for s in records]).to(device)
    expanded=sf.repeat_interleave(torch.tensor(lengths,device=device),dim=0)
    scorer=r['scorer']; sh=scorer.shared(torch.cat([pf,mem,expanded],1))
    tail=torch.cat([s['F_pool'][:,sh.shape[1]+3:] for s in records]).to(device)
    old_base=torch.where(pf[:,299]<.5,pf[:,296],pf[:,298])[:,None]
    features=torch.cat([sh,scorer.rank_head(sh),scorer.usefulness_head(sh),old_base,tail],1)
    fs=list(features.split(lengths))
    stats=torch.stack([torch.stack([f.new_tensor(len(f)/100),f[:,258].max(),f[:,258].mean(),
                                   f[:,256].max(),f[:,257].max()]) for f in fs])
    fp=pad(fs); valid3=torch.arange(fp.shape[1],device=device)[None]<torch.tensor(lengths,device=device)[:,None]
    evidence=pad([s['evid'].to(device) if s['evid'] is not None else fp.new_zeros((n,r['policy'].m3.evidence_dim))
                  for s,n in zip(records,lengths)])
    traj=torch.stack([s['traj_ctx'].to(device) for s in records])
    pair=pad([s['pair_mask'].to(device) for s in records])
    logits,_,_,_=r['policy'].m3.action_and_base_logits_batched(fp,sf,stats,evidence,traj,valid3,pair)
    logits=logits+pad([s.get('m3_exploration_bias',torch.zeros(n)).to(device)
                     for s,n in zip(records,lengths)])
    probs3=torch.softmax(logits/float(C.TO1_R13_TEMP),1)
    probs3=(1-eps)*probs3+eps*valid3/probs3.new_tensor(lengths)[:,None]
    actions=torch.tensor([s['a'] for s in records],device=device)
    from .step20_training import batch_logp
    lp3=batch_logp(probs3,records)
    return dict(lp2=lp2,lp3=lp3,draw_lp=draw_lp,draw_owner=owners,
                draw_old=draw_lp.new_tensor([q[2] for q in queries]),features=fs)


def loss_batch(out, records):
    adv2=out['lp2'].new_tensor([s['adv2'] for s in records])[out['draw_owner']]
    ratio=(out['draw_lp']-out['draw_old']).clamp(-20,20).exp()
    loss2=-torch.minimum(ratio*adv2,ratio.clamp(.8,1.2)*adv2)
    w2=ratio.new_tensor([s['_w2']/max(len(s['m2_rec']['draws']),1) for s in records])
    adv3=ratio.new_tensor([s['adv3'] for s in records])
    ratio3=(out['lp3']-ratio.new_tensor([s['logp_old'] for s in records])).clamp(-20,20).exp()
    loss3=-torch.minimum(ratio3*adv3,ratio3.clamp(.8,1.2)*adv3)
    return (loss2*w2[out['draw_owner']]).sum()+(loss3*ratio.new_tensor([s['_w3'] for s in records])).sum()

# V22 ordered candidate-set training
