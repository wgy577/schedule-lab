"""Differentiable replay on the exact behavior-policy candidate support.

Enumeration, pruning, Memory retrieval and schedules are discrete observations.
Only their identities are cached; learned embeddings/scores are recomputed.
"""
import copy
import dataclasses
import torch
from . import config as C
from .rolling_grpo import mixture_logp, mixture_pmf


def move(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if dataclasses.is_dataclass(x):
        return dataclasses.replace(x, **{f.name: move(getattr(x, f.name), device)
                                       for f in dataclasses.fields(x)})
    if isinstance(x, dict):
        return {k: move(v, device) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(move(v, device) for v in x)
    if isinstance(x, list):
        return [move(v, device) for v in x]
    return x


def capture_decision(ast, metas, selected, memory, prop_feats=None):
    assert ast['_e2e_input'] is not None
    atoms = []; proposals=[]; atom_index={}
    for k in selected:
        m = metas[k]
        ids=[]
        for idx in (m['i'], m['j']):
            if idx is None: continue
            if idx not in atom_index:
                r=ast['pool'][idx]
                roots=(r['root_ops'] if r['e'].edit_type!='ROUTE' else (r['e'].operation_id,))
                atom_index[idx]=len(atoms)
                atoms.append(dict(ni=r['ni'],feat=r['feat'].clone(),roots=roots))
            ids.append(atom_index[idx])
        proposals.append(dict(atoms=ids, family=m.get('family','single'),
            pstruct=(prop_feats[k,280:296].clone() if prop_feats is not None else torch.zeros(16))))
    return dict(input=ast['_e2e_input'], atoms=atoms, proposals=proposals,
                node_index=ast['node_index'], attributed=tuple(ast['op_b5']),
                app_rows=ast.get('_e2e_app_rows', {}),
                memory=memory[selected].clone())


def modules(r):
    return dict(encoder_prior=r['model_b5'], utility=r['single_head'], pair_utility=r['direct_head'],
                scorer=r['scorer'], target=r['policy'].m2, operator=r['policy'].m3)


def reset_rl_actors(r):
    """Keep original SFT upstream weights; discard old RL residual/trace weights."""
    from .hierarchical_residual import M2RootSetResidualActor, M3ProposalSetResidualActor
    a,b = r['policy'].m2,r['policy'].m3
    r['policy'].m2 = M2RootSetResidualActor(**{k:getattr(a,k) for k in (
        'root_latent_dim','local_dim','state_dim','memory_dim','cap','target_alpha',
        'warmup_fraction','d_model','nhead','num_layers','dim_feedforward',
        'relation_dim','root_value_scale')})
    r['policy'].m2.enable_probability_trace()
    r['policy'].m3 = M3ProposalSetResidualActor(b.r6,**{k:getattr(b,k) for k in (
        'evidence_dim','trajectory_dim','use_trajectory','cap_prop','cap_stop',
        'target_alpha','warmup_fraction','d_model','nhead','num_layers','dim_feedforward')})


def configure(r, trainable):
    # Use the same unfused FP32 attention paths during collection and replay.
    # Inference-only Transformer fusion can otherwise differ from autograd paths.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    C.TO1_R14_BETA_M2 = 0.0
    C.TO1_R14_BETA_M3 = 0.0
    r['model_b5'].e2e_capture = True
    r['model_b5'].e2e_fast_encoder = True
    r['policy'].m3.r6_view.end_to_end = True
    for m in modules(r).values():
        # Disable dropout in both behavior and update; eval does NOT block grad.
        m.eval().requires_grad_(trainable)


def prepare(r, rec, device, encoded=None):
    d = rec['e2e']
    inp = d['input']
    batch = move(inp['batch'], device)
    model = r['model_b5']
    model.bind_runtime_context(inp['context'], node_ids=list(inp['node_ids']))
    enc = model._encode_h_a(batch) if encoded is None else encoded
    ha, hc, qa, _ = enc
    scores = model.attribution_forward(batch, move(inp['c_prior'], device),
                                       encoded=enc).per_block_candidate_score
    ni = d['node_index']
    # Membership and support stay fixed during the PPO epochs.
    attributed = list(d['attributed'])
    att_values = scores[:, [ni['operation:'+op] for op in attributed]].amax(dim=0)
    att = dict(zip(attributed, att_values.unbind()))
    zero = hc.new_zeros(())
    mr = rec['m2_rec']
    roots = mr['ops']
    vals = torch.stack([att.get(op, zero) for op in roots])
    norm = (vals - vals.mean()) / (vals.std(unbiased=False) + 1e-6)
    local = mr['cand_f'].to(device).clone()
    local[:, 0] = vals / (torch.stack(list(att.values())).max().clamp_min(1e-12)
                          if att else hc.new_tensor(1.))
    latent = hc[[ni['operation:'+op] for op in roots]]
    app = mr['cand_app_raw'].to(device).clone()
    h = hc.shape[-1]
    entries = [(row, slot, bi, ni['operation:'+op])
               for row, op in enumerate(roots)
               for slot, bi in enumerate(d['app_rows'].get(op, ()))]
    if entries:
        rows, slots, blocks, nodes = torch.tensor(entries, device=device).unbind(1)
        app[rows, slots, :2*h+1] = torch.cat(
            [qa[blocks], ha[blocks, nodes], scores[blocks, nodes, None]], dim=1)
    graph = move(mr['cand_trace_graph'], device)
    graph['nodes'] = hc[list(graph['node_indices'])]
    cand = dict(n_cand=len(roots), base=norm, feats=local, latents=latent,
                state_context=mr['state_context'].to(device),
                memory_evidence=mr['cand_memory'].to(device), trace_graph=graph,
                app_raw=app, app_mask=mr['cand_app_mask'].to(device))
    return proposal_inputs(r, rec, device, hc, att_values, att, cand)


def proposal_inputs(r, rec, device, hc, att_values, att, cand):
    d=rec['e2e']; attributed=list(d['attributed']); zero=hc.new_zeros(())
    atoms=d['atoms']
    feat=torch.stack([atom['feat'] for atom in atoms]).to(device).clone()
    op_index={op:i for i,op in enumerate(attributed)}
    width=max(len(a['roots']) for a in atoms)
    indices=[[op_index.get(op,len(attributed)) for op in a['roots']]
             +[len(attributed)]*(width-len(a['roots'])) for a in atoms]
    vals=torch.cat([att_values,zero[None]])
    mask=torch.arange(width,device=device)[None]<torch.tensor([len(a['roots']) for a in atoms],device=device)[:,None]
    feat[:,10]=vals[torch.tensor(indices,device=device)].masked_fill(~mask,-torch.inf).amax(1)
    emb=hc[[a['ni'] for a in atoms]]
    utility=r['single_head'](torch.cat([emb,feat],1)).reshape(-1)
    ps=d['proposals']; first=[p['atoms'][0] for p in ps]
    second=[p['atoms'][1] if len(p['atoms'])==2 else 0 for p in ps]
    pair=torch.tensor([len(p['atoms'])==2 for p in ps],device=device)
    hu=emb[first]; hv=emb[second]*pair[:,None]; fu=feat[first]; fv=feat[second]*pair[:,None]
    struct=torch.stack([p['pstruct'] for p in ps]).to(device)
    uu=utility[first]; uv=utility[second]*pair
    direct=uu+uv
    routes=[i for i,p in enumerate(ps) if p['family']=='ROUTE+ROUTE']
    if routes:
        direct=direct.clone()
        direct[routes]=r['direct_head'](torch.cat([hu[routes],hv[routes],fu[routes],fv[routes],struct[routes]],1)).reshape(-1)
    pf=torch.cat([hu,hv,fu,fv,struct,torch.stack([uu,uv,direct,pair.to(uu.dtype)],1)],1)
    return dict(cand=cand,pf=pf,rec=rec)


def recompute(r, rec, device):
    prepared=prepare(r,rec,device)
    cand=prepared['cand']; pf=prepared['pf']; mr=rec['m2_rec']; d=rec['e2e']
    latent=cand['latents']; local=cand['feats']; norm=cand['base']; app=cand['app_raw']; graph=cand['trace_graph']
    zero=pf.new_zeros(()); lp2,old2=zero,0.0
    # Identical graph/state for all without-replacement draws. Share this
    # differentiable graph within ONE decision; never cache across updates.
    cand['_trace_bias'] = r['policy'].m2.trace_bias(graph,cand['state_context'])
    draws = []
    # Teacher-forced without-replacement supports are already fixed by rollout.
    # Evaluate all selection queries in one actual Transformer batch.
    supports = [list(draw['rem_ids']) for draw in mr['draws']]
    width = max(map(len, supports))
    index = torch.tensor([ids + [0]*(width-len(ids)) for ids in supports], device=device)
    valid = torch.arange(width, device=device)[None] < torch.tensor(
        list(map(len, supports)), device=device)[:, None]
    bias = cand['_trace_bias']
    logits_batch = r['policy'].m2.final_logits_batched(
        latent[index], local[index], norm[index],
        cand['state_context'].reshape(1,-1).expand(len(supports),-1),
        cand['memory_evidence'][index], valid, app_raw=app[index],
        app_mask=cand['app_mask'][index],
        trace_bias_override=(bias[index] if bias is not None else torch.zeros_like(norm[index])),
        end_to_end=True)
    for draw_index, draw in enumerate(mr['draws']):
        ids = list(draw['rem_ids'])
        logits = logits_batch[draw_index, :len(ids)]
        if 'exploration_bias' in mr:
            logits=logits+mr['exploration_bias'].to(device)[ids]
        p = torch.softmax(logits / float(C.TO1_R13_TEMP_M2), 0)
        eps = float(C.TO1_R13_MIX_EPS)
        p = (1-eps)*p + eps/len(ids)
        draw_lp = p[ids.index(draw['idx'])].clamp_min(1e-12).log()
        lp2 = lp2 + draw_lp
        draws.append((draw_lp, draw['logp']))
        old2 += draw['logp']
    mem = d['memory'].to(device)
    sf = rec['sf_t'].to(device)
    scorer = r['scorer']
    sh = scorer.shared(torch.cat([pf, mem, sf[None].expand(len(pf), -1)], 1))
    rank = scorer.rank_head(sh)
    useful = scorer.usefulness_head(sh)
    # Unchanged role/state/retrieved Memory columns from collection.
    tail = rec['F_pool'][:, sh.shape[1]+3:].to(device)
    old_base=torch.where(pf[:,299]<.5,pf[:,296],pf[:,298])[:,None]
    F = torch.cat([sh, rank, useful, old_base, tail], 1)
    stats = torch.stack([F.new_tensor(len(F)/100), F[:,258].max(),
                         F[:,258].mean(), F[:,256].max(), F[:,257].max()])[None]
    logits = r['policy'].m3.action_logits(
        F, sf, stats, evid=move(rec['evid'], device),
        traj_ctx=rec['traj_ctx'].to(device),
        pair_mask=rec['pair_mask'].to(device))[:rec['M']]
    if 'm3_exploration_bias' in rec:
        logits=logits+rec['m3_exploration_bias'].to(device)
    from .step20_training import logp_from_logits
    lp3 = logp_from_logits(logits, rec, C.TO1_R13_TEMP, C.TO1_R13_MIX_EPS)
    return lp2, lp3, old2, dict(F=F, base=norm, logits=logits,
                               draws=draws)


def clipped_loss(lp, old, advantage, clip=.2):
    ratio = (lp-float(old)).clamp(-20,20).exp()
    return -torch.minimum(ratio*advantage, ratio.clamp(1-clip,1+clip)*advantage)

# V22 ordered candidate-set training
