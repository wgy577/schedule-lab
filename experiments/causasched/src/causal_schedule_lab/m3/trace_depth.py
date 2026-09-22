"""Explicit depth expansion: preserve old rows and shared transition weights."""
import torch
from . import probability_trace as PT
from . import proposal_features as PF


def expand_trace_state(params, hops, prefix='probability_trace.'):
    embed=prefix+'hop_embedding.weight'
    weight=prefix+'depth.weight'
    bias=prefix+'depth.bias'
    if any(k not in params for k in (embed,weight,bias)):
        raise ValueError('Checkpoint lacks probability-trace parameters; cannot silently initialize them')
    old=int(params[embed].shape[0]); extra=int(hops)-old
    if old<1 or extra<0:
        raise ValueError(f'Unsupported trace-depth migration {old}->{hops}; shrinking is not allowed')
    if params[weight].shape[0]!=old+1 or params[bias].shape!=(old+1,):
        raise ValueError('Inconsistent checkpoint trace-depth dimensions')
    out=dict(params)
    if not extra:
        return out,[]
    out[embed]=torch.cat((params[embed],params[embed][-1:].repeat(extra,1)),0)
    out[weight]=torch.cat((params[weight],params[weight][-1:].repeat(extra,1)),0)
    # New depths initially get exp(-4) times the previous deepest-level mass.
    # This does not promise identical full outputs: the reachable graph expands.
    out[bias]=torch.cat((params[bias],params[bias][-1:].repeat(extra)-4.),0)
    return out,[embed,weight,bias]


def configure_trace_depth(runtime, hops):
    hops=int(hops)
    if hops<1:
        raise ValueError('trace hops must be positive')
    actor=runtime['policy'].m2
    actor.enable_probability_trace()
    old=actor.probability_trace
    params,changed=expand_trace_state(old.state_dict(),hops,prefix='')
    if changed:
        latent=old.seed.in_features
        state=old.depth.in_features-latent
        # Do not consume sampler RNG while creating replacement modules.
        with torch.random.fork_rng(devices=[]):
            new=PT.ProbabilityTrace(latent,state,hops=hops,hop_dim=old.hop_dim)
        new.to(device=old.strength.device,dtype=old.strength.dtype)
        new.load_state_dict(params,strict=True)
        new.train(old.training)
        old_named=dict(old.named_parameters())
        for name,param in new.named_parameters():
            param.requires_grad_(old_named[name].requires_grad)
        actor.probability_trace=new
    PF.ROOT_APP_MAX_HOPS=hops
    PT.TRACE_MAX_HOPS=hops
    assert actor.probability_trace.hops==PF.ROOT_APP_MAX_HOPS==PT.TRACE_MAX_HOPS


def reset_expanded_optimizer_states(optimizer,named,changed):
    """Call after loading old Adam state; reset only the resized tensors."""
    table=dict(named)
    if not set(changed)<=set(table):
        raise ValueError('Cannot identify resized optimizer parameters')
    for name in changed:
        optimizer.state.pop(table[name],None)
    for group in optimizer.param_groups:
        for p in group['params']:
            for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                value=optimizer.state.get(p,{}).get(key)
                if value is not None and value.shape!=p.shape:
                    raise ValueError(f'Remaining optimizer shape mismatch for {key}')
