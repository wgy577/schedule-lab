"""State-specific soft tabu biases, recorded exactly for on-policy replay."""
import math
import torch
from collections import Counter


def bias_for(context, state_hash, category, identities, temperature):
    counts=context.get(state_hash,{}).get(category,{})
    return torch.tensor([-float(temperature)*math.log(4)*min(counts.get(str(k),0),3)
                         for k in identities],dtype=torch.float32)


def remember_failed_batch(state, trajectories, improved):
    if improved:
        state['choice_history']={}
        return
    history=state.setdefault('choice_history',{})
    # Frequency-normalized per batch: repeated choices receive a larger penalty,
    # without making the scale grow with the number of parallel siblings.
    attempted={}
    for tr in trajectories:
        for s in tr['steps']:
            h=s['state_hash']
            row=attempted.setdefault(h,{'roots':Counter(),'actions':Counter()})
            row['roots'].update(map(str,s['m2_rec'].get('retained',())))
            if not s.get('is_stop',False): row['actions'].update([str(s['action_signature'])])
    for h,new in attempted.items():
        previous=history.pop(h,{'roots':{},'actions':{}})
        for category,capacity in (('roots',128),('actions',256)):
            for key in sorted(new[category]):
                frequency=new[category][key]/max(new[category].values())
                previous[category][key]=min(previous[category].get(key,0)+frequency,3)
            previous[category]=dict(list(previous[category].items())[-capacity:])
        history[h]=previous
    while len(history)>64: history.pop(next(iter(history)))
