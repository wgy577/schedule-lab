"""Restart only from candidates generated directly from the current incumbent.

The archive is separate from the exploration start. Never add candidates from
an already perturbed start to the restart pool, so restarts cannot chain away.
"""


def refresh_pool(state, candidates, max_gap=.10, capacity=8, source_hash=None):
    incumbent=state['current']
    anchor=incumbent.state_hash
    if state.get('perturb_anchor_hash')!=anchor:
        state['perturb_pool']=[]
        state['perturb_history']=[]
        state['perturb_anchor_hash']=anchor
    # Legacy pools without provenance are cleared above, not trusted on resume.
    additions=list(candidates) if source_hash==anchor else []
    unique={s.state_hash:s for s in state.get('perturb_pool',[])+additions
            if s.state_hash!=incumbent.state_hash and
            incumbent.ms<=s.ms<=incumbent.ms*(1+max_gap)}
    state['perturb_pool']=sorted(unique.values(),key=lambda s:(s.ms,s.state_hash))[:capacity]


def choose_start(state, patience=3, max_gap=.10):
    incumbent=state['current']
    if patience<=0 or state.get('stall_segments',0)<patience:
        return incumbent,False
    refresh_pool(state,[],max_gap)
    stalls=state.get('stall_segments',0)
    if stalls-state.get('last_perturb_stall',0)<patience:
        return incumbent,False
    history=state.setdefault('perturb_history',[])
    eligible=state['perturb_pool']
    if not eligible:
        return incumbent,False
    # Avoid recent restart states; when exhausted reuse the least recent one.
    unseen=[s for s in eligible if s.state_hash not in history]
    chosen=(unseen[0] if unseen else min(eligible,key=lambda s:history.index(s.state_hash)))
    state['perturb_history']=(history+[chosen.state_hash])[-8:]
    # Keep the stagnation counter for 20-step exploration; throttle restarts
    # separately. After this attempt, ordinary batches return to incumbent.
    state['last_perturb_stall']=stalls
    return chosen,True
