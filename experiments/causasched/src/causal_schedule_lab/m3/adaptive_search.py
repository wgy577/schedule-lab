"""Checkpointable per-instance episode budgets for variable rollout horizons."""
import copy


def selected_ids(ids, episode, roots):
    return [ids[(episode*roots+j)%len(ids)] for j in range(min(roots,len(ids)))]


def restore_clock(checkpoint, ids, roots, episode_steps, base_horizon, reset=False):
    if checkpoint is None or reset:
        return dict(episode=0,updates=0,budgets={})
    if 'episode_clock' in checkpoint:
        clock=copy.deepcopy(checkpoint['episode_clock'])
        expected=set(selected_ids(ids,clock['episode'],roots))
        if clock['budgets'] and set(clock['budgets'])!=expected:
            raise ValueError('Episode clock cohort mismatch')
        if any(not 0<=v<=episode_steps for v in clock['budgets'].values()):
            raise ValueError('Episode clock budget out of bounds')
        return clock
    # V8/V9 stored completed fixed-width updates, not an explicit step ledger.
    origin=checkpoint['config'].get('episode_origin_cycle',0)
    elapsed=checkpoint['cycle']-origin
    if elapsed<0:
        raise ValueError('Invalid legacy episode origin')
    completed,updates=divmod(elapsed,episode_steps//base_horizon)
    return dict(episode=completed,updates=updates,
                budgets=({iid:updates*base_horizon for iid in selected_ids(ids,completed,roots)}
                         if updates else {}))


def begin_episode(clock, ids, roots, episode_steps):
    """Return whether a new episode was initialized; caller resets schedules."""
    if clock['budgets'] and not all(v>=episode_steps for v in clock['budgets'].values()):
        return False
    if clock['budgets']:
        clock['episode']+=1
    clock['updates']=0
    clock['budgets']={iid:0 for iid in selected_ids(ids,clock['episode'],roots)}
    return True


def rollout_horizon(state, used, episode_steps, base=10, longer=20, patience=2):
    requested=longer if patience>0 and state.get('stall_segments',0)>=patience else base
    return max(0,min(requested,episode_steps-used))
