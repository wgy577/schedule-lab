"""Immediate-only credit; unchanged search, grouping and optimizer."""
import math

def add_load_credit(tr, problem, root, weight):
    from .load_reward import add_load_credit as original
    original(tr, problem, root, weight)
    lam = float(tr['net_regression_weight'])
    for s in tr['steps']:
        before = float(s['state_makespan_before'])
        after = s.get('successor_makespan')
        delta = 0.0 if after is None else before-float(after)
        penalty = .05*before if s.get('execution_reason') == 'infeasible' else 0.0
        # Preserve the configured deterioration coefficient, now applied locally.
        main = max(delta, 0.0)-lam*max(-delta, 0.0)-penalty
        aux = float(s.get('load_immediate_reward', 0.0))
        if not math.isfinite(main+aux):
            raise ValueError('Nonfinite immediate reward')
        s.update(makespan_future_reward=main, load_future_reward=aux,
                 m2_future_net_reward=main+aux, immediate_reward=main+aux,
                 immediate_makespan_reward=main, credit_horizon=1)
    tr['makespan_reward'] = sum(s['makespan_future_reward'] for s in tr['steps'])
    tr['load_reward'] = sum(s['load_future_reward'] for s in tr['steps'])
    tr['reward'] = tr['makespan_reward']+tr['load_reward']
    tr['U2'] = tr['net_intervention_reward'] = tr['reward']
    tr['reward_semantics'] = 'one_step_immediate_v1'
    return tr

def finish_group(trajs, *args, **kwargs):
    from . import joint_grpo as JG
    totals = [(t['reward'], t['U2'], t['reward_semantics']) for t in trajs]
    try:
        for t in trajs:
            # The existing first-decision hierarchical grouping must use only
            # first-decision reward, never the trajectory total.
            first = t['steps'][0]['m2_future_net_reward'] if t['steps'] else 0.0
            t['reward'] = t['U2'] = first
            t['reward_semantics'] = 'unified_net_intervention_return'
        result = JG._finish_group_r14(trajs, *args, **kwargs)
    finally:
        for t, (reward, u2, semantics) in zip(trajs, totals):
            t['reward'], t['U2'], t['reward_semantics'] = reward, u2, semantics
            for s in t['steps']:
                s['credit_source'] = 'one_step_immediate_hierarchical_grpo'
                s['reward_objective'] = s['m2_future_net_reward']
                s['credit_horizon'] = 1
    return result
