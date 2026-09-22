"""Bounded load-change auxiliary return; not an optimal-policy guarantee.

Cost = maximum resource processing load + 0.1 * total resource processing.
This targets bottleneck load, not arbitrary equality of heterogeneous machines.
Sequence-only moves get zero signal; makespan remains the selection criterion.
"""
def load_cost(problem, schedule):
    modes = problem.mode_map()
    loads = {r.id: 0.0 for r in problem.resources}
    for a in schedule.assignments:
        mode = modes[a.mode_id][1]
        for rid in mode.resources:
            loads[rid] += a.end - a.start
    return max(loads.values(), default=0.0) + 0.1 * sum(loads.values())


def load_bonus(before, after, root_ms, weight):
    cap = 0.05 * max(float(root_ms), 1.0)
    return max(-cap, min(cap, float(weight) * (before - after)))


def add_load_credit(tr, problem, root, weight):
    before = load_cost(problem, root)
    terminal = load_cost(problem, tr['terminal_schedule'])
    bonus = load_bonus(before, terminal, tr['root_ms'], weight)
    tr['makespan_reward'] = tr['reward']
    tr['load_reward'] = bonus
    tr['load_cost_start'] = before
    tr['load_cost_terminal'] = terminal
    tr['reward'] += bonus
    tr['U2'] += bonus
    tr['net_intervention_reward'] = tr['reward']
    tr['reward_semantics'] = 'net_makespan_return_plus_bounded_load_change'
    value = before
    for step in tr['steps']:
        if weight and step.get('successor_makespan') is not None and 'load_cost_before' not in step:
            raise RuntimeError('Executed step missing load statistics')
        value = step.get('load_cost_before', value)
        step['load_immediate_reward'] = load_bonus(
            value, step.get('load_cost_after', value), tr['root_ms'], weight)
        future = load_bonus(value, terminal, tr['root_ms'], weight)
        step['makespan_future_reward'] = step['m2_future_net_reward']
        step['load_future_reward'] = future
        step['m2_future_net_reward'] += future
        step['reward_objective'] = tr['reward']
        value = step.get('load_cost_after', value)
    return tr


def balance_group(trajs, share=0.35):
    """Match auxiliary L1 share before joint GRPO normalization.

    Each timestep's future credit is calibrated separately. Zero auxiliary
    remains zero; zero main signal keeps existing nonzero auxiliary (100%).
    This is group-dependent multi-objective training, not policy-invariant PBRS.
    """
    def scale(rows, main_key, aux_key):
        main=sum(abs(r[main_key]) for r in rows)
        aux=sum(abs(r[aux_key]) for r in rows)
        factor=(share/(1-share)*main/aux if main>1e-12 and aux>1e-12 else 1.0)
        if share==0: factor=0.0
        for row in rows:
            row[aux_key+'_raw']=row[aux_key]
            row[aux_key]*=factor
        final_aux=sum(abs(r[aux_key]) for r in rows)
        return dict(scale=factor,actual_share=final_aux/max(main+final_aux,1e-12),
                    main_abs=main,load_abs=final_aux)
    result=scale(trajs,'makespan_reward','load_reward')
    for tr in trajs:
        tr['reward']=tr['makespan_reward']+tr['load_reward']
        tr['U2']=tr['net_intervention_reward']=tr['reward']
        tr['load_balance']=result
    for idx in range(max((len(t['steps']) for t in trajs),default=0)):
        rows=[t['steps'][idx] for t in trajs if idx<len(t['steps'])]
        scale(rows,'makespan_future_reward','load_future_reward')
        for step in rows:
            step['m2_future_net_reward']=step['makespan_future_reward']+step['load_future_reward']
    for tr in trajs:
        for step in tr['steps']: step['reward_objective']=tr['reward']
    return result
