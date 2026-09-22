"""Raw global processing-load standard deviation reward, no reward calibration."""
import math

LOAD_METRIC='global_processing_load_std_raw_v1'

def load_cost(problem, schedule):
    modes = problem.mode_map()
    loads = {r.id: 0.0 for r in problem.resources}
    for a in schedule.assignments:
        mode = modes[a.mode_id][1]
        for rid in mode.resources:
            loads[rid] += a.end - a.start
    values=list(loads.values())
    if not values:return 0.0
    mean=math.fsum(values)/len(values)
    if mean<=0:return 0.0
    return math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values))


def load_bonus(before, after, root_ms, weight):
    return float(weight) * (before - after)


def add_load_credit(tr, problem, root, weight):
    before = load_cost(problem, root)
    terminal = load_cost(problem, tr['terminal_schedule'])
    bonus = load_bonus(before, terminal, tr['root_ms'], weight)
    tr['makespan_reward'] = tr['reward']
    tr['load_reward'] = bonus
    tr['load_cost_start'] = before
    tr['load_cost_terminal'] = terminal
    tr['load_cost_best'] = load_cost(problem,tr['best_schedule']) if 'best_schedule' in tr else None
    tr['load_metric'] = LOAD_METRIC
    tr['reward'] += bonus
    tr['U2'] += bonus
    tr['net_intervention_reward'] = tr['reward']
    tr['reward_semantics'] = 'net_makespan_return_plus_raw_std_change'
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


def balance_group(trajs, share=None):
    """Compatibility entry point: report observed share, never rescale rewards.

    The legacy share argument is intentionally ignored. GRPO advantage
    normalization remains in the trainer and is not modified here.
    """
    main = sum(abs(t['makespan_reward']) for t in trajs)
    aux = sum(abs(t['load_reward']) for t in trajs)
    result = dict(scale=1.0, actual_share=aux/max(main+aux, 1e-12),
                  main_abs=main, load_abs=aux, calibration=False,
                  metric=LOAD_METRIC)
    for tr in trajs:
        tr['load_reward_raw'] = tr['load_reward']
        tr['reward'] = tr['makespan_reward'] + tr['load_reward']
        tr['U2'] = tr['net_intervention_reward'] = tr['reward']
        tr['load_balance'] = result
        for step in tr['steps']:
            step['load_future_reward_raw'] = step['load_future_reward']
            step['m2_future_net_reward'] = step['makespan_future_reward'] + step['load_future_reward']
            step['reward_objective'] = tr['reward']
    return result
