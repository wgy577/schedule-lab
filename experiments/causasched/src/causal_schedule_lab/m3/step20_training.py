"""On-policy ordered candidate-set sampling followed by deterministic best execution."""
import functools
import inspect
import os
import torch


def candidate_count():
    value = int(os.environ.get('E2E_STEP_CANDIDATES', '20'))
    if not 1 <= value <= 100:
        raise ValueError('E2E_STEP_CANDIDATES must be in [1, 100]')
    return value


def ordered_sample(weights, k, rng):
    remaining = [i for i, w in enumerate(weights) if w > 0]
    chosen = []
    for _ in range(min(k, len(remaining))):
        threshold = rng.random() * sum(weights[i] for i in remaining)
        index = remaining[-1]
        for i in remaining:
            threshold -= weights[i]
            if threshold < 0:
                index = i
                break
        chosen.append(index)
        remaining.remove(index)
    return chosen


def batch_logp(probs, records):
    """Exact ordered sampling-without-replacement log likelihood; legacy K=1 allowed."""
    sequences = [s.get('m3_candidate_draws', [int(s['a'])]) for s in records]
    if any(not q or len(q) != len(set(q)) for q in sequences):
        raise ValueError('Empty or duplicate candidate draws')
    for q, s in zip(sequences, records):
        if min(q) < 0 or max(q) >= int(s['M']):
            raise ValueError('Candidate draw outside original pool')
    width = max(map(len, sequences))
    ids = torch.tensor([q + [0] * (width-len(q)) for q in sequences],
                       device=probs.device, dtype=torch.long)
    active = torch.arange(width, device=probs.device)[None] < torch.tensor(
        list(map(len, sequences)), device=probs.device)[:, None]
    picked = torch.nn.functional.one_hot(ids, probs.shape[1]).to(probs.dtype)
    picked = picked * active[:, :, None]
    remaining = 1 - (picked.cumsum(1) - picked)
    denominators = (remaining * probs[:, None, :]).sum(-1)
    numerators = probs.gather(1, ids)
    terms = numerators.clamp_min(1e-30).log() - denominators.clamp_min(1e-30).log()
    return torch.where(active, terms, torch.zeros_like(terms)).sum(1)


def logp_from_logits(logits, rec, temperature, epsilon):
    p = torch.softmax(logits / float(temperature), 0)
    p = (1-float(epsilon))*p + float(epsilon)/len(p)
    return batch_logp(p[None], [rec])[0]


@functools.lru_cache(maxsize=4)
def build_collector(k):
    from . import joint_grpo as JG

    def compare(logits, temperature, epsilon, rng, pool, ast, metas,
                executor, problem, schedule, makespan, state_hash, replay_cache):
        if torch.is_grad_enabled():
            raise RuntimeError('Collection must run under no_grad')
        probabilities = JG.mixture_pmf(logits, temperature, epsilon)
        draws = ordered_sample(probabilities.detach().cpu().tolist(), min(k, len(pool)), rng)
        if not draws:
            raise RuntimeError('No supported candidates')
        winner, best, successor = draws[0], None, None
        pairs = feasible = 0
        old_hash = JG.schedule_hash(schedule)
        for a in draws:
            meta = metas[pool[a]]
            pairs += int(meta.get('kind') == 'pair')
            edits, _ = JG._edits_for(ast, meta)
            result, _, _ = replay_cache.execute(JG._execute_step_with_reason,
                executor, problem, schedule, edits, makespan, state_hash)
            if JG.schedule_hash(schedule) != old_hash:
                raise RuntimeError('Candidate evaluation mutated the shared start state')
            if result is not None:
                feasible += 1
                value = int(result['schedule'].makespan)
                if best is None or value < best:
                    winner, best, successor = a, value, result
        rec = dict(M=len(pool), a=winner, m3_candidate_draws=draws)
        lp = float(batch_logp(probabilities[None], [rec])[0])
        return winner, successor, draws, lp, dict(trials=len(draws),
            trial_pairs=pairs, feasible=feasible, chosen_makespan=best)

    source = inspect.getsource(JG.collect_trajectory_r14)
    start = source.index('        if (t == 0 and action_space == "policy_sampled" and not anchor_mode')
    end = source.index('        is_stop = bool(a == M)', start)
    source = source[:start] + '''        if action_space != "policy_sampled" or anchor_mode or allow_policy_stop:
            raise RuntimeError("Candidate-set training requires policy sampling without anchors/STOP")
        a, anchor_successor, candidate_draws, logp_old, trial_info = _compare_train(
            sample_logits,T,eps,rng,pool,ast,gated_metas,executor,problem,
            schedule,ms_cur,h,replay_cache)
        m3_sampling_mode = "ordered_candidate_set_best_feasible"
''' + source[end:]
    needle = '            "m3_sampling_mode": m3_sampling_mode,'
    if source.count(needle) != 1:
        raise RuntimeError('Unexpected collector record layout')
    source = source.replace(needle, needle + '''
            "m3_candidate_draws": candidate_draws,
            "step_compare": trial_info,
            "m3_probability_semantics": "ordered_without_replacement_joint_logp",
''')
    source = source.replace('execution_reason = "cached_anchor_probe"',
                            'execution_reason = "cached_candidate_set_winner"')
    scope = dict(JG.collect_trajectory_r14.__globals__)
    scope['_compare_train'] = compare
    exec(compile(source, '<candidate-set-training>', 'exec'), scope)
    return scope['collect_trajectory_r14']


def collect_trajectory(*args, **kwargs):
    tr = build_collector(candidate_count())(*args, **kwargs)
    steps = tr.get('steps', [])
    tr['candidate_set_diagnostics'] = dict(
        requested=candidate_count(),
        trials=sum(s.get('step_compare', {}).get('trials', 0) for s in steps),
        trial_pairs=sum(s.get('step_compare', {}).get('trial_pairs', 0) for s in steps),
        executed_pairs=sum(bool(s.get('selected_is_pair')) for s in steps))
    return tr


def report(trajs, iid, cycle, output):
    import json
    from pathlib import Path
    rows = [s for t in trajs for s in t['steps']]
    data = dict(cycle=int(cycle), instance_id=str(iid), decisions=len(rows),
        requested_candidates=candidate_count(),
        trials=sum(s.get('step_compare', {}).get('trials', 0) for s in rows),
        trial_pairs=sum(s.get('step_compare', {}).get('trial_pairs', 0) for s in rows),
        feasible_trials=sum(s.get('step_compare', {}).get('feasible', 0) for s in rows),
        executed_pairs=sum(bool(s.get('selected_is_pair')) for s in rows),
        probability_scope='joint probability of ordered candidate draws, not winner marginal')
    print(f'[candidate-set] {iid} decisions={data["decisions"]} '
          f'trials={data["trials"]} trial_pairs={data["trial_pairs"]} '
          f'executed_pairs={data["executed_pairs"]}', flush=True)
    with (Path(output)/'candidate_set_diagnostics.jsonl').open('a') as f:
        f.write(json.dumps(data)+'\n')
        f.flush()
        os.fsync(f.fileno())
