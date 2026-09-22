"""Persistent trajectory-level workers; explicit audited single-only mode."""
import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import torch
import run_fast_inference as base
from causal_schedule_lab.m3 import joint_grpo as JG, config as C, proposal_features as PF
from causal_schedule_lab.m3.fast_inference import FastAnalyzeCache, companion_pairs
from causal_schedule_lab.m3.persistent_frontier_grpo import FrontierState
from causal_schedule_lab.ir import Problem, Schedule
from causal_schedule_lab.core_validation import validate_schedule
from causal_schedule_lab.validation import schedule_hash

CACHE = REPLAY = None


class AuditCache(FastAnalyzeCache):
    def __init__(self, *args, single_only, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_only = single_only
        self.single_seen = self.pair_seen = 0

    def proposals(self, *args):
        result = super().proposals(*args)
        pairs = sum(m['kind'] == 'pair' for m in result[1])
        self.single_seen += len(result[1]) - pairs
        self.pair_seen += pairs
        if self.single_only and pairs:
            raise AssertionError('single-only violation: pair candidate reached policy')
        return result


def set_single_only():
    # Modules imported constants by value: configure BOTH copies, not only C.
    for name in ('T2L_PAIR_TOTAL_CAP', 'T2L_PAIRS_PER_FAMILY',
                 'T2L_PAIR_ROUTE_ATOMS', 'T2L_PAIR_SEQUENCE_ATOMS'):
        setattr(C, name, 0)
        setattr(PF, name, 0)
    C.T2L_BASE_PAIR_ACTIONS = C.T2L_PLATEAU_PAIR_ACTIONS = 0
    # Empty fake inputs are safe only when construction truly returns early.
    if companion_pairs(None, None) != []:
        raise AssertionError('Pair construction did not short-circuit')


def initialize(runtime, roots, single_only):
    global CACHE, REPLAY
    if single_only:
        set_single_only()
    base.initialize(runtime, roots)
    if single_only:
        set_single_only()
    r = base.RUNTIME
    CACHE = AuditCache(r['model_b5'], r['single_head'], r['direct_head'],
                       max_entries=6, single_only=single_only)
    REPLAY = JG._new_replay_cache()
    print(f'[mode] worker={os.getpid()} single_only={single_only} '
          f'pair_construction={"OFF" if single_only else "ON"}', flush=True)


def trajectory(task):
    problem, iid, root, batch, branch, cfg = task
    started = time.perf_counter()
    r = base.RUNTIME
    memory = copy.deepcopy(r['memory'])
    for rec in root.records:
        memory.add_executed(iid, int(rec['written_at_step']), copy.deepcopy(rec))
    seed = int.from_bytes(hashlib.sha256(
        f'{cfg["seed"]}:{iid}:{batch}:{branch}'.encode()).digest()[:4], 'little')
    single_before, pair_before = CACHE.single_seen, CACHE.pair_seen
    with torch.no_grad():
        tr = JG.collect_trajectory_r14(
            r['policy'], r['scorer'], r['executor'], r['model_b5'], r['single_head'],
            r['direct_head'], problem, root.schedule, root.ms, iid, 900_000_000,
            memory, seed, branch, horizon=cfg['horizon'], step_offset=root.gstep,
            stop_on_negative=False, action_space='policy_sampled', analyze_cache=CACHE,
            allow_policy_stop=False, feasible_fallback=True, anchor_trajectories=0,
            replay_cache=REPLAY)
    states = []
    for sch, records, steps in (
        (tr['best_schedule'], tr['best_memory_records'], tr['best_step']),
        (tr['terminal_schedule'], tr['terminal_memory_records'], tr['n_steps'])):
        if not validate_schedule(problem, sch).feasible:
            raise AssertionError('Invalid retained schedule')
        states.append(FrontierState(sch, int(sch.makespan), root.gstep + steps,
                      records=tuple(root.records) + tuple(records), generation=batch+1,
                      parent_hash=root.state_hash))
    compact = [{k: rec.get(k) for k in (
        'step_idx', 'state_makespan_before', 'action_signature', 'selected_is_pair',
        'edit_types', 'execution_reason', 'is_anchor')} for rec in tr['steps']]
    pair_actions = sum(bool(rec.get('selected_is_pair')) for rec in compact)
    if cfg['single_only'] and pair_actions:
        raise AssertionError('single-only violation: selected pair action')
    row = dict(instance_id=iid, batch=batch+1, branch=branch, seed=seed,
               start_hash=root.state_hash, start_makespan=root.ms,
               best_makespan=tr['best_ms'], terminal_makespan=tr['final_ms'],
               n_steps=tr['n_steps'], terminal=tr['terminal'], steps=compact,
               profile_seconds=tr['prof'], pair_actions=pair_actions,
               single_candidates_seen=CACHE.single_seen-single_before,
               pair_candidates_seen=CACHE.pair_seen-pair_before,
               real_execution_calls=tr['real_execution_calls'],
               worker_pid=os.getpid(), worker_load_seconds=base.LOAD_SECONDS,
               trajectory_seconds=time.perf_counter()-started)
    return iid, branch, states, row


def choose_root(state, batch, branch):
    if branch == 0:
        return state['best']
    if branch == 1 and batch % 5 == 0:
        return state['initial']
    others = [s for s in state['pool'] if s.state_hash != state['best'].state_hash]
    return others[(batch+branch) % len(others)] if others else state['initial']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, default=ROOT/'inference_assets/runtime.pt')
    p.add_argument('--bank', type=Path, default=ROOT/'data/train128')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--instances', nargs='*')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--branches', type=int, default=16)
    p.add_argument('--batches', type=int, default=101)
    p.add_argument('--horizon', type=int, default=10)
    p.add_argument('--roots', type=int, default=6)
    p.add_argument('--pool-size', type=int, default=6)
    p.add_argument('--seed', type=int, default=922026)
    p.add_argument('--single-only', action='store_true')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if min(a.workers, a.branches, a.batches, a.horizon, a.roots) < 1 or a.pool_size < 2:
        p.error('Invalid budget')
    if a.single_only:
        set_single_only()
    if a.check_only:
        print(json.dumps(dict(single_only=a.single_only,
              pair_total_cap=PF.T2L_PAIR_TOTAL_CAP,
              pair_construction='OFF' if a.single_only else 'ON',
              parallelism='trajectory', workers=a.workers)))
        return
    manifest = json.loads((a.bank/'protocol.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Incomplete bank')
    selected = [r for r in manifest['entries'] if not a.instances or r['instance_id'] in a.instances]
    if not selected or (a.instances and set(a.instances) != {r['instance_id'] for r in selected}):
        raise ValueError('Missing requested instance')
    metadata = json.loads(a.runtime.with_suffix('.json').read_text())
    arena = {}
    for r in selected:
        x = json.loads((a.bank/r['file']).read_text())
        problem, schedule = Problem.model_validate(x['problem']), Schedule.model_validate(x['schedule'])
        if not validate_schedule(problem, schedule).feasible:
            raise ValueError('Invalid initial schedule')
        iid = r['instance_id']
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in iid)
        initial = FrontierState(schedule, int(schedule.makespan))
        arena[iid] = dict(problem=problem, initial=initial, best=initial, pool=[initial],
                          folder=a.output/(safe+'_'+hashlib.sha256(iid.encode()).hexdigest()[:8]),
                          steps=0, executions=0, stall=0)
    a.output.mkdir(parents=True, exist_ok=False)
    for s in arena.values():
        s['folder'].mkdir()
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    cfg.update(parallelism='trajectory', anchors=0, worker_threads=1, device='cpu',
               pair_mode='disabled' if a.single_only else 'one_related_companion_per_atom',
               frozen_batch_start=True, worse_intermediate_states=True,
               model_update=metadata['optimizer_update'])
    base.write_json(a.output/'protocol.json', cfg)
    started = time.perf_counter()
    nworkers = min(a.workers, len(arena)*a.branches)
    print(f'[parallel] workers={nworkers} jobs_per_batch={len(arena)*a.branches} '
          f'single_only={a.single_only}', flush=True)
    with ProcessPoolExecutor(max_workers=nworkers, mp_context=multiprocessing.get_context('spawn'),
                            initializer=initialize, initargs=(a.runtime, a.roots, a.single_only)) as executor:
        for batch in range(a.batches):
            before = {iid: s['best'].ms for iid, s in arena.items()}
            # Select all sibling starts before execution; completion order cannot affect search.
            tasks = [(s['problem'], iid, choose_root(s, batch, b), batch, b, cfg)
                     for b in range(a.branches) for iid, s in arena.items()]
            iterator = iter(tasks)
            pending, results = set(), {iid: {} for iid in arena}
            done = 0
            for _ in range(min(len(tasks), nworkers*2)):
                pending.add(executor.submit(trajectory, next(iterator)))
            while pending:
                finished, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
                if not finished:
                    print(f'[parallel] batch={batch+1} waiting jobs={done}/{len(tasks)}', flush=True)
                for future in finished:
                    iid, branch, states, row = future.result()
                    s = arena[iid]
                    results[iid][branch] = (states, row)
                    for state in states:
                        if state.ms < s['best'].ms:
                            s['best'] = state
                    with (s['folder']/'trajectories.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    base.write_json(s['folder']/'best_schedule.json', dict(
                        instance_id=iid, problem=s['problem'].model_dump(mode='json'),
                        schedule=s['best'].schedule.model_dump(mode='json'),
                        makespan=s['best'].ms, feasible=True, batch=batch+1))
                    done += 1
                    task = next(iterator, None)
                    if task is not None:
                        pending.add(executor.submit(trajectory, task))
                    print(f'[parallel] batch={batch+1} jobs={done}/{len(tasks)} '
                          f'pid={row["worker_pid"]} {iid} pair_candidates='
                          f'{row["pair_candidates_seen"]} pair_actions={row["pair_actions"]}', flush=True)
            for iid, s in arena.items():
                ordered = [results[iid][b] for b in sorted(results[iid])]
                candidates = list(s['pool']) + [state for states, _ in ordered for state in states]
                # Tie-break archive by deterministic branch order, not worker arrival.
                best_ms = s['best'].ms
                s['best'] = next(state for state in candidates if state.ms == best_ms)
                s['pool'] = base.retain_diverse(candidates, s['best'], a.pool_size)
                base.write_json(s['folder']/'best_schedule.json', dict(
                    instance_id=iid, problem=s['problem'].model_dump(mode='json'),
                    schedule=s['best'].schedule.model_dump(mode='json'),
                    makespan=s['best'].ms, feasible=True, batch=batch+1))
                s['steps'] += sum(r['n_steps'] for _, r in ordered)
                s['executions'] += sum(r['real_execution_calls'] for _, r in ordered)
                s['stall'] = s['stall']+1 if s['best'].ms >= before[iid] else 0
                summary = dict(instance_id=iid, initial_makespan=s['initial'].ms,
                    best_makespan=s['best'].ms, completed_batches=batch+1,
                    initial_hash=s['initial'].state_hash, best_hash=s['best'].state_hash,
                    attempted_steps=s['steps'], real_execution_calls=s['executions'],
                    no_improvement_batches=s['stall'], feasible=True,
                    single_only=a.single_only, pool_makespans=[v.ms for v in s['pool']],
                    pair_candidates_seen=sum(r['pair_candidates_seen'] for _, r in ordered),
                    pair_actions=sum(r['pair_actions'] for _, r in ordered),
                    worker_pids=sorted({r['worker_pid'] for _, r in ordered}),
                    elapsed_wall_seconds=time.perf_counter()-started,
                    trajectory_seconds_sum=sum(r['trajectory_seconds'] for _, r in ordered),
                    in_training_set=iid in metadata['train_ids'], complete=batch+1 == a.batches)
                base.write_json(s['folder']/f'batch_{batch+1:04d}.json', summary)
                base.write_json(s['folder']/'result.json', summary)
                s['summary'] = summary
                print(f'[fast] {iid} batch={batch+1}/{a.batches} best={s["best"].ms} '
                      f'terminals={[r["terminal_makespan"] for _, r in ordered]} '
                      f'pair_candidates={summary["pair_candidates_seen"]} '
                      f'pair_actions={summary["pair_actions"]}', flush=True)
    base.write_json(a.output/'summary.json', dict(complete=True,
        end_to_end_seconds=time.perf_counter()-started, results=[s['summary'] for s in arena.values()]))


if __name__ == '__main__':
    main()
