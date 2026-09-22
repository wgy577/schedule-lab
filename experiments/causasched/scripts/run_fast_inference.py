"""Persistent CPU inference; multistep exploration, independent best archive.

No optimizer, training replay construction, or per-batch weight loading.
Each persistent worker loads the runtime once and processes whole instances.
"""
import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import torch
from causal_schedule_lab.m3 import joint_grpo as JG
from causal_schedule_lab.m3 import config as C
from causal_schedule_lab.m3.fast_inference import FastAnalyzeCache, FrozenBaseMemory
from causal_schedule_lab.m3.persistent_frontier_grpo import FrontierState
from causal_schedule_lab.m3.upstream import load_upstream
from causal_schedule_lab.validation import schedule_hash
from causal_schedule_lab.core_validation import validate_schedule
from causal_schedule_lab.ir import Problem, Schedule

RUNTIME = None
LOAD_SECONDS = 0.0


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def initialize(runtime_path, roots):
    global RUNTIME, LOAD_SECONDS
    start = time.perf_counter()
    torch.set_num_threads(1)
    load_upstream()
    RUNTIME = torch.load(runtime_path, map_location='cpu', weights_only=False)
    if RUNTIME['meta']['format'] != 'causasched-fast-runtime-v1':
        raise ValueError('Unsupported runtime')
    from causal_schedule_lab.m3.trace_depth import configure_trace_depth
    trace=getattr(RUNTIME['policy'].m2,'probability_trace',None)
    if trace is not None:
        hops=int(trace.hop_embedding.weight.shape[0])
        if int(RUNTIME['meta'].get('trace_hops',hops)) != hops:
            raise ValueError('Runtime trace-depth metadata mismatch')
        configure_trace_depth(RUNTIME,hops)
    if RUNTIME['meta'].get('pair_mode') == 'bounded-relations':
        from causal_schedule_lab.m3 import proposal_features as pf
        for name,value in dict(T2L_PAIR_ROUTE_ATOMS=24,T2L_PAIR_SEQUENCE_ATOMS=24,
                               T2L_PAIRS_PER_FAMILY=32,T2L_PAIR_TOTAL_CAP=96).items():
            setattr(C,name,value);setattr(pf,name,value)
        C.T2L_BASE_PAIR_ACTIONS=64;C.T2L_PLATEAU_PAIR_ACTIONS=96
    C.E2E_LOAD_REWARD_ENABLED=False
    for m in (RUNTIME['policy'].m2, RUNTIME['policy'].m3, RUNTIME['model_b5'],
              RUNTIME['scorer'], RUNTIME['single_head'], RUNTIME['direct_head']):
        m.eval()
        m.requires_grad_(False)
    RUNTIME['memory'] = FrozenBaseMemory.from_memory(RUNTIME['memory'])
    C.T2L_M2_ROOT_TOP_K = roots
    LOAD_SECONDS = time.perf_counter() - start
    print(f'[fast] worker={os.getpid()} loaded-once update='
          f'{RUNTIME["meta"]["optimizer_update"]} seconds={LOAD_SECONDS:.2f}', flush=True)


def retain_diverse(states, best, capacity):
    unique = {}
    for state in states:
        unique[state.state_hash] = state
    unique[best.state_hash] = best
    ranked = sorted(unique.values(), key=lambda s: (s.ms, s.state_hash))
    elites = ranked[:max(1, capacity // 2)]
    chosen = {s.state_hash for s in elites}
    # Keep recent exploratory terminal states, including worse ones.
    fresh = sorted((s for s in unique.values() if s.state_hash not in chosen),
                   key=lambda s: (-s.generation, -s.gstep, s.state_hash))
    return (elites + fresh)[:capacity]


def search_one(task):
    row, cfg = task
    started = time.perf_counter()
    x = json.loads(Path(row['path']).read_text())
    problem, schedule = Problem.model_validate(x['problem']), Schedule.model_validate(x['schedule'])
    if not validate_schedule(problem, schedule).feasible:
        raise ValueError('Invalid initial schedule: ' + row['iid'])
    iid = row['iid']
    safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in iid)
    folder = Path(cfg['output']) / (safe_name + '_' + hashlib.sha256(iid.encode()).hexdigest()[:8])
    folder.mkdir()
    initial_hash = schedule_hash(schedule)
    best = initial = FrontierState(schedule, int(schedule.makespan))
    pool = [initial]
    cache = FastAnalyzeCache(RUNTIME['model_b5'], RUNTIME['single_head'], RUNTIME['direct_head'], max_entries=8)
    replay_cache = JG._new_replay_cache()
    episode = 900_000_000
    attempted_steps = total_executions = 0
    stall = 0
    summary = None
    for batch in range(cfg['batches']):
        before = best.ms
        candidates = list(pool)
        branch_rows = []
        for branch in range(cfg['branches']):
            # Always explore the incumbent, but also other frontier states.
            if branch == 0:
                root = best
            elif branch == 1 and batch % 5 == 0:
                root = initial
            else:
                alternatives = [s for s in pool if s.state_hash != best.state_hash]
                root = alternatives[(batch + branch) % len(alternatives)] if alternatives else initial
            memory = copy.deepcopy(RUNTIME['memory'])
            for rec in root.records:
                memory.add_executed(iid, int(rec['written_at_step']), copy.deepcopy(rec))
            digest = hashlib.sha256(f'{cfg["seed"]}:{iid}:{batch}:{branch}'.encode()).digest()
            seed = int.from_bytes(digest[:4], 'little')
            with torch.no_grad():
                tr = JG.collect_trajectory_r14(
                    RUNTIME['policy'], RUNTIME['scorer'], RUNTIME['executor'],
                    RUNTIME['model_b5'], RUNTIME['single_head'], RUNTIME['direct_head'],
                    problem, root.schedule, root.ms, iid, episode, memory, seed, branch,
                    horizon=cfg['horizon'], step_offset=root.gstep,
                    stop_on_negative=False, action_space='policy_sampled',
                    analyze_cache=cache, allow_policy_stop=False, feasible_fallback=True,
                    anchor_trajectories=0, replay_cache=replay_cache)
            attempted_steps += tr['n_steps']
            total_executions += tr['real_execution_calls']
            for kind, sch, records, nsteps in (
                ('best', tr['best_schedule'], tr['best_memory_records'], tr['best_step']),
                ('terminal', tr['terminal_schedule'], tr['terminal_memory_records'], tr['n_steps'])):
                if not validate_schedule(problem, sch).feasible:
                    raise AssertionError('Invalid retained ' + kind)
                state = FrontierState(
                    sch, int(sch.makespan), gstep=root.gstep + nsteps,
                    records=tuple(root.records) + tuple(records), generation=batch + 1,
                    parent_hash=root.state_hash)
                candidates.append(state)
                if state.ms < best.ms:
                    best = state
            compact_steps = [{k: rec.get(k) for k in (
                'step_idx', 'state_makespan_before', 'action_signature', 'selected_is_pair',
                'edit_types', 'execution_reason', 'is_anchor')} for rec in tr['steps']]
            branch_row = {
                'batch': batch + 1, 'branch': branch, 'seed': seed,
                'start_hash': root.state_hash, 'start_makespan': root.ms,
                'best_makespan': tr['best_ms'], 'terminal_makespan': tr['final_ms'],
                'n_steps': tr['n_steps'], 'terminal': tr['terminal'],
                'profile_seconds': tr['prof'], 'steps': compact_steps,
            }
            branch_rows.append(branch_row)
            # Each finished trajectory is durable, not only end-of-run output.
            with (folder / 'trajectories.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(branch_row) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            write_json(folder / 'best_schedule.json', {
                'instance_id': iid, 'problem': problem.model_dump(mode='json'),
                'schedule': best.schedule.model_dump(mode='json'), 'makespan': best.ms,
                'feasible': True, 'batch': batch + 1, 'branch': branch})
        pool = retain_diverse(candidates, best, cfg['pool_size'])
        stall = stall + 1 if best.ms >= before else 0
        summary = {
            'instance_id': iid, 'initial_makespan': initial.ms, 'best_makespan': best.ms,
            'initial_hash': initial_hash, 'best_hash': best.state_hash,
            'completed_batches': batch + 1, 'attempted_steps': attempted_steps,
            'real_execution_calls': total_executions, 'elapsed_seconds': time.perf_counter()-started,
            'worker_load_seconds': LOAD_SECONDS, 'feasible': True, 'no_improvement_batches': stall,
            'in_training_set': iid in RUNTIME['meta']['train_ids'],
            'pool_makespans': [s.ms for s in pool], 'complete': False,
        }
        write_json(folder / f'batch_{batch+1:04d}.json', summary)
        write_json(folder / 'result.json', summary)
        print(f'[fast] {iid} batch={batch+1}/{cfg["batches"]} '
              f'best={best.ms} terminals={[r["terminal_makespan"] for r in branch_rows]} '
              f'seconds={summary["elapsed_seconds"]:.1f}', flush=True)
        if cfg['stall_batches'] and stall >= cfg['stall_batches']:
            break
    summary['complete'] = True
    summary['stop_reason'] = 'stall_budget' if batch + 1 < cfg['batches'] else 'batch_budget'
    write_json(folder / 'result.json', summary)
    assert schedule_hash(initial.schedule) == initial_hash
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, default=ROOT / 'inference_assets/runtime.pt')
    p.add_argument('--bank', type=Path, default=ROOT / 'data/drl_public41/daniel_sample100')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--instances', nargs='*')
    p.add_argument('--batches', type=int, default=10)
    p.add_argument('--branches', type=int, default=4)
    p.add_argument('--horizon', type=int, default=10)
    p.add_argument('--roots', type=int, default=6)
    p.add_argument('--pool-size', type=int, default=6)
    p.add_argument('--stall-batches', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--seed', type=int, default=922026)
    a = p.parse_args()
    if min(a.batches, a.branches, a.horizon, a.roots, a.workers) < 1 or a.pool_size < 2 or a.stall_batches < 0:
        p.error('Positive budgets and pool-size >= 2 required')
    manifest = json.loads((a.bank / 'protocol.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Incomplete bank')
    rows = [{'iid': r['instance_id'], 'path': str((a.bank / r['file']).resolve())}
            for r in manifest['entries'] if not a.instances or r['instance_id'] in a.instances]
    if not rows or (a.instances and set(a.instances) != {r['iid'] for r in rows}):
        raise ValueError('Requested instances missing')
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = vars(a).copy()
    cfg['output'] = str(a.output.resolve())
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}
    protocol.update(anchors=0, inference_only=True, pair_mode='one_related_companion_per_atom',
                    worse_intermediate_states=True, fixed_memory='shared_read_only',
                    device='cpu', worker_threads=1,
                    note='Modified search protocol, not the original paper baseline')
    write_json(a.output / 'protocol.json', protocol)
    start = time.perf_counter()
    tasks = [(row, cfg) for row in rows]
    if a.workers == 1:
        initialize(a.runtime, a.roots)
        results = [search_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(a.workers, len(rows)),
                                 mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize, initargs=(a.runtime, a.roots)) as executor:
            results = [f.result() for f in as_completed([executor.submit(search_one, t) for t in tasks])]
    write_json(a.output / 'summary.json', {'complete': True, 'results': results,
                                         'end_to_end_seconds': time.perf_counter()-start})


if __name__ == '__main__':
    main()
