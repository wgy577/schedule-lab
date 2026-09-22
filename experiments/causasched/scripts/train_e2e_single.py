"""Single + bounded companion-pair end-to-end GRPO, batched backprop."""
import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import shutil
import pickle
import zlib
import contextlib
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import torch
import run_fast_inference as base
from run_fast_parallel import set_single_only
from causal_schedule_lab.m3 import joint_grpo as JG, config as C
from causal_schedule_lab.m3.end_to_end import configure, modules, recompute, clipped_loss, reset_rl_actors
from causal_schedule_lab.m3.fast_inference import FastAnalyzeCache
from causal_schedule_lab.m3.persistent_frontier_grpo import FrontierState
from causal_schedule_lab.ir import Problem, Schedule
from causal_schedule_lab.solvers.dispatching import solve_dispatching
from causal_schedule_lab.core_validation import validate_schedule
from causal_schedule_lab.validation import schedule_hash
from causal_schedule_lab.m3.stall_perturbation import choose_start, refresh_pool
from causal_schedule_lab.m3.state_exploration import remember_failed_batch
from causal_schedule_lab.m3.adaptive_search import restore_clock, begin_episode, rollout_horizon
from causal_schedule_lab.m3.run_logging import log as print, configure as configure_logging
from causal_schedule_lab.m3.trace_depth import configure_trace_depth, expand_trace_state, reset_expanded_optimizer_states

VERSION = None
CACHE = None


def enable_pairs():
    from causal_schedule_lab.m3 import proposal_features as pf
    for name,value in dict(T2L_PAIR_ROUTE_ATOMS=24,T2L_PAIR_SEQUENCE_ATOMS=24,
                           T2L_PAIRS_PER_FAMILY=32,T2L_PAIR_TOTAL_CAP=96).items():
        setattr(C,name,value); setattr(pf,name,value)
    C.T2L_BASE_PAIR_ACTIONS=64; C.T2L_PLATEAU_PAIR_ACTIONS=96


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    while True:
        try:
            with tmp.open('wb') as f:
                torch.save(value, f)
                f.flush(); os.fsync(f.fileno())
            tmp.replace(path)
            return
        except (RuntimeError, OSError) as exc:
            if isinstance(exc, RuntimeError) and not any(s in str(exc) for s in
                ('PytorchStreamWriter','file write failed','unexpected pos')):
                raise
            with contextlib.suppress(OSError): tmp.unlink(missing_ok=True)
            print(f'[storage] PAUSED save={path} error={exc}; previous checkpoint retained. '
                  'Free disk space/check quota; retry in 30s.',flush=True)
            time.sleep(30)


def weights(r):
    result = {k: {n: t.detach().cpu().clone() for n,t in m.state_dict().items()}
              for k,m in modules(r).items()}
    result['_progress'] = dict(m2=r['policy'].m2.alpha_fraction,
                               m3=r['policy'].m3.alpha_fraction)
    return result


def restore_progress(r, ws):
    for key,value in ws.get('_progress',{}).items():
        getattr(r['policy'],key).alpha_fraction=float(value)


def worker_init(runtime, roots, penalty, initialization, seed, trace_hops=6):
    global VERSION
    enable_pairs()
    base.initialize(runtime, roots)
    if initialization == 'sft':
        torch.manual_seed(seed); reset_rl_actors(base.RUNTIME)
    configure_trace_depth(base.RUNTIME,trace_hops)
    configure(base.RUNTIME, False)
    C.T2L_NET_REGRESSION_WEIGHT = penalty
    VERSION = None


def collect(task):
    global VERSION, CACHE
    version, weightfile, iid, problem, root, branch, cfg, destination = task
    r = base.RUNTIME
    C.E2E_LOAD_REWARD_ENABLED = cfg.get('load_weight', 0.0) > 0
    C.STATE_EXPLORATION=cfg.get('choice_history',{})
    if VERSION != version:
        ws = torch.load(weightfile, map_location='cpu', weights_only=True)
        for key,m in modules(r).items():
            m.load_state_dict(ws[key], strict=True)
        restore_progress(r,ws)
        CACHE = FastAnalyzeCache(r['model_b5'], r['single_head'], r['direct_head'],
                                 max_entries=3)
        VERSION = version
    if cfg['initialization'] == 'sft':
        r['policy'].m2.set_progress(version,cfg['cycles'])
        r['policy'].m3.set_progress(version,cfg['cycles'])
    CACHE.pair_diagnostics = dict(states=0,singles=0,pairs=0,zero_pair_states=0)
    memory = copy.deepcopy(r['memory'])
    for record in root.records:
        memory.add_executed(iid, int(record['written_at_step']), copy.deepcopy(record))
    start = time.perf_counter()
    with torch.no_grad():
        from causal_schedule_lab.m3.step20_training import collect_trajectory
        tr = collect_trajectory(
            r['policy'], r['scorer'], r['executor'], r['model_b5'], r['single_head'],
            r['direct_head'], problem, root.schedule, root.ms, iid, 900000000,
            memory, cfg['seed']+version*100003+branch*1009,
            branch, horizon=cfg['horizon'], step_offset=root.gstep,
            stop_on_negative=False, action_space='policy_sampled', analyze_cache=CACHE,
            allow_policy_stop=False, feasible_fallback=True, anchor_trajectories=0)
    assert all(not s['is_anchor'] for s in tr['steps'])
    from causal_schedule_lab.m3.one_step_credit import add_load_credit
    add_load_credit(tr, problem, root.schedule, cfg.get('load_weight', 0.0))
    # Bytes travel via the process pipe, not torch shared-memory handles or disk.
    payload=zlib.compress(pickle.dumps(tr,protocol=pickle.HIGHEST_PROTOCOL),level=1)
    outcomes=[]
    for step in tr['steps']:
        before=step.get('state_makespan_before')
        after=step.get('successor_makespan')
        outcomes.append(dict(
            step=step.get('step_idx'),state_hash=step.get('state_hash'),
            action=step.get('action_signature'),family=step.get('action_family'),
            pair=bool(step.get('selected_is_pair')),before=before,after=after,
            gain=None if before is None or after is None else before-after,
            relative_change=None if before is None or after is None else (after-before)/max(before,1),
            execution_reason=step.get('execution_reason')))
    return dict(iid=iid, branch=branch, payload=payload, reward=tr['reward'],
                pair_diagnostics=dict(CACHE.pair_diagnostics),
                action_outcomes=outcomes, load_reward=tr['load_reward'],
                makespan_reward=tr['makespan_reward'],load_metric=tr['load_metric'],
                load_cv_start=tr['load_cost_start'],load_cv_best=tr['load_cost_best'],
                load_cv_terminal=tr['load_cost_terminal'],
                terminal_regression=max(0,tr['final_ms']-tr['best_ms']),
                start=root.ms, best=tr['best_ms'], gain=root.ms-tr['best_ms'], best_step=tr['best_step'],
                pair_actions=sum(bool(s['selected_is_pair']) for s in tr['steps']),
                terminal=tr['final_ms'], stop_reason=tr['terminal'] or 'horizon_complete',steps=tr['n_steps'],
                seconds=time.perf_counter()-start)


def select_episode_incumbent(root, candidates, budget_step, cycle):
    # Root comes first so equal-quality moves do not displace the incumbent.
    chosen=min([root]+list(candidates),key=lambda st:st.ms)
    assert chosen.ms<=root.ms
    return FrontierState(chosen.schedule,chosen.ms,budget_step,
                         records=tuple(chosen.records),generation=cycle)


def parameters(r):
    # Deduplicate shared references. Zero gradients do not create fictitious updates.
    seen, rows = set(), []
    for prefix,m in modules(r).items():
        for name,p in m.named_parameters():
            if id(p) not in seen:
                seen.add(id(p)); rows.append((prefix+'.'+name,p))
    return rows


def update(r, groups, opt, named, device, epochs, check):
    update_started=time.perf_counter()
    parity = dict(m2_logp=0., m3_logp=0., features=0.)
    # Check *all* sampled decisions before any optimizer step, not just one.
    if check:
        with torch.no_grad():
            for g in groups:
                for tr in g:
                    for rec in tr['steps']:
                        p2,p3,old2,aux = recompute(r,rec,device)
                        parity['m2_logp'] = max(parity['m2_logp'],abs(float(p2)-old2))
                        parity['m3_logp'] = max(parity['m3_logp'],abs(float(p3)-rec['logp_old']))
                        parity['features'] = max(parity['features'],float((aux['F'].cpu()-rec['F_pool']).abs().max()))
        print('[replay parity] '+json.dumps(parity),flush=True)
        if max(parity['m2_logp'],parity['m3_logp']) > .005 or parity['features'] > .005:
            raise RuntimeError('Behavior replay mismatch; no update performed')
    parity_seconds=time.perf_counter()-update_started
    initial = {n:p.detach().clone() for n,p in named}
    grads = {n:0. for n,p in named}
    count = sum(len(tr['steps']) for g in groups for tr in g)
    informative2 = sum(any(s['inf2'] for s in t['steps']) for g in groups for t in g)
    informative3 = sum(any(s['inf3'] for s in t['steps']) for g in groups for t in g)
    losses = []
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        processed = 0
        print(f'[update] epoch={epoch+1}/{epochs} START decisions={count}', flush=True)
        if not (informative2 or informative3):
            # No KL-only updates, and no Adam momentum drift on empty signal.
            losses.append(0.)
            break
        opt.zero_grad(set_to_none=True)
        loss_terms = []
        for g in groups:
            for tr in g:
                n2 = sum(s['inf2'] for s in tr['steps'])
                n3 = sum(s['inf3'] for s in tr['steps'])
                for rec in tr['steps']:
                    p2,p3,old2,aux = recompute(r,rec,device)
                    l2 = sum(clipped_loss(lp,old,float(rec['adv2']))
                             for lp,old in aux['draws'])/max(len(aux['draws']),1)
                    l3 = clipped_loss(p3,rec['logp_old'],float(rec['adv3']))
                    loss = (l2*bool(rec['inf2'])/max(n2*informative2,1) +
                            l3*bool(rec['inf3'])/max(n3*informative3,1))
                    loss.backward()
                    loss_terms.append(loss.detach())
                    processed += 1
                    if processed == 1 or processed % 10 == 0 or processed == count:
                        if str(device).startswith('cuda'):
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter()-epoch_start
                        gpu = (f' allocated_GB={torch.cuda.memory_allocated()/1e9:.2f}'
                               if str(device).startswith('cuda') else '')
                        print(f'[update] epoch={epoch+1}/{epochs} decisions={processed}/{count}'
                              f' seconds={elapsed:.1f} seconds_per_decision={elapsed/processed:.3f}'
                              f'{gpu}', flush=True)
        total=float(torch.stack(loss_terms).sum()) if loss_terms else 0.
        if not __import__('math').isfinite(total):
            raise RuntimeError('Nonfinite RL loss; optimizer not stepped')
        norms=torch.stack([p.grad.detach().norm() if p.grad is not None else
                           p.new_zeros(()) for n,p in named]).cpu().tolist()
        for (n,p),norm in zip(named,norms):
            grads[n] = max(grads[n],norm)
        torch.nn.utils.clip_grad_norm_([p for n,p in named],1.,error_if_nonfinite=True)
        opt.step()
        losses.append(total)
        print(f'[update] epoch={epoch+1}/{epochs} DONE seconds={time.perf_counter()-epoch_start:.1f}',flush=True)
    changes=torch.stack([(p.detach()-initial[n]).abs().max()
                          for n,p in named]).cpu().tolist()
    audit = {n:dict(grad_norm=grads[n],max_weight_change=delta)
             for (n,p),delta in zip(named,changes)}
    return dict(loss=losses, decisions=count, parity=parity if check else None,
                parity_checked=check,parity_seconds=parity_seconds,
                backward_and_audit_seconds=time.perf_counter()-update_started-parity_seconds,
                parameters=audit,
                nonzero_advantages=sum(bool(rec['adv2'] or rec['adv3'])
                   for g in groups for tr in g for rec in tr['steps']))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',type=Path,default=ROOT/'inference_assets/runtime.pt')
    p.add_argument('--bank',type=Path,default=ROOT/'data/train128')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--branches',type=int,default=16)
    p.add_argument('--roots-per-cycle',type=int,default=26)
    p.add_argument('--load-weight',type=float,default=0.2)
    p.add_argument('--load-share',type=float,default=0.30)
    p.add_argument('--decision-batch',type=int,default=16)
    p.add_argument('--trace-hops',type=int,choices=(4,6),default=6)
    p.add_argument('--root-top-k',type=int,default=6)
    p.add_argument('--horizon',type=int,default=10)
    p.add_argument('--episode-steps',type=int,default=200)
    p.add_argument('--expand-training-cohort',action='store_true')
    p.add_argument('--verbose',action='store_true')
    p.add_argument('--perturb-after',type=int,default=3)
    p.add_argument('--perturb-max-gap',type=float,default=.10)
    p.add_argument('--long-horizon',type=int,default=20)
    p.add_argument('--adaptive-after',type=int,default=2,
                   help='Use long-horizon after this many non-improving batches; 0 disables')
    p.add_argument('--long-every',type=int,default=5)
    p.add_argument('--cycles',type=int,default=100)
    p.add_argument('--additional-cycles',type=int,
                   help='Run this many update chunks after the loaded checkpoint')
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--parity-every',type=int,default=10,
                   help='Full behavior replay check on first/resumed cycle and every N cycles; 1=always')
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--pretrained-lr',type=float,default=1e-5)
    p.add_argument('--regression-weight',type=float,default=0.2)
    p.add_argument('--initialization',choices=['sft','u356'],default='sft')
    p.add_argument('--schedule-init',choices=['earliest_finish','spt','balanced','drl'],
                   default='earliest_finish')
    p.add_argument('--seed',type=int,default=9162026)
    p.add_argument('--instances',nargs='*',help='Training-bank subset for smoke test only')
    p.add_argument('--resume',type=Path)
    p.add_argument('--allow-reward-change',action='store_true')
    p.add_argument('--reset-episode-on-resume',action='store_true',
                   help='Keep learned weights/optimizer, restart the episode from rule schedules')
    a=p.parse_args()
    a.perturb_after=0  # Disable stagnation schedule restarts; episode resets unchanged.
    from causal_schedule_lab.m3.step20_training import candidate_count
    a.step_candidates=candidate_count()
    print(f"[candidate-set] TRAIN branches={a.branches} workers={a.workers} "
          f"candidates/step={a.step_candidates}; joint ordered-draw likelihood; "
          "reward unchanged; best archive retained",flush=True)
    if min(a.workers,a.roots_per_cycle,a.horizon,a.cycles,a.epochs)<1 or a.branches<2:
        p.error('Positive budgets and >=2 trajectories required')
    if a.episode_steps < a.horizon or a.episode_steps % a.horizon:
        p.error('episode-steps must be a positive multiple of horizon')
    if a.adaptive_after<0 or a.long_horizon<a.horizon:
        p.error('adaptive-after must be nonnegative and long-horizon >= horizon')
    if a.perturb_after<0 or not 0<=a.perturb_max_gap<=1:
        p.error('Invalid perturbation settings')
    a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=bool(a.resume))
    configure_logging(a.output,a.verbose)
    print(f'[storage] output={a.output} trajectory_transport=compressed-memory-pipe '
          f'free_GB={shutil.disk_usage(a.output).free/(1024**3):.2f}',flush=True)
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    enable_pairs()
    C.T2L_NET_REGRESSION_WEIGHT=a.regression_weight
    if a.decision_batch<1: p.error('decision-batch must be positive')
    base.initialize(a.runtime,a.root_top_k)
    r=base.RUNTIME
    if a.initialization=='sft':
        reset_rl_actors(r)
        print('[E2E-init] discarded old RL actor weights; retained SFT initialization; NEW RL optimizer',flush=True)
    configure_trace_depth(r,a.trace_hops)
    print(f'[trace] candidate_hops={a.trace_hops} graph_hops={a.trace_hops} propagation_hops={a.trace_hops}',flush=True)
    configure(r,True)
    for m in modules(r).values(): m.to(a.device)
    named=parameters(r)
    pretrained=[p for n,p in named if not n.startswith(('target.','operator.')) or '.r6_view.' in n]
    residual=[p for n,p in named if n.startswith(('target.','operator.')) and '.r6_view.' not in n]
    opt=torch.optim.AdamW([dict(params=pretrained,lr=a.pretrained_lr),
                           dict(params=residual,lr=a.lr)],weight_decay=0.)
    manifest=json.loads((a.bank/'protocol.json').read_text())
    original_ids=set(r['meta']['train_ids'])
    if manifest.get('cohort')=='fixed128-v1':
        if manifest.get('purpose')!='TRAIN' or len(manifest['entries'])!=128:
            raise ValueError('Invalid TRAIN128 manifest')
        if set(manifest['retained26'])!=original_ids:
            raise ValueError('Original 26-instance cohort mismatch')
        train_ids={row['instance_id'] for row in manifest['entries']}
        if len(train_ids)!=128 or not original_ids<=train_ids:raise ValueError('Invalid expanded IDs')
    else:
        train_ids=original_ids
    requested=set(a.instances or train_ids)
    if not requested <= train_ids: raise ValueError('Instances must belong to the approved TRAIN bank')
    arena={}
    initial_rows=[]
    (a.output/'initial_schedules').mkdir(exist_ok=True)
    for row in manifest['entries']:
        iid=row['instance_id']
        if iid not in requested: continue
        data=(a.bank/row['file']).read_bytes()
        if row.get('sha256'):
            import hashlib
            if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Bank checksum mismatch: '+iid)
        x=json.loads(data)
        problem=Problem.model_validate(x['problem'])
        drl=Schedule.model_validate(x['schedule'])
        sch=(drl if a.schedule_init=='drl' else
             solve_dispatching(problem,rule=a.schedule_init))
        if not validate_schedule(problem,sch).feasible:
            raise ValueError('Invalid initial schedule: '+iid)
        row=dict(instance_id=iid,rule=a.schedule_init,
                 initial_makespan=int(sch.makespan),bank_makespan=int(drl.makespan),
                 schedule_hash=schedule_hash(sch))
        initial_rows.append(row)
        base.write_json(a.output/'initial_schedules'/f'{iid}.json',
                        dict(problem=x['problem'],schedule=sch.model_dump(mode='json'),**row))
        initial=FrontierState(sch,int(sch.makespan))
        arena[iid]=dict(problem=problem,initial=initial,
                        best=initial,pool=[initial])
    if set(arena)!=requested: raise ValueError('Missing bank instances')
    if not 0 <= a.load_weight <= 1: p.error('load-weight must be in [0,1]')
    if not 0 <= a.load_share < 1: p.error('load-share must be in [0,1)')
    a.roots_per_cycle = len(arena)
    base.write_json(a.output/'initial_manifest.json',dict(
        rule=a.schedule_init,count=len(initial_rows),entries=initial_rows))
    print('[initial schedules] '+json.dumps(dict(rule=a.schedule_init,count=len(initial_rows),
        mean_initial=sum(row['initial_makespan'] for row in initial_rows)/len(initial_rows),
        mean_bank=sum(row['bank_makespan'] for row in initial_rows)/len(initial_rows))),flush=True)
    start_cycle=0
    episode_origin_cycle=0
    ck=None
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=False)
        from causal_schedule_lab.m3.load_reward import LOAD_METRIC
        if ck['config'].get('load_metric')!=LOAD_METRIC:
            if not a.allow_reward_change:
                raise ValueError('Load metric changed to global CV: pass --allow-reward-change')
            base.write_json(a.output/'load_metric_migration.json',dict(
                old=ck['config'].get('load_metric',ck['config'].get('load_cost')),
                new=LOAD_METRIC,share=a.load_share,checkpoint=str(a.resume)))
            print(f'[reward migration] global load CV; target share={a.load_share:.0%}; fresh rollouts only',flush=True)
        if ck['config'].get('load_weight',0.0) != a.load_weight and not a.allow_reward_change:
            raise ValueError('Load reward changed: pass --allow-reward-change')
        if ck['config'].get('load_share',0.0) != a.load_share and not a.allow_reward_change:
            raise ValueError('Load reward share changed: pass --allow-reward-change')
        if ck['config'].get('episode_cohort') != 'all-instances-synchronized':
            if not a.reset_episode_on_resume:
                raise ValueError('Migrating to all-instance episodes requires --reset-episode-on-resume')
        previous_reward_weight = ck['config']['regression_weight']
        if previous_reward_weight != a.regression_weight:
            if not a.allow_reward_change:
                raise ValueError('Reward changed on resume: pass --allow-reward-change explicitly')
            print(f'[reward migration] lambda={previous_reward_weight}->{a.regression_weight}; '
                  'weights/optimizer/frontier retained; fresh rollouts use new reward',flush=True)
            base.write_json(a.output/'reward_migration.json',dict(
                checkpoint=str(a.resume),checkpoint_cycle=ck['cycle'],
                previous_lambda=previous_reward_weight,new_lambda=a.regression_weight))
        for key in ('episode_steps','horizon','roots_per_cycle','branches'):
            if key in ('episode_steps','roots_per_cycle') and a.reset_episode_on_resume:
                continue
            if ck['config'].get(key)!=getattr(a,key):
                raise ValueError('Episode protocol changed on resume: '+key)
        if ck['config']['initialization']!=a.initialization: raise ValueError('Initialization changed on resume')
        if ck['config'].get('schedule_init','drl')!=a.schedule_init:
            raise ValueError('Cannot resume an old DRL frontier into rule-initialized training')
        previous_arena=ck['arena']
        if set(previous_arena)!=set(arena):
            if not (a.expand_training_cohort and a.reset_episode_on_resume and set(previous_arena)<set(arena)):
                raise ValueError('Expanded cohort requires --expand-training-cohort --reset-episode-on-resume')
            from build_train128 import fingerprint
            for iid in previous_arena:
                if fingerprint(previous_arena[iid]['problem'])!=fingerprint(arena[iid]['problem']):
                    raise ValueError('Existing training problem changed: '+iid)
            arena.update(previous_arena)
            print(f'[cohort] expanded {len(previous_arena)}->{len(arena)}; new instances start from fixed rule schedules',flush=True)
            base.write_json(a.output/'cohort_migration.json',dict(old_ids=sorted(previous_arena),new_ids=sorted(arena)))
        else:
            arena=previous_arena
        old_hops=int(ck['weights']['target']['probability_trace.hop_embedding.weight'].shape[0])
        migrated_target,resized=expand_trace_state(ck['weights']['target'],a.trace_hops)
        for k,m in modules(r).items():
            m.load_state_dict(migrated_target if k=='target' else ck['weights'][k],strict=True)
        restore_progress(r,ck['weights'])
        opt.load_state_dict(ck['optimizer']); start_cycle=ck['cycle']
        changed_names=['target.'+name for name in resized]
        reset_expanded_optimizer_states(opt,named,changed_names)
        base.write_json(a.output/'trace_migration.json',dict(checkpoint=str(a.resume),
            old_hops=old_hops,new_hops=a.trace_hops,resized_parameters=changed_names,
            optimizer_reset_only=changed_names,old_rows_preserved=True))
        print(f'[trace migration] L={old_hops}->{a.trace_hops}; resized={len(resized)} tensors; '
              'other weights/optimizer states retained',flush=True)
        episode_origin_cycle=ck['config'].get('episode_origin_cycle',0)
        print(f'[resume] restored checkpoint={a.resume} cycle={start_cycle}; '
              f'weights/optimizer/frontier loaded; episode_origin_cycle={episode_origin_cycle}',flush=True)
        if ck['config'].get('episode_transition')!='episode-best-including-start':
            if not a.reset_episode_on_resume:
                raise ValueError('Old terminal-continuation checkpoint: pass --reset-episode-on-resume')
        if a.reset_episode_on_resume:
            episode_origin_cycle=start_cycle
            print('[resume] weights/optimizer retained; new fixed-length episode from rule schedules',flush=True)
    if a.additional_cycles is not None:
        if a.additional_cycles<1: p.error('additional-cycles must be positive')
        a.cycles=start_cycle+a.additional_cycles
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(single_only=False,anchors=0,train_ids=sorted(arena),
                  weight_source=(f'resumed weights and optimizer from {a.resume}' if a.resume else
                                 ('SFT initialization; fresh RL actors and optimizer' if a.initialization=='sft'
                                  else 'u356 warm start; fresh optimizer')),
                  reward='start-best - lambda*(terminal-best) - infeasible_penalty + group_calibrated_load_change',
                  load_cost='population_std(all_resource_processing_loads)/mean(all_resource_processing_loads)',
                  load_metric='global_processing_load_cv_v1',raw_load_reward_cap_fraction=0.05,
                  episode_cohort='all-instances-synchronized',
                  memory='prepared TRAIN retrieval bank; branch-local writes',
                  inactive='STOP output; unused legacy auxiliary heads',
                  candidate_support='fixed during PPO epochs; recomputed for new rollouts',
                  dropout=False,kl_reference=False,training_variant='e2e-dispatch-v14-global-load30',
                  restart_policy='incumbent-origin-pool-only; no chained restarts',
                  budget_semantics='per-instance allocated rollout steps; early termination consumes allotted budget',
                  pair_matching='bounded-relation-count-v1; not counterfactually validated',
                  episode_transition='episode-best-including-start',episode_origin_cycle=episode_origin_cycle,
                  beta_m2=C.TO1_R14_BETA_M2,beta_m3=C.TO1_R14_BETA_M3,
                  m2_surrogate='same per-draw clipping/trajectory normalization as GPU trainer')
    base.write_json(a.output/'config.json',config)
    print('[E2E] '+json.dumps(config,ensure_ascii=False),flush=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(a.output/'tensorboard'),purge_step=start_cycle+1 if a.resume else None)
    except ImportError:
        print('[warning] tensorboard not installed; audit JSON and rollout JSONL remain available',flush=True)
        class NullWriter:
            def add_scalar(self,*args,**kwargs): pass
            def flush(self): pass
            def close(self): pass
        writer=NullWriter()
    ids=sorted(arena)
    clock=restore_clock(ck,ids,a.roots_per_cycle,a.episode_steps,a.horizon,
                        reset=a.reset_episode_on_resume)
    if a.resume:
        print('[episode-clock] restored '+json.dumps(clock),flush=True)
    try:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),
               initializer=worker_init,initargs=(a.runtime,a.root_top_k,a.regression_weight,a.initialization,a.seed,a.trace_hops)) as ex:
            for cycle in range(start_cycle+1,a.cycles+1):
                started=time.perf_counter()
                if a.initialization=='sft':
                    r['policy'].m2.set_progress(cycle,a.cycles)
                    r['policy'].m3.set_progress(cycle,a.cycles)
                new_episode=begin_episode(clock,ids,a.roots_per_cycle,a.episode_steps)
                episode=clock['episode']; chunk=clock['updates']
                selected=[iid for iid,used in clock['budgets'].items() if used<a.episode_steps]
                if new_episode:
                    for iid in selected:
                        arena[iid]['current']=arena[iid]['initial']
                        arena[iid]['pool']=[arena[iid]['initial']]
                        arena[iid]['stall_segments']=0
                        arena[iid]['perturb_pool']=[]
                        arena[iid]['perturb_history']=[]
                        arena[iid]['perturb_anchor_hash']=arena[iid]['initial'].state_hash
                        arena[iid]['last_perturb_stall']=0
                        arena[iid]['choice_history']={}
                    print(f'[episode] {episode+1} START reset_to_initial instances={len(selected)} budget={a.episode_steps}',flush=True)
                horizons={iid:rollout_horizon(arena[iid],clock['budgets'][iid],a.episode_steps,
                          a.horizon,a.long_horizon,a.adaptive_after) for iid in selected}
                if len(selected) != len(ids) or len(set(clock['budgets'].values())) != 1:
                    raise RuntimeError('All-instance synchronized budget violated')
                # One common depth per update: all instances finish/reset together.
                common_horizon=max(horizons.values())
                horizons={iid:common_horizon for iid in selected}
                print(f'[episode] {episode+1} update={chunk+1} '
                      f'budget_min/max={min(clock["budgets"].values())}/{max(clock["budgets"].values())}'
                      f'/{a.episode_steps} horizons='+json.dumps(horizons),flush=True)
                weightfile=a.output/'behavior_weights.pt'
                atomic_save(weights(r),weightfile)
                with contextlib.nullcontext():
                    futures={}; starts={}; perturbed={}
                    for iid in selected:
                        state=arena[iid]
                        horizon=horizons[iid]
                        root,perturbed[iid]=choose_start(state,a.perturb_after,a.perturb_max_gap)
                        root=FrontierState(root.schedule,root.ms,clock['budgets'][iid],
                            records=tuple(root.records),generation=cycle)
                        if perturbed[iid]:
                            print(f'[perturb] {iid} incumbent={state["current"].ms} '
                                  f'explore_start={root.ms} origin=incumbent incumbent_preserved=True horizon={horizon}',flush=True)
                        elif (a.perturb_after>0 and
                              state.get('stall_segments',0)-state.get('last_perturb_stall',0)>=a.perturb_after and
                              not state.get('perturb_pool')):
                            print(f'[perturb] {iid} no bounded incumbent-origin candidate; keeping best',flush=True)
                        starts[iid]=root
                        if state.get('choice_history'):
                            print(f'[choice-perturb] {iid} state_histories={len(state["choice_history"])} roots+actions=True',flush=True)
                        for b in range(a.branches):
                            dest=None
                            task=(cycle,str(weightfile),iid,state['problem'],root,b,
                                  dict(seed=a.seed+ids.index(iid)*10000019,horizon=horizon,
                                       cycles=a.cycles,initialization=a.initialization,load_weight=a.load_weight,
                                       choice_history=state.get('choice_history',{})),dest)
                            futures[ex.submit(collect,task)]=(iid,b)
                    paths={iid:[] for iid in selected}
                    with (a.output/'rollout_metrics.jsonl').open('a') as f:
                        for future in as_completed(futures):
                            row=future.result(); paths[row['iid']].append(row)
                            row['incumbent_before']=arena[row['iid']]['current'].ms
                            row['incumbent_gain']=max(0,row['incumbent_before']-row['best'])
                            futures.pop(future)
                            f.write(json.dumps({**{k:v for k,v in row.items() if k!='payload'},'cycle':cycle})+'\n');f.flush();os.fsync(f.fileno())
                            print(f'[collect] cycle={cycle} {sum(map(len,paths.values()))}/{len(selected)*a.branches} '
                                  f'{row["iid"]} steps={row["steps"]} start={row["start"]} best={row["best"]} '
                                  f'gain={row["gain"]:+} best_step={row["best_step"]} pairs={row["pair_actions"]} '
                                  f'end={row["terminal"]} regression={row["terminal_regression"]} provisional_reward={row["reward"]:+.3f} '
                                  f'makespan_reward={row["makespan_reward"]:+.3f} raw_load_reward={row["load_reward"]:+.3f} '
                                  f'incumbent={row["incumbent_before"]} incumbent_gain={row["incumbent_gain"]:+}',flush=True)
                    collected_at=time.perf_counter()
                    groups=[]
                    for iid in selected:
                        trs=[pickle.loads(zlib.decompress(x.pop('payload')))
                             for x in sorted(paths[iid],key=lambda x:x['branch'])]
                        from causal_schedule_lab.m3.step20_training import report as report_pair_selection
                        report_pair_selection(trs,iid,cycle,a.output)
                        root=starts[iid]
                        ds=[x['pair_diagnostics'] for x in paths[iid]]
                        ns=sum(d['states'] for d in ds)
                        print(f'[pair-candidates] {iid} states={ns} single_mean={sum(d["singles"] for d in ds)/max(ns,1):.1f} pair_mean={sum(d["pairs"] for d in ds)/max(ns,1):.1f} zero_pair_states={sum(d["zero_pair_states"] for d in ds)} selected_pairs={sum(x["pair_actions"] for x in paths[iid])}',flush=True)
                        from causal_schedule_lab.m3.load_reward import balance_group
                        load_diag=balance_group(trs,a.load_share)
                        print(f'[reward-mix] {iid} load_share={load_diag["actual_share"]:.1%} '
                              f'main_abs={load_diag["main_abs"]:.3f} load_abs={load_diag["load_abs"]:.3f}',flush=True)
                        with (a.output/'reward_components.jsonl').open('a') as f:
                            f.write(json.dumps(dict(cycle=cycle,iid=iid,**load_diag,
                                trajectories=[dict(branch=t['traj_id'],main=t['makespan_reward'],
                                    load=t['load_reward'],total=t['reward'],
                                    load_cv_start=t['load_cost_start'],load_cv_best=t['load_cost_best'],
                                    load_cv_terminal=t['load_cost_terminal']) for t in trs]))+'\n')
                            f.flush();os.fsync(f.fileno())
                        from causal_schedule_lab.m3.one_step_credit import finish_group as finish_one_step_group
                        finish_one_step_group(trs,iid,root.state_hash,root.ms,r['memory'],
                                             time.perf_counter()-started,a.workers)
                        groups.append(trs)
                        state=arena[iid]; incumbent=state['current']; candidates=[incumbent]
                        for tr in trs:
                            for prefix in ('best','terminal'):
                                sch=tr[prefix+'_schedule']
                                steps=tr['best_step'] if prefix=='best' else tr['n_steps']
                                candidates.append(FrontierState(sch,int(sch.makespan),root.gstep+steps,
                                    records=tuple(root.records)+tuple(tr[prefix+'_memory_records']),generation=cycle))
                        state['best']=min(candidates+[state['best']],key=lambda st:st.ms)
                        clock['budgets'][iid]+=horizons[iid]
                        state['current']=select_episode_incumbent(
                            incumbent,candidates,clock['budgets'][iid],cycle)
                        improved=state['current'].ms<incumbent.ms
                        remember_failed_batch(state,trs,improved)
                        state['stall_segments']=0 if improved else state.get('stall_segments',0)+1
                        if improved:
                            state['last_perturb_stall']=0
                        refresh_pool(state,candidates,a.perturb_max_gap,source_hash=root.state_hash)
                        state['pool']=[state['current']]
                        print(f'[continue] {iid} previous={incumbent.ms} next={state["current"].ms} '
                              f'episode_gain={state["initial"].ms-state["current"].ms:+}',flush=True)
                    update_at=time.perf_counter()
                    print(f'[update] cycle={cycle} START collect_seconds={collected_at-started:.1f} '
                          f'prepare_seconds={update_at-collected_at:.1f}',flush=True)
                    from causal_schedule_lab.m3.e2e_update import update as batched_update
                    try:
                        audit=batched_update(r,groups,opt,named,a.device,a.epochs,
                            check=cycle==start_cycle+1 or (a.parity_every>0 and cycle%a.parity_every==0),
                            decision_batch=a.decision_batch)
                    except RuntimeError as exc:
                        if hasattr(exc,'parity_payload'):
                            diagnostic=a.output/'parity_failure.pt'
                            atomic_save(exc.parity_payload,diagnostic)
                            print(f'[parity failure saved] {diagnostic}',flush=True)
                        raise
                    save_at=time.perf_counter()
                    audit['phase_seconds']=dict(collect=collected_at-started,
                        prepare=update_at-collected_at,update=save_at-update_at)
                    clock['updates']+=1
                    audit.update(cycle=cycle,episode=episode+1,episode_chunk=clock['updates'],
                        perturbations=sum(perturbed.values()),
                        horizons=horizons,episode_budget_steps_by_instance=dict(clock['budgets']),
                        episode_budget_steps=min(clock['budgets'].values()),seconds=time.perf_counter()-started,
                        mean_best=sum(s['best'].ms for s in arena.values())/len(arena),
                        mean_reward=sum(t['reward'] for g in groups for t in g)/sum(map(len,groups)),
                        mean_load_reward=sum(t['load_reward'] for g in groups for t in g)/sum(map(len,groups)),
                        mean_makespan_reward=sum(t['makespan_reward'] for g in groups for t in g)/sum(map(len,groups)),
                        mean_terminal=sum(t['final_ms'] for g in groups for t in g)/sum(map(len,groups)),
                        mean_terminal_relative_gain=sum((starts[iid].ms-t['final_ms'])/max(starts[iid].ms,1)
                            for iid,g in zip(selected,groups) for t in g)/sum(map(len,groups)))
                    base.write_json(a.output/f'audit_{cycle:04d}.json',audit)
                    ck=dict(format='causasched-e2e-single-v1',cycle=cycle,weights=weights(r),
                            optimizer=opt.state_dict(),arena=arena,config=config,
                            episode_clock=copy.deepcopy(clock))
                    atomic_save(ck,a.output/'latest.pt')
                    if cycle%10==0:
                        atomic_save(dict(weights=ck['weights'],config=config,cycle=cycle),
                                    a.output/f'policy_{cycle:04d}.pt')
                    audit['phase_seconds']['save']=time.perf_counter()-save_at
                    base.write_json(a.output/f'audit_{cycle:04d}.json',audit)
                    for phase,seconds in audit['phase_seconds'].items():
                        writer.add_scalar('time/'+phase,seconds,cycle)
                    print('[timing] '+json.dumps(audit['phase_seconds']),flush=True)
                    for key in ('mean_best','mean_reward','mean_load_reward','mean_makespan_reward','mean_terminal','mean_terminal_relative_gain',
                                'seconds','nonzero_advantages'):
                        writer.add_scalar('train/'+key,audit[key],cycle)
                    writer.add_scalar('search/mean_horizon',sum(horizons.values())/len(horizons),cycle)
                    writer.add_scalar('search/perturbed_instances',sum(perturbed.values()),cycle)
                    writer.add_scalar('search/episode_budget_min',min(clock['budgets'].values()),cycle)
                    writer.add_scalar('search/episode_budget_max',max(clock['budgets'].values()),cycle)
                    for prefix in modules(r):
                        rows=[v for n,v in audit['parameters'].items() if n.startswith(prefix+'.')]
                        writer.add_scalar('gradient/'+prefix,max(v['grad_norm'] for v in rows),cycle)
                        writer.add_scalar('weight_change/'+prefix,max(v['max_weight_change'] for v in rows),cycle)
                    writer.flush()
                    from collections import Counter
                    stop_counts=dict(Counter(row['stop_reason'] for rows in paths.values() for row in rows))
                    total_rows=sum(map(len,paths.values()))
                    improved_rows=sum(row['incumbent_gain']>0 for rows in paths.values() for row in rows)
                    print(f'[summary] cycle={cycle} episode={episode+1} instances={len(selected)} '
                          f'budget={min(clock["budgets"].values())}/{a.episode_steps} trajectories={total_rows} '
                          f'improving={improved_rows}/{total_rows} best_mean={audit["mean_best"]:.3f} '
                          f'main_reward={audit["mean_makespan_reward"]:+.3f} load_reward={audit["mean_load_reward"]:+.3f} '
                          f'sec={audit["seconds"]:.1f} stops={stop_counts}',flush=True)
                    print(f'[saved] cycle={cycle} mean_best={audit["mean_best"]:.3f} '
                          f'checkpoint={a.output/"latest.pt"}',flush=True)
                    del groups,ck
    finally:
        writer.close()


if __name__=='__main__':
    main()

# V22 ordered candidate-set training
