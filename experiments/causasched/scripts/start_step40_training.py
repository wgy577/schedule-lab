"""Resume an explicitly selected 26-instance checkpoint with K=40; audit the cohort."""
import argparse
import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--additional-cycles', type=int, default=5000)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    os.chdir(ROOT)
    trainer = (ROOT/'scripts/train_e2e_single.py').read_text()
    if '# V22 ordered candidate-set training' not in trainer:
        raise RuntimeError('Install the V22 patch first')
    resume=a.resume.resolve()
    ck=torch.load(resume,map_location='cpu',weights_only=False)
    if not isinstance(ck,dict) or len(ck.get('arena',{}))!=26:
        raise RuntimeError('The selected checkpoint does not contain exactly 26 instances')
    cfg = ck['config']
    for key, required in (('branches',16), ('roots_per_cycle',26), ('episode_steps',200)):
        if int(cfg.get(key, -1)) != required:
            raise RuntimeError(f'{key}={cfg.get(key)} differs from requested {required}; '
                               'refusing to silently reset the episode')
    env = dict(os.environ)
    env['E2E_STEP_CANDIDATES'] = '40'
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'
    env['PYTHONPATH'] = str(ROOT/'src')+os.pathsep+str(ROOT/'scripts')+os.pathsep+env.get('PYTHONPATH','')
    args = [sys.executable, '-u', str(ROOT/'scripts/train_e2e_single.py'),
        '--resume', str(resume), '--output', str(a.output.resolve()),
        '--additional-cycles', str(a.additional_cycles), '--workers','16',
        '--branches','16', '--roots-per-cycle','26', '--episode-steps','200',
        '--perturb-after','0', '--epochs','1']
    for key in ('runtime','bank','device','decision_batch','trace_hops','root_top_k',
                'horizon','long_horizon','adaptive_after','long_every','parity_every',
                'lr','pretrained_lr','regression_weight','initialization','schedule_init',
                'seed','load_weight','load_share','perturb_max_gap'):
        if cfg.get(key) is not None:
            args += ['--'+key.replace('_','-'), str(cfg[key])]
    if cfg.get('instances'):
        args += ['--instances', *cfg['instances']]
    print('[resume]', resume, 'cycle=', ck['cycle'], flush=True)
    rows=[]
    for iid, state in sorted(ck['arena'].items()):
        problem=state['problem']
        payload=problem.model_dump(mode='json')
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        rows.append(dict(instance_id=iid,problem_sha256=digest,
            initial_makespan=int(state['initial'].ms),
            current_makespan=int(state.get('current',state['initial']).ms),
            best_makespan=int(state['best'].ms)))
    fingerprint=hashlib.sha256(json.dumps(
        [(s['instance_id'],s['problem_sha256']) for s in rows],separators=(',',':')).encode()).hexdigest()
    cohort=dict(source_checkpoint=str(resume),bank=str(cfg.get('bank')),
        schedule_init=cfg.get('schedule_init'),count=len(rows),
        instance_and_problem_sha256=fingerprint,entries=rows)
    print('[cohort] ids='+','.join(s['instance_id'] for s in rows),flush=True)
    print('[cohort] instance/problem fingerprint='+fingerprint,flush=True)
    print('[cohort] initial/current/best means='+ '/'.join(
        f'{sum(s[key] for s in rows)/len(rows):.3f}' for key in
        ('initial_makespan','current_makespan','best_makespan'))+
        '; schedule_init='+str(cfg.get('schedule_init')),flush=True)
    print('[V23] 26 instances; 16 workers x 16 trajectories; each step up to 40 distinct '
          'candidates; 200-step episodes; current weights/optimizer/reward retained', flush=True)
    print('[launch]', ' '.join(args), flush=True)
    if a.check_only:
        return
    a.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(resume=str(resume), previous_cycle=ck['cycle'],
        branches=16, workers=16, candidates_per_step=40, episode_steps=200,
        probability='ordered without-replacement candidate-set joint likelihood',
        winner='minimum feasible successor makespan; ties use draw order',
        reward='unchanged one-step credit of executed winner; not per-losing-trial rewards',
        permits_worse_successor=True, schedule_stagnation_restart=False,
        old_rollouts_reused=False)
    (a.output/'candidate_set_protocol.json').write_text(json.dumps(protocol, indent=2))
    (a.output/'resumed_cohort.json').write_text(json.dumps(cohort,indent=2))
    raise SystemExit(subprocess.call(args, env=env))


if __name__ == '__main__':
    main()
