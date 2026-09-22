"""Reproducible, outcome-blind sampling of 102 old TRAIN problems + fixed 26."""
import argparse, collections, hashlib, json, sys, tarfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from causal_schedule_lab.core_validation import validate_schedule
from causal_schedule_lab.ir import Problem, Schedule

def fingerprint(p):
    def clean(x):
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items() if k not in ('metadata','name','tags','provenance')}
        if isinstance(x,list): return [clean(v) for v in x]
        return x
    x=clean(p.model_dump(mode='json'));x.pop('id',None)
    return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--source-bank',type=Path,required=True,help='External original DRL bank')
    p.add_argument('--source-runtime-meta',type=Path,required=True,help='External original 26-ID runtime metadata')
    p.add_argument('--output',type=Path,default=ROOT/'data/train128');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    bank=a.source_bank
    ids=set(json.loads(a.source_runtime_meta.read_text())['train_ids'])
    entries=json.loads((bank/'protocol.json').read_text())['entries']
    chosen=[];seen=set()
    for row in entries:
        if row['instance_id'] not in ids:continue
        x=json.loads((bank/row['file']).read_text());problem=Problem.model_validate(x['problem'])
        chosen.append((row['instance_id'],problem,Schedule.model_validate(x['schedule']),'retained26'))
        seen.add(fingerprint(problem))
    assert len(chosen)==26
    with tarfile.open(a.archive) as tar:
        ck=torch.load(tar.extractfile('outputs/t2m_hierarchical_joint_fresh/p2/latest.pt'),map_location='cpu',weights_only=False)
    defs=ck['state']['frontier_state']['active_definitions'];assert len(defs)==200
    buckets=collections.defaultdict(list)
    for iid,episode,problem,source,schedule in defs:
        if iid in ids or fingerprint(problem) in seen:continue
        buckets[problem.kind].append((iid,problem,schedule,'old200_train_snapshot'))
    for rows in buckets.values():
        rows.sort(key=lambda row:hashlib.sha256(('train128-v1:'+row[0]).encode()).hexdigest())
    while len(chosen)<128:
        added=0
        for kind in sorted(buckets):
            while buckets[kind]:
                row=buckets[kind].pop(0);fp=fingerprint(row[1])
                if fp in seen:continue
                chosen.append(row);seen.add(fp);added+=1;break
            if len(chosen)==128:break
        if not added:raise ValueError('Insufficient unique old training problems')
    a.output.mkdir(parents=True)
    rows=[]
    for iid,problem,schedule,source in chosen:
        assert validate_schedule(problem,schedule).feasible,iid
        filename=hashlib.sha256(iid.encode()).hexdigest()[:16]+'.json'
        payload=dict(problem=problem.model_dump(mode='json'),schedule=schedule.model_dump(mode='json'))
        content=json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()
        (a.output/filename).write_bytes(content)
        rows.append(dict(instance_id=iid,file=filename,source=source,kind=problem.kind,
            jobs=len(problem.jobs),resources=len(problem.resources),operations=len(problem.operations),
            sha256=hashlib.sha256(content).hexdigest(),problem_fingerprint=fingerprint(problem)))
    manifest=dict(complete=True,purpose='TRAIN',cohort='fixed128-v1',retained26=sorted(ids),
        old200_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),
        sampling='kind-round-robin; SHA256(train128-v1:ID); no outcome ranking; structural deduplication',
        initial_schedule_rule='earliest_finish',entries=rows)
    (a.output/'protocol.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('TRAIN128',dict(collections.Counter(r['kind'] for r in rows)))

if __name__=='__main__':main()
