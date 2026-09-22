import collections
import json
import re
import sys
import tarfile
from pathlib import Path
import torch

archive = Path(sys.argv[1])
with tarfile.open(archive) as tar:
    checkpoint = torch.load(tar.extractfile('outputs/t2m_hierarchical_joint_fresh/p2/latest.pt'),
                            map_location='cpu', weights_only=False)
frontier = checkpoint['state']['frontier_state']
records = {}
snapshots = set()

def visit(value):
    if hasattr(value, 'records'):
        snapshots.add((value.schedule.problem_id, value.state_hash))
        for rec in value.records:
            key = tuple(rec.get(k) for k in ('instance_id','state_hash','proposal_signature','successor_state_hash'))
            records[key] = rec
    elif isinstance(value, dict):
        for item in value.values(): visit(item)
    elif isinstance(value, (list,tuple)):
        for item in value: visit(item)

visit(frontier['states'])
def atom(s):
    kind = s.split(':')[0]
    ops = re.findall(r'J\d+\.O\d+',s)
    if kind == 'ROUTE':
        src,dst = s.rsplit(':',1)[1].split('->')
        return dict(kind=kind,ops=ops,src=src,dst=dst,machines={src,dst})
    machine = s.split('|')[-1] if kind=='SEQ_SWAP' else s.split('@')[1].split(':')[0]
    # Insert neighbors describe the destination, not all edited operations.
    return dict(kind=kind,ops=ops if kind=='SEQ_SWAP' else ops[:1],machines={machine})

groups = collections.defaultdict(list)
examples = collections.defaultdict(list)
matched = 0
for rec in records.values():
    if rec.get('proposal_type')!='pair' or rec.get('true_U') is None: continue
    parts=rec['proposal_signature'].split('||')
    if len(parts)!=2: raise ValueError(parts)
    a,b=map(atom,parts)
    family='+'.join(sorted([a['kind'],b['kind']]))
    labels=[]
    if family=='ROUTE+ROUTE':
        if a['src']==b['src']: labels.append('shared_source')
        if a['dst']==b['dst']: labels.append('shared_target')
        if a['dst']==b['src'] or b['dst']==a['src']: labels.append('in_out_link')
    if a['machines'] & b['machines']: labels.append('shared_machine')
    else: labels.append('disjoint_machines')
    if {o.split('.')[0] for o in a['ops']} & {o.split('.')[0] for o in b['ops']}:
        labels.append('same_job')
    matched += (rec['instance_id'],rec['state_hash']) in snapshots
    for label in ['all']+labels:
        k=family+'/'+label
        groups[k].append(float(rec['true_U']))
        if rec['true_U']>0 and len(examples[k])<3:
            examples[k].append({k:rec[k] for k in ('instance_id','proposal_signature','true_U','state_hash')})
out=dict(checkpoint_meta={k:checkpoint['meta'][k] for k in ('train_graphs','optimizer_update')},
         scope='Deduplicated retained-state memory; NOT all sampled training actions. Relationship labels overlap. No isolated-action counterfactuals.',
         unique_records=len(records),pair_records_with_prestate_snapshot=matched,
         groups={k:dict(n=len(v),positive=sum(x>0 for x in v),negative=sum(x<0 for x in v),
                        mean_immediate_gain=sum(v)/len(v),examples=examples[k]) for k,v in sorted(groups.items())})
Path(sys.argv[2]).write_text(json.dumps(out,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in out.items() if k!='groups'},indent=2))
for k,v in out['groups'].items():
    print(k,'n=',v['n'],'positive=',v['positive'],'negative=',v['negative'],'mean=',round(v['mean_immediate_gain'],3))
