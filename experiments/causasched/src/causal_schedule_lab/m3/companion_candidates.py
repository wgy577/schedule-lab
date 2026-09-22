"""Bounded target-wise pairs, reusable partners and state-specific exploration."""
import hashlib
import json
from collections import defaultdict
from . import config as C, proposal_features as PF
from causal_schedule_lab.validation import schedule_hash

def companion_pairs(pool, graph, partner_scan=8):
    cap=max(0,int(PF.T2L_PAIR_TOTAL_CAP))
    if not cap or partner_scan<=0:return []
    counts=getattr(C,'PAIR_BUILD_CONTEXT',{}).get('actions',{})
    neighbors=defaultdict(set)
    for op,preds in graph.predecessors.items():
        for pred in preds:neighbors[op].add(pred);neighbors[pred].add(op)
    machines=defaultdict(list)
    for interval in graph.intervals:machines[interval.machine_id].append(interval)
    for rows in machines.values():
        rows=sorted(rows,key=lambda r:(r.start,r.end,r.operation_id))
        for a,b in zip(rows,rows[1:]):
            neighbors[a.operation_id].add(b.operation_id);neighbors[b.operation_id].add(a.operation_id)
    touched=[PF._touched_operations(r['e']) for r in pool]
    by_op=defaultdict(set);sources=defaultdict(set);targets=defaultdict(set)
    for i,r in enumerate(pool):
        for op in touched[i]:by_op[op].add(i)
        if r['e'].edit_type==PF.EDIT_ROUTE:
            sources[r['e'].source_machine].add(i);targets[r['e'].target_machine].add(i)
    allowed={'ROUTE+ROUTE','ROUTE+SEQ_SWAP','ROUTE+SEQ_INSERT','SEQ_SWAP+SEQ_SWAP','SEQ_SWAP+SEQ_INSERT'}
    pairs=[];seen=set();checks=0
    per_target=max(1,int(partner_scan))*4
    total_budget=cap*max(1,int(partner_scan))
    for i in sorted(range(len(pool)),key=lambda k:(-pool[k]['uhat'],pool[k]['sig'])):
        if len(pairs)>=cap or checks>=total_budget:break
        u=pool[i];adjacent=set()
        for op in touched[i]:
            for neighbor in neighbors[op]:adjacent.update(by_op[neighbor])
        vacancy=set();shared=set()
        if u['e'].edit_type==PF.EDIT_ROUTE:
            vacancy=sources[u['e'].target_machine] | targets[u['e'].source_machine]
            shared=sources[u['e'].source_machine]
        def rank(j):
            a,b=u['sig'],pool[j]['sig']
            attempted=max(counts.get(a+'||'+b,0),counts.get(b+'||'+a,0))
            # A failed batch demotes tried pairs; structural signals precede prior.
            return (attempted>0,attempted,-int(j in vacancy),-int(j in adjacent),
                    -int(j in shared),-pool[j]['uhat'],pool[j]['sig'])
        tested=0
        for j in sorted((adjacent|vacancy|shared)-{i},key=rank):
            key=tuple(sorted((i,j)))
            if key in seen or touched[i]&touched[j]:continue
            v=pool[j]
            if PF._pair_family(u,v) not in allowed:continue
            if not PF._joint_operator_contract_compatible(u,v)[0]:continue
            if tested>=per_target or checks>=total_budget:break
            tested+=1;checks+=1
            if not PF.check_composite_structural_legality(graph,(u['e'],v['e'])).legal:continue
            pairs.append((i,j));seen.add(key);break
    return pairs

def cached_proposals(self,problem,schedule,iid):
    h=schedule_hash(schedule)
    context=getattr(C,'STATE_EXPLORATION',{}).get(h,{})
    digest=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()[:16]
    key=f'{iid}::{h}::companion-v17::{digest}'
    if key not in self._prop:
        previous=getattr(C,'PAIR_BUILD_CONTEXT',{})
        C.PAIR_BUILD_CONTEXT=context
        try:
            self._prop[key]=PF.build_proposal_features(
                self.ast(problem,schedule,iid),self.single_head,self.direct_head,fast_pairs=True)
        finally:C.PAIR_BUILD_CONTEXT=previous
        self._trim(self._prop)
    self._prop.move_to_end(key)
    result=self._prop[key]
    metas=result[1]
    diag=getattr(self,'pair_diagnostics',None)
    if diag is not None:
        diag['states']+=1
        diag['singles']+=sum(m['kind']=='single' for m in metas)
        diag['pairs']+=sum(m['kind']=='pair' for m in metas)
        diag['zero_pair_states']+=not any(m['kind']=='pair' for m in metas)
    return result
