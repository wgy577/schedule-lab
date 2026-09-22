"""Inference-only fast candidates and immutable-base/branch-local Memory."""
import copy
from collections import defaultdict

from . import proposal_features as PF
from .memory import ProgressiveMemory
from causal_schedule_lab.validation import schedule_hash


def companion_pairs(pool, graph, partner_scan=8):
    """A matching: each atom participates in at most one compatible pair.

    Candidate relations: dependency/machine adjacency, shared route source,
    and bidirectional route in/out links. Rank by number of relation signals,
    then existing single prior; no relation type is claimed empirically best.
    Unrelated atoms are not forced into pairs. This is a structural heuristic,
    not a counterfactual estimate or proof of improvement.
    """
    if PF.T2L_PAIR_TOTAL_CAP <= 0 or partner_scan <= 0:
        return []
    neighbors = defaultdict(set)
    for op, preds in graph.predecessors.items():
        for pred in preds:
            neighbors[op].add(pred)
            neighbors[pred].add(op)
    machines = defaultdict(list)
    for interval in graph.intervals:
        machines[interval.machine_id].append(interval)
    for intervals in machines.values():
        ordered = sorted(intervals, key=lambda r: (r.start, r.end, r.operation_id))
        for left, right in zip(ordered, ordered[1:]):
            neighbors[left.operation_id].add(right.operation_id)
            neighbors[right.operation_id].add(left.operation_id)
    by_op = defaultdict(set)
    route_sources = defaultdict(set)
    route_targets = defaultdict(set)
    for i, rec in enumerate(pool):
        for op in PF._touched_operations(rec['e']):
            by_op[op].add(i)
        if rec['e'].edit_type == PF.EDIT_ROUTE:
            route_sources[rec['e'].source_machine].add(i)
            route_targets[rec['e'].target_machine].add(i)
    rank = sorted(range(len(pool)), key=lambda i: (-pool[i]['uhat'], pool[i]['sig']))
    used, pairs = set(), []
    allowed = {'ROUTE+ROUTE', 'ROUTE+SEQ_SWAP', 'ROUTE+SEQ_INSERT',
               'SEQ_SWAP+SEQ_SWAP', 'SEQ_SWAP+SEQ_INSERT'}
    for i in rank:
        if len(pairs) >= int(PF.T2L_PAIR_TOTAL_CAP):
            break
        if i in used:
            continue
        u = pool[i]
        touched = PF._touched_operations(u['e'])
        related = set()
        for op in touched:
            for neighbor in neighbors[op]:
                related.update(by_op[neighbor])
        adjacent = set(related)
        vacancy, shared_source = set(), set()
        if u['e'].edit_type == PF.EDIT_ROUTE:
            vacancy = (route_sources[u['e'].target_machine] |
                       route_targets[u['e'].source_machine])
            shared_source = route_sources[u['e'].source_machine]
            related.update(vacancy)
            related.update(shared_source)
        def relation_key(j):
            signals = int(j in adjacent)+int(j in vacancy)+int(j in shared_source)
            return (-signals, -pool[j]['uhat'], pool[j]['sig'])
        candidates = sorted(related - used - {i},
                            key=relation_key)
        # The scan budget bounds expensive legality calls, not just saved pairs.
        for j in candidates[:partner_scan]:
            v = pool[j]
            if touched & PF._touched_operations(v['e']):
                continue
            if PF._pair_family(u, v) not in allowed:
                continue
            if not PF._joint_operator_contract_compatible(u, v)[0]:
                continue
            if not PF.check_composite_structural_legality(graph, (u['e'], v['e'])).legal:
                continue
            pairs.append((i, j))
            used.update((i, j))
            break
    return pairs


class FastAnalyzeCache(PF.AnalyzeCache):
    def proposals(self, problem, schedule, iid):
        key = f'{iid}::{schedule_hash(schedule)}'
        if key not in self._prop:
            self._prop[key] = PF.build_proposal_features(
                self.ast(problem, schedule, iid), self.single_head, self.direct_head,
                fast_pairs=True)
            self._trim(self._prop)
        self._prop.move_to_end(key)
        return self._prop[key]


class FrozenBaseMemory(ProgressiveMemory):
    """Share read-only historical evidence, copy only branch-local writes."""
    @classmethod
    def from_memory(cls, memory):
        out = cls()
        out.__dict__.update(memory.__dict__)
        # Original replay's executed entries are not this inference trajectory.
        out.executed = {}
        out._diag = {'neighbors_per_query': [], 'zero_queries': 0, 'n_queries': 0,
                     'rejected_same_instance_dump_records': 0, 'rejected_total_records': 0}
        return out

    def __deepcopy__(self, memo):
        out = object.__new__(type(self))
        memo[id(self)] = out
        out.__dict__ = self.__dict__.copy()
        out.executed = copy.deepcopy(self.executed, memo)
        out._diag = copy.deepcopy(self._diag, memo)
        return out

    def add_episode_record(self, *args, **kwargs):
        raise RuntimeError('Fixed inference Memory is read-only')

    def start_episode(self):
        raise RuntimeError('Fixed inference Memory is read-only')
