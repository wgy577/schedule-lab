"""Inference-only fast candidates and immutable-base/branch-local Memory."""
import copy
from collections import defaultdict

from . import proposal_features as PF
from .memory import ProgressiveMemory
from causal_schedule_lab.validation import schedule_hash


from .companion_candidates import companion_pairs, cached_proposals


class FastAnalyzeCache(PF.AnalyzeCache):
    def proposals(self, problem, schedule, iid):
        return cached_proposals(self, problem, schedule, iid)


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
