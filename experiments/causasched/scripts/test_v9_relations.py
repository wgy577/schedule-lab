"""Deterministic matching and reward checks; actual-network parity is separate."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from causal_schedule_lab.m3 import fast_inference as FI
from causal_schedule_lab.m3 import joint_grpo as JG, config as C

def route(op,src,dst,score=1):
    return dict(e=NS(edit_type='ROUTE',operation_id=op,source_machine=src,target_machine=dst),
                sig=f'ROUTE:{op}:{src}->{dst}',uhat=score)

class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.graph=NS(predecessors={},intervals=[])
        patches=[patch.object(FI.PF,'T2L_PAIR_TOTAL_CAP',96),
                 patch.object(FI.PF,'_touched_operations',side_effect=lambda e:{e.operation_id}),
                 patch.object(FI.PF,'_pair_family',return_value='ROUTE+ROUTE'),
                 patch.object(FI.PF,'_joint_operator_contract_compatible',return_value=(True,'ok')),
                 patch.object(FI.PF,'check_composite_structural_legality',return_value=NS(legal=True))]
        for p in patches: p.start(); self.addCleanup(p.stop)
    def test_shared_source_without_adjacency(self):
        pool=[route('a','M1','M2'),route('b','M1','M3')]
        self.assertEqual(FI.companion_pairs(pool,self.graph),[(0,1)])
        self.assertEqual(len(pool),2)  # matching never removes single candidates
    def test_incoming_link_symmetric(self):
        pool=[route('a','M1','M2',10),route('b','M3','M1')]
        self.assertEqual(FI.companion_pairs(pool,self.graph),[(0,1)])
    def test_dependency_across_machines(self):
        self.graph.predecessors={'b':['a']}
        self.assertEqual(FI.companion_pairs([route('a','M1','M2'),route('b','M3','M4')],self.graph),[(0,1)])
    def test_no_forced_unrelated(self):
        self.assertEqual(FI.companion_pairs([route('a','M1','M2'),route('b','M3','M4')],self.graph),[])
    def test_matching_unique_deterministic(self):
        pool=[route(str(i),'M1','M2',i) for i in range(7)]
        pairs=FI.companion_pairs(pool,self.graph)
        self.assertEqual(pairs,FI.companion_pairs(pool,self.graph))
        atoms=[i for pair in pairs for i in pair]
        self.assertEqual(len(atoms),len(set(atoms)))
        self.assertEqual(len(pairs),3)
    def test_illegal_rejected(self):
        FI.PF.check_composite_structural_legality.return_value=NS(legal=False)
        self.assertEqual(FI.companion_pairs([route('a','M1','M2'),route('b','M1','M3')],self.graph),[])
    def test_caps(self):
        pool=[route(str(i),'M1','M2') for i in range(12)]
        with patch.object(FI.PF,'T2L_PAIR_TOTAL_CAP',1):
            self.assertEqual(len(FI.companion_pairs(pool,self.graph)),1)
        self.assertEqual(FI.companion_pairs(pool,self.graph,partner_scan=0),[])
    def test_more_relations_then_prior(self):
        self.graph.predecessors={'b':['a']}
        pool=[route('a','M1','M2',100),route('b','M1','M3',1),route('c','M1','M4',10)]
        self.assertEqual(FI.companion_pairs(pool,self.graph)[0],(0,1))

class RewardTests(unittest.TestCase):
    def test_lambda02_keeps_best_credit_and_penalizes_regression(self):
        with patch.object(C,'T2L_NET_REGRESSION_WEIGHT',.2):
            rows=[dict(state_makespan_before=100,successor_makespan=90),
                  dict(state_makespan_before=90,successor_makespan=130)]
            JG._assign_t2l_future_credit(rows,terminal_ms=130,root_ms=100)
            self.assertAlmostEqual(rows[0]['m2_future_net_reward'],2.)
            self.assertAlmostEqual(rows[1]['m2_future_net_reward'],-8.)

if __name__=='__main__': unittest.main()
