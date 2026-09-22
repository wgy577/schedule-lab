import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from causal_schedule_lab.m3.adaptive_search import restore_clock,begin_episode,rollout_horizon
from causal_schedule_lab.m3.stall_perturbation import choose_start,refresh_pool

def node(h,ms): return NS(state_hash=h,ms=ms)

class RestartTests(unittest.TestCase):
    def setUp(self):
        self.best=node('best',100)
        self.state=dict(current=self.best,stall_segments=3)
    def test_no_chained_restart(self):
        first=node('first',105); second=node('second',109)
        refresh_pool(self.state,[first],source_hash='best')
        root,changed=choose_start(self.state)
        self.assertTrue(changed);self.assertIs(root,first)
        self.assertIs(self.state['current'],self.best)
        refresh_pool(self.state,[second],source_hash='first')
        self.assertNotIn(second,self.state['perturb_pool'])
        self.state['stall_segments']=4
        self.assertEqual(choose_start(self.state),(self.best,False))
        self.state['stall_segments']=6
        self.assertEqual(choose_start(self.state),(first,True))
    def test_stall_not_reset_by_restart(self):
        refresh_pool(self.state,[node('x',105)],source_hash='best')
        choose_start(self.state)
        self.assertEqual(self.state['stall_segments'],3)
        self.assertEqual(rollout_horizon(self.state,30,500),20)
    def test_new_best_invalidates_old_pool(self):
        refresh_pool(self.state,[node('x',105)],source_hash='best')
        self.state['current']=node('new',98)
        refresh_pool(self.state,[node('y',100)],source_hash='best')
        self.assertEqual(self.state['perturb_pool'],[])
        self.assertEqual(self.state['perturb_anchor_hash'],'new')
    def test_legacy_provenance_not_trusted(self):
        self.state['perturb_pool']=[node('old',105)]
        self.assertEqual(choose_start(self.state),(self.best,False))
    def test_gap_bound_and_disabled(self):
        refresh_pool(self.state,[node('too_far',111),node('ok',109)],source_hash='best')
        self.assertEqual([s.ms for s in self.state['perturb_pool']],[109])
        self.assertEqual(choose_start(self.state,patience=0),(self.best,False))

class ClockTests(unittest.TestCase):
    def test_legacy_cycle_50_is_370_steps(self):
        ck=dict(cycle=50,config=dict(episode_origin_cycle=13))
        c=restore_clock(ck,['a','b'],2,500,10)
        self.assertEqual(c,dict(episode=0,updates=37,budgets={'a':370,'b':370}))
        self.assertFalse(begin_episode(c,['a','b'],2,500))
    def test_legacy_boundary_resets_next_cohort(self):
        c=restore_clock(dict(cycle=63,config=dict(episode_origin_cycle=13)),['a','b','c'],2,500,10)
        self.assertEqual(c['episode'],1)
        self.assertTrue(begin_episode(c,['a','b','c'],2,500))
        self.assertEqual(c['budgets'],{'c':0,'a':0})
    def test_mixed_horizons_never_overrun(self):
        c=restore_clock(None,['a','b'],2,500,10)
        begin_episode(c,['a','b'],2,500)
        while min(c['budgets'].values())<500:
            for iid in c['budgets']:
                h=rollout_horizon({'stall_segments':3 if iid=='b' else 0},c['budgets'][iid],500)
                c['budgets'][iid]+=h
                self.assertLessEqual(c['budgets'][iid],500)
        self.assertEqual(c['budgets'],{'a':500,'b':500})
        self.assertTrue(begin_episode(c,['a','b'],2,500))
        self.assertEqual(c['episode'],1)
    def test_resume_exact_and_no_alias(self):
        saved=dict(episode=0,updates=30,budgets={'a':490,'b':500})
        c=restore_clock({'episode_clock':saved},['a','b'],2,500,10)
        self.assertEqual(c,saved)
        self.assertEqual(rollout_horizon({'stall_segments':3},490,500),10)
        c['budgets']['a']=500
        self.assertEqual(saved['budgets']['a'],490)
    def test_improvement_shortens_again(self):
        self.assertEqual(rollout_horizon({'stall_segments':2},20,500),20)
        self.assertEqual(rollout_horizon({'stall_segments':0},40,500),10)

if __name__=='__main__':unittest.main()
