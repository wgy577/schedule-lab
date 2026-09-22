import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
from types import SimpleNamespace as NS
from causal_schedule_lab.m3.load_reward import load_cost, load_bonus, add_load_credit, balance_group
from causal_schedule_lab.m3.adaptive_search import begin_episode

class Tests(unittest.TestCase):
    def test_share(self):
        rows=[dict(makespan_reward=m,load_reward=a,steps=[dict(
            makespan_future_reward=m,load_future_reward=a)]) for m,a in [(10,1),(-20,-2)]]
        d=balance_group(rows,.35)
        self.assertAlmostEqual(d['actual_share'],.35)
        self.assertAlmostEqual(sum(abs(r['load_reward']) for r in rows),30*.35/.65)
        self.assertEqual(rows[0]['steps'][0]['m2_future_net_reward'],rows[0]['reward'])
        self.assertGreater(rows[0]['load_reward'],0)
        self.assertLess(rows[1]['load_reward'],0)
        self.assertEqual(balance_group([dict(makespan_reward=3,load_reward=0,steps=[])])['actual_share'],0)
        self.assertEqual(balance_group([dict(makespan_reward=0,load_reward=2,steps=[])])['actual_share'],1)
    def test_load(self):
        p=NS(resources=[NS(id='a'),NS(id='b')], mode_map=lambda:{
            'a':(None,NS(resources=('a',))), 'b':(None,NS(resources=('b',)))})
        def sched(x,y,offset=0):
            return NS(assignments=[NS(mode_id='a',start=offset,end=offset+x),
                                   NS(mode_id='b',start=0,end=y)])
        self.assertAlmostEqual(load_cost(p,sched(100,20)),2/3)
        self.assertGreater(load_bonus(2/3,load_cost(p,sched(70,50)),120,.2),0)
        self.assertGreater(load_bonus(2/3,load_cost(p,sched(100,100)),120,.2),0)
        self.assertAlmostEqual(load_cost(p,sched(100,20,50)),2/3)
        self.assertAlmostEqual(load_cost(p,sched(200,40)),2/3)
        self.assertEqual(load_cost(p,sched(100,0)),1)
        self.assertEqual(load_bonus(10000,0,100,.2),5)
        self.assertEqual(load_bonus(0,10000,100,.2),-5)
        self.assertEqual(load_bonus(112,80,100,0),0)
        tr=dict(root_ms=120,terminal_schedule=sched(70,50),reward=0.,U2=0.,
                steps=[dict(load_cost_before=2/3,load_cost_after=1/6,m2_future_net_reward=0)])
        add_load_credit(tr,p,sched(100,20),.2)
        self.assertAlmostEqual(tr['reward'],.1)
        self.assertAlmostEqual(tr['steps'][0]['m2_future_net_reward'],.1)
        self.assertEqual(tr['makespan_reward'],0)

    def test_default_share30(self):
        rows=[dict(makespan_reward=7.,load_reward=.1,steps=[])]
        self.assertAlmostEqual(balance_group(rows)['actual_share'],.30)
        self.assertAlmostEqual(rows[0]['load_reward'],3.)

    def test_all26_every_episode(self):
        ids=[str(i) for i in range(26)]; clock=dict(episode=0,updates=0,budgets={})
        for _ in range(3):
            self.assertTrue(begin_episode(clock,ids,26,500))
            self.assertEqual(set(clock['budgets']),set(ids))
            self.assertEqual(set(clock['budgets'].values()),{0})
            while min(clock['budgets'].values())<500:
                h=min(20,500-min(clock['budgets'].values()))
                for iid in ids: clock['budgets'][iid]+=h
                if h and min(clock['budgets'].values())<500:
                    self.assertFalse(begin_episode(clock,ids,26,500))
            self.assertEqual(set(clock['budgets'].values()),{500})

if __name__=='__main__':unittest.main()
