import copy
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch import nn
from causal_schedule_lab.m3 import probability_trace as PT, proposal_features as PF
from causal_schedule_lab.m3.trace_depth import configure_trace_depth,expand_trace_state,reset_expanded_optimizer_states

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.probability_trace=PT.ProbabilityTrace(3,2,hops=4)
        self.other=nn.Linear(3,1)
    def enable_probability_trace(self): pass

class TraceTests(unittest.TestCase):
    def test_rows_and_shared_weights_preserved(self):
        model=Actor(); before=copy.deepcopy(model.state_dict())
        after,changed=expand_trace_state(before,6)
        self.assertEqual(len(changed),3)
        for name,tensor in before.items():
            torch.testing.assert_close(after[name][:len(tensor)] if name in changed else after[name],tensor,rtol=0,atol=0)
        self.assertEqual(after['probability_trace.hop_embedding.weight'].shape,(6,8))
        self.assertEqual(after['probability_trace.depth.weight'].shape,(7,5))
        self.assertEqual(before['probability_trace.depth.weight'].shape,(5,5))
    def test_new_depth_initial_mass_small(self):
        actor=Actor(); old=actor.probability_trace
        context=torch.randn(20,5)
        configure_trace_depth(dict(policy=NS(m2=actor)),6)
        p=actor.probability_trace.depth(context).softmax(-1)
        self.assertLess(float(p[:,5:].sum(-1).max().detach()),2*math.exp(-4))
        torch.testing.assert_close(actor.probability_trace.depth(context)[:,:5],old.depth(context))
    def test_graph_and_ancestry_both_six(self):
        configure_trace_depth(dict(policy=NS(m2=Actor())),6)
        nodes=[f'operation:o{i}' for i in range(9)]
        ast=dict(trace_appearance_seeds=[('b',[nodes[-1]])],node_index={n:i for i,n in enumerate(nodes)},
                 trace_causal_edges=list(zip(nodes[:-1],nodes[1:])),h_c=torch.randn(9,3))
        graph=PT.build_trace_graph(ast,['o2','o1'])
        self.assertEqual(len(graph['nodes']),7)
        self.assertGreaterEqual(int(graph['roots'][0]),0)
        self.assertEqual(int(graph['roots'][1]),-1)
        batch=NS(node_numeric=torch.zeros(9,1),reverse_causal_mask=torch.ones(8,dtype=torch.bool),
                 edge_index=torch.tensor([list(range(8)),list(range(1,9))]),
                 symptom_block_node_index=(torch.tensor([0]),torch.tensor([8])))
        hops=PF._bounded_reverse_hops(batch,1)
        self.assertEqual(int(hops[0,2]),6)
        self.assertEqual(int(hops[0,1]),-1)
        self.assertEqual((PF.ROOT_APP_MAX_HOPS,PT.TRACE_MAX_HOPS),(6,6))
    def test_no_rng_consumption(self):
        actor=Actor(); rng=torch.get_rng_state().clone()
        configure_trace_depth(dict(policy=NS(m2=actor)),6)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_no_silent_shrink_and_idempotent(self):
        actor=Actor(); r=dict(policy=NS(m2=actor));configure_trace_depth(r,6)
        params=copy.deepcopy(actor.state_dict());configure_trace_depth(r,6)
        for n,p in actor.state_dict().items():torch.testing.assert_close(params[n],p,atol=0,rtol=0)
        with self.assertRaises(ValueError):expand_trace_state(params,4)
    def test_optimizer_only_resized_states_reset(self):
        old=Actor(); opt=torch.optim.AdamW(old.parameters())
        sum(p.sum() for p in old.parameters()).backward();opt.step()
        old_weights=copy.deepcopy(old.state_dict()); saved=copy.deepcopy(opt.state_dict())
        new=Actor();configure_trace_depth(dict(policy=NS(m2=new)),6)
        migrated,changed=expand_trace_state(old_weights,6);new.load_state_dict(migrated)
        opt2=torch.optim.AdamW(new.parameters());opt2.load_state_dict(saved)
        reset_expanded_optimizer_states(opt2,list(new.named_parameters()),changed)
        old_named=dict(old.named_parameters())
        for name,p in new.named_parameters():
            if name in changed:self.assertNotIn(p,opt2.state)
            else:
                for key,value in opt.state[old_named[name]].items():
                    torch.testing.assert_close(opt2.state[p][key],value,atol=0,rtol=0)
        sum(p.sum() for p in new.parameters()).backward();opt2.step()
        for p in new.parameters():self.assertEqual(opt2.state[p]['exp_avg'].shape,p.shape)

if __name__=='__main__':unittest.main()
