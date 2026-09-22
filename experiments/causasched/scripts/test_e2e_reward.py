import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from causal_schedule_lab.m3 import config as C, joint_grpo as JG

C.T2L_NET_REGRESSION_WEIGHT=1.
for terminal in (80,100,130):
    rows=[dict(state_makespan_before=100,successor_makespan=90),
          dict(state_makespan_before=90,successor_makespan=terminal)]
    JG._assign_t2l_future_credit(rows,terminal_ms=terminal,root_ms=100)
    for row in rows:
        assert row['m2_future_net_reward']==row['state_makespan_before']-terminal
rows=[dict(state_makespan_before=100,execution_reason='infeasible')]
JG._assign_t2l_future_credit(rows,terminal_ms=100,root_ms=100)
assert rows[0]['m2_future_net_reward']==-5
print('PASS: lambda=1 terminal improvement identity; infeasible penalty retained')
