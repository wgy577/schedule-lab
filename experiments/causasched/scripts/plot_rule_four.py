"""Reproduce and plot the exact earliest_finish training initialization."""
import json
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from causal_schedule_lab.ir import Problem
from causal_schedule_lab.solvers.dispatching import solve_dispatching
from causal_schedule_lab.core_validation import validate_schedule

out = ROOT/'outputs/rule_gantt_behnke4_7'
out.mkdir(parents=True, exist_ok=True)
def key(s):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', s)]
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(2, 2, figsize=(19, 15))
summary=[]
bank=ROOT/'data/train128'
files={row['instance_id']:row['file'] for row in json.loads((bank/'protocol.json').read_text())['entries']}
for ax, number in zip(axes.flat, range(4,8)):
    iid=f'Behnke{number}'
    data=json.loads((bank/files[iid]).read_text())
    problem=Problem.model_validate(data['problem'])
    schedule=solve_dispatching(problem, rule='earliest_finish')
    assert validate_schedule(problem, schedule).feasible
    ops=problem.operation_map()
    machines=sorted([r.id for r in problem.resources], key=key)
    jobs=sorted([j.id for j in problem.jobs], key=key)
    job_index={j:i for i,j in enumerate(jobs)}
    op_index={}
    for j in jobs:
        for i, op in enumerate(sorted([o.id for o in problem.operations if o.job_id==j],key=key),1):
            op_index[op]=i
    colors=plt.get_cmap('tab20').colors
    for a in schedule.assignments:
        op=ops[a.operation_id]
        mode=next(m for m in op.modes if m.id==a.mode_id)
        ji=job_index[op.job_id]
        for resource in mode.resources:
            y=machines.index(resource)
            ax.barh(y, a.end-a.start, left=a.start, height=.73,
                    color=colors[ji%20], edgecolor='#39434c', linewidth=.45)
            ax.text((a.start+a.end)/2,y,f'{ji+1}.{op_index[op.id]}',
                    ha='center',va='center',fontsize=6.5)
    ax.set_yticks(range(len(machines)), machines)
    ax.invert_yaxis()
    ax.set_xlim(0, schedule.makespan*1.04)
    ax.set_xlabel('Time'); ax.set_ylabel('Machine')
    ax.set_title(f'{iid} | {len(jobs)} jobs, {len(machines)} machines, '
                 f'{len(ops)} operations\nEarliest-finish rule | makespan = {schedule.makespan}',
                 fontsize=13, fontweight='bold',pad=12)
    ax.set_axisbelow(True); ax.grid(axis='x',color='#e2e5e8',linewidth=.6)
    ax.axvline(schedule.makespan,color='#ad3344',ls='--',lw=1)
    ax.legend(handles=[Patch(facecolor=colors[i%20],label=f'J{i+1}') for i in range(len(jobs))],
              loc='upper center',bbox_to_anchor=(.5,-.10),ncol=10,fontsize=7,
              frameon=False,columnspacing=.7,handlelength=1)
    (out/f'{iid}_rule.json').write_text(json.dumps(dict(problem=data['problem'],
        schedule=schedule.model_dump(mode='json')),indent=2))
    summary.append(dict(instance=iid,rule_makespan=schedule.makespan,
                        drl_makespan=data['makespan'],feasible=True))
fig.suptitle('Rule-generated initial schedules (not RL-improved schedules)\n'
             'Color = job; bar label = job.operation; blank intervals = idle time',fontsize=16,y=.995)
fig.tight_layout(rect=(0,.025,1,.965),h_pad=5,w_pad=3)
fig.savefig(out/'four_rule_gantts.png',dpi=180,bbox_inches='tight')
fig.savefig(out/'four_rule_gantts.svg',bbox_inches='tight')
print(json.dumps(summary,indent=2)); print(out/'four_rule_gantts.png')
