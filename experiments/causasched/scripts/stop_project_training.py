"""Linux: stop only named project training/inference trees; preserve all files."""
import os
import signal
import time
from pathlib import Path

ROOTS = tuple(Path('/root') / n for n in (
    't2m_e2e_dispatch_v8',
    't2m_e2e_dispatch_v7',
    't2m_e2e_dispatch_v3', 't2m_e2e_dispatch_v4', 't2m_e2e_dispatch_v5', 't2m_e2e_dispatch_v6',
    't2m_e2e_single_no_kl_v2', 't2m_e2e_single_v1', 't2m_fast_u356',
    't2m_hierarchical_joint_grpo_autodl'))
NAMES = ('train_e2e_single.py', 'run_e2e_single.sh', 'run_fast_inference.py',
         'run_fast_parallel.py', 'run_t2m.sh', 'run_t2l.sh',
         'run_t2l_unified_intervention_grpo_gpu.py')

def snapshot():
    rows = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
            rows[int(p.name)] = (int(fields[1]), fields[19],
                (p/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'),
                (p/'cwd').resolve())
        except (OSError, ValueError):
            pass
    return rows

rows = snapshot()
protected = {os.getpid()}
parent = os.getpid()
while parent in rows and rows[parent][0] not in protected:
    parent = rows[parent][0]
    protected.add(parent)
targets = {pid for pid, (_, _, cmd, cwd) in rows.items()
           if pid not in protected and any(cwd == r or r in cwd.parents for r in ROOTS)
           and any(n in cmd for n in NAMES)}
while True:
    more = {pid for pid, row in rows.items() if row[0] in targets} - protected
    if more <= targets:
        break
    targets |= more
for pid in sorted(targets):
    print(f'STOP {pid}: {rows[pid][2]}', flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(3)
fresh = snapshot()
for pid in targets:
    if pid in fresh and fresh[pid][1] == rows[pid][1]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
print(f'Stopped {len(targets)} matching processes. No result files deleted.')
