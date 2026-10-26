"""Read-only logger of Stage 2 training throughput and machine load (interference audit).

Never signals or touches Stage 2 processes; only reads their log files and /proc.
"""
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LOGS = Path('/home/haodong/workspace/projects/eventwm-falcon-stage2-repair-20261025/output/claude_stage2/logs')
OUT = Path(sys.argv[1])


def rate(path):
    rows = []
    try:
        for line in path.open():
            line = line.strip()
            if line.startswith("{'epoch'"):
                try:rows.append(ast.literal_eval(line))
                except Exception:pass
    except OSError:
        return None
    if len(rows) < 2:return None
    a, b = rows[-2], rows[-1]
    if b['elapsed'] <= a['elapsed']:return None
    return dict(step=b['step'], elapsed=b['elapsed'], steps_per_s=(b['step'] - a['step']) / (b['elapsed'] - a['elapsed']))


while True:
    active = {}
    try:
        ps = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
        names = {part.split('--name ')[1].split()[0] for part in ps.splitlines() if 'stage2_claude.train' in part and '--name ' in part}
    except Exception:
        names = set()
    for n in sorted(names):
        active[n] = rate(LOGS / (n + '.log'))
    mine = [l for l in ps.splitlines() if 'stage1_hwm' in l and 'python' in l]
    row = dict(time=time.time(), load=os.getloadavg(), stage2=active, stage1_processes=len(mine))
    with OUT.open('a') as f:f.write(json.dumps(row) + '\n')
    time.sleep(120)
