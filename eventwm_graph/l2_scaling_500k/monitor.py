"""Read-only progress and collector status; writes only research evidence."""
import datetime
import json
import os
import re
import time
from .common import OUT,SOURCE

def last_json(path):
    if not path.exists():return {}
    for line in reversed(path.read_text().splitlines()):
        try:return json.loads(line)
        except ValueError:continue
    return {}

def run():
    with (OUT/'progress_monitor.jsonl').open('x') as evidence:
        while True:
            source=json.loads((SOURCE/'status.json').read_text());counts=[]
            for i in range(4):
                path=OUT/('added_%d.log'%i)
                matches=re.findall(r'train (\d+) supervised (\d+) failures (\d+)',path.read_text() if path.exists() else '')
                counts.append(tuple(map(int,matches[-1])) if matches else (0,0,0))
            fit=last_json(OUT/'fit.log');evaluated=0
            for path in (OUT/'evaluation_arm_new').glob('run_*.log'):
                progress=last_json(path);evaluated+=progress.get('roots',0)
            row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                prepared_added_roots=sum(x[0] for x in counts),supervised_added_roots_seen=sum(x[1] for x in counts),
                preparation_failures=sum(x[2] for x in counts),epoch=fit.get('epoch'),dev_es=fit.get('dev_es'),evaluation_roots_seen=evaluated,
                complete=(OUT/'evaluation_complete.json').exists(),collector_status=source['status'],collector_workers=source['active_worker_count'],
                collector_outcomes=source['outcomes'],collector_restart_count=source['collector_restart_count'],
                collector_configuration=source['configuration_identity'],automatic_training=source['automatic_training'],load_average=os.getloadavg())
            line=json.dumps(row);evidence.write(line+'\n');evidence.flush();print(line,flush=True)
            if row['complete']:break
            time.sleep(45)

if __name__=='__main__':run()
