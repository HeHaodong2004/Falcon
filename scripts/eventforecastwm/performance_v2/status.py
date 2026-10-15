"""Summarize actual experiment receipts; checkpoint presence is not completion."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

from eventbeliefwm.data import read
from eventbeliefwm.experiment import ResearchGuard,save_json
from eventforecastwm.workflow import active


def main():
    p=argparse.ArgumentParser();p.add_argument('--configs',default='configs/eventwm_performance_v2')
    p.add_argument('--output');a=p.parse_args();rows=[];destinations={}
    for config in sorted(Path(a.configs).glob('*_launch.json')):
        cfg=read(config);destinations.setdefault(str(Path(cfg['run']).resolve()),[]).append((config,cfg))
    for directory,aliases in destinations.items():
        config,cfg=aliases[0];run=Path(directory);progress=read(run/'progress.json') if (run/'progress.json').exists() else {}
        running=active(run)
        row=dict(run=str(run),config=str(config),launch_configs=[str(path) for path,_ in aliases],
                 module=cfg['module'],active=running)
        if (run/'result.json').exists():
            receipt=read(run/'result.json')
            row['status']='completed' if receipt.get('status')!='queue_blocked' else 'completed_with_failed_children'
            row['receipt_status']=receipt.get('status','evaluation_complete')
            for k in ('steps','presentations','training_roots','supported_roots','supported_windows','best_step','runtime_s','duration_s','checkpoint_sha256'):
                if k in receipt:row[k]=receipt[k]
        elif running:
            row['status']='running'
            if progress.get('collector_guard_healthy') is False:row['status']='monitoring_resource_pause'
        elif (run/'STOP_RESEARCH').exists():row['status']='cooperatively_stopped'
        elif (run/'failure.json').exists():
            error=read(run/'failure.json').get('error','')
            row['status']='paused_resource' if 'Research stopped:' in error else 'failed'
            row['reason']=error.splitlines()[-1] if error else 'see failure.json'
        elif (run/'last.pt').exists():row['status']='interrupted_checkpoint_available'
        else:row['status']='not_executed'
        row['progress']=progress
        rows.append(row)
    collector=ResearchGuard.sample()
    result=dict(timestamp=time.time(),counts=dict(Counter(r['status'] for r in rows)),experiments=rows,
        collector=collector,live_query_count_is_not_a_validated_snapshot=True,production_actions=[],
        counting_unit='Unique output directory; superseded launch-config aliases do not multiply experiments. Owner/verification jobs remain explicitly identified by module.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()))
    if a.output:save_json(a.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
