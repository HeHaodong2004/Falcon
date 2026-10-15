"""Persistent collector safeguards; never signals production processes."""
import json
from pathlib import Path
import shutil
import time
from eventbeliefwm.experiment import ResearchGuard,save_json


class Guard:
    def __init__(self,directory):
        self.directory=Path(directory);self.last=0
        path=self.directory/'guard.json'
        if path.exists():self.state=json.loads(path.read_text())
        else:
            s=ResearchGuard.sample()
            self.state=dict(collector_pid=s['collector_pid'],baseline_qph=3615.965194750963,
                            samples=[],degraded=0,stopped=False)
        self.check(True)

    def check(self,force=False):
        now=time.time()
        if not force and now-self.last<60:return
        # A new short job must not reset the shared full-pool safety history.
        for parent in self.directory.parents:
            shared=parent/'cache_full/guard.json'
            if shared.exists() and shared.parent!=self.directory:
                evidence=json.loads(shared.read_text())
                if evidence['stopped']:raise RuntimeError('Research stopped: shared full-pool collector guard')
                if evidence['samples'] and now-evidence['samples'][-1]['timestamp']>180:
                    raise RuntimeError('Research stopped: shared collector evidence stale')
                break
        self.last=now;s=ResearchGuard.sample();history=self.state['samples']
        old=[v for v in history if s['timestamp']-v['timestamp']>=600]
        rate=(s['queries']-old[-1]['queries'])*3600/(s['timestamp']-old[-1]['timestamp']) if old else None
        self.state['degraded']=self.state['degraded']+1 if rate is not None and rate<.65*self.state['baseline_qph'] else 0
        s['recent_qph']=rate;history.append(s);self.state['samples']=history[-120:]
        failure=None
        if shutil.disk_usage(self.directory).free<50*1024**3:failure='disk reserve below50GiB'
        if s['collector_pid']!=self.state['collector_pid'] or s['state']!='running' or now-s['heartbeat_timestamp']>180:
            failure='collector ownership/state/heartbeat changed'
        if self.state['degraded']>=3:failure='three degraded10minute windows below65percent baseline'
        if self.state['stopped']:
            recovered=[x for x in history[-3:] if x.get('recent_qph') is not None and x['recent_qph']>=.85*self.state['baseline_qph']]
            if len(recovered)!=3 or recovered[-1]['timestamp']-recovered[0]['timestamp']<120:
                failure='recovery requires3 sustained windows above85percent original baseline'
            elif failure is None:self.state['stopped']=False
        if failure:self.state['stopped']=True
        save_json(self.directory/'guard.json',self.state)
        with (self.directory/'collector_monitor.jsonl').open('a') as f:f.write(json.dumps(dict(**s,failure=failure))+'\n')
        if failure:raise RuntimeError('Research stopped: '+failure)
