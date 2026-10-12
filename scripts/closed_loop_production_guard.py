"""Read-only C24 publication guard, isolated from offline training dependencies.

The sampling and guard implementations are copied unchanged from the accepted
V4 research runner. They can only stop the research caller; they never invoke
collector control APIs or write production data.
"""
import json
from pathlib import Path
import sqlite3
import time

PROJECT = Path('/home/haodong/workspace/projects/eventwm-falcon')
DATA = Path('/tmp/eventwm_v4_300k')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def ro(path):
    db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, timeout=10)
    db.execute('PRAGMA query_only=ON')
    return db


def production_sample():
    with ro(DATA/'pipeline.sqlite3') as db:
        db.execute('BEGIN')
        m = json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])
        p = json.loads(db.execute("SELECT value FROM meta WHERE key='progress'").fetchone()[0])
    pid = m['runner']['pid']
    return dict(timestamp=time.time(),progress=p,runner_pid=pid,
                runner_alive=Path(f'/proc/{pid}').exists(),state=m['state'],
                native_limit=read(DATA/'native_limit.json')['limit'],
                source_commit=m['frozen_commit'],pipeline_settings=m['pipeline_settings'])


class ProductionGuard:
    """Stops research only after sustained publication degradation or PID change."""
    def __init__(self,run):
        self.run=run
        self.last=0
        self.samples=[]
        self.bad=0
        self.start=production_sample()
        self.baseline=self.historical_rate()
        save(run/'production_guard.json',dict(baseline_qph=self.baseline,window_seconds=600,degradation_fraction=.25,
                                             consecutive_checks=3,check_interval=60,actions='raise and stop this research process only'))

    def historical_rate(self):
        log=PROJECT/'output/eventwm_300k_logs/progress.jsonl'
        if not log.exists():
            return None
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-8*1024*1024))
            rows=[]
            for line in f.read().splitlines():
                try:
                    r=json.loads(line)
                    if r.get('source_commit')==self.start['source_commit'] and r.get('runner',{}).get('pid')==self.start['runner_pid']:
                        rows.append(r)
                except ValueError:
                    pass
        now=self.start['timestamp']
        rates={}
        for seconds in (600,1800):
            candidates=[r for r in rows if seconds<=now-r['timestamp']<=seconds+180]
            if candidates:
                r=min(candidates,key=lambda x:abs(now-x['timestamp']-seconds))
                rates[str(seconds)]=(self.start['progress']['primary_queries']-r['new_retained'])/(now-r['timestamp'])*3600
        save(self.run/'publication_baselines.json',dict(timestamp=now,rates_qph=rates,
            policy='Use the lower contemporaneous 10/30 minute baseline; preserve both windows. Publication is bursty and scene admission varies, so this prevents attributing an existing decline to training. A later sustained additional decline still stops research.'))
        return min(rates.values()) if rates else None

    def check(self):
        now=time.time()
        if now-self.last<60:
            return
        self.last=now
        sample=production_sample()
        self.samples.append(sample)
        earlier=[s for s in self.samples if now-s['timestamp']>=600]
        rate=None
        if earlier:
            old=earlier[-1]
            rate=(sample['progress']['primary_queries']-old['progress']['primary_queries'])/(now-old['timestamp'])*3600
        self.bad=self.bad+1 if rate is not None and self.baseline and rate<.75*self.baseline else 0
        with (self.run/'production_monitor.jsonl').open('a') as f:
            f.write(json.dumps(dict(**sample,qph_10min=rate,baseline_qph=self.baseline,consecutive_degraded=self.bad))+'\n')
        if sample['runner_pid']!=self.start['runner_pid'] or not sample['runner_alive'] or self.bad>=3:
            raise RuntimeError('Production guard stopping RESEARCH: runner changed/unavailable or sustained publication degradation')
