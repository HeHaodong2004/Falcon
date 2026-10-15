"""User-authorized training-priority resource sharing; no production mutations.

Throughput is a soft constraint. Only health, storage, RAM and OOM faults stop
research. Runtime injection keeps the frozen physical optimizer identity intact.
"""
import json
import math
import os
from pathlib import Path
import shutil
import time

from eventbeliefwm.experiment import ResearchGuard,save_json

VERSION='training_priority_absolute_qph_v2_20261015'
SETTINGS=dict(target_qph=1500.,acceptable_min_qph=1000.,low_windows=3,recovery_windows=2,
    window_seconds=600.,sample_spacing_seconds=60.,maximum_window_seconds=900.,
    heartbeat_max_age_seconds=180.,ram_reserve_gib=48.,disk_reserve_gib=50.,
    training_duty_cycles=[1.,.9,.8,.6],minimum_recovery_span_seconds=60.)
collector_sample=ResearchGuard.sample


def valid_samples(history):
    result=[]
    for row in history:
        rate=row.get('recent_qph')
        valid=(row.get('measurement_valid',True) and rate is not None and math.isfinite(rate) and rate>=0 and
               row.get('state')=='running' and 0<=row['timestamp']-row['heartbeat_timestamp']<=180)
        if valid and (not result or row['timestamp']-result[-1]['timestamp']>=60):result.append(row)
    return result


def throughput_state(state):
    """Hysteretic policy; sub-1000 throughput never creates a new full stop."""
    history=valid_samples(state['samples']);recent=history[-1]['recent_qph'] if history else None
    recovered=(len(history)>=2 and history[-1]['timestamp']-history[-2]['timestamp']>=60 and
               all(row['recent_qph']>1500 for row in history[-2:]))
    low=0
    for row in reversed(history):
        if row['recent_qph']>=1000:break
        low+=1
    level=state.get('throttle_level',0)
    if recovered:level=0;state['legacy_resume_pending']=False
    elif low>=3:level=max(level,min(3,low//3))
    state.update(policy_version=VERSION,policy=SETTINGS,low_windows=low,throttle_level=level,
        duty_cycle=SETTINGS['training_duty_cycles'][level],defer_auxiliary=False,throttle_active=level>0,
        recovered_above_target=recovered,latest_valid_qph=recent,
        throughput_status=('throttled' if level else 'warming_measurement' if recent is None else
                           'target_met' if recent>=1500 else 'acceptable' if recent>=1000 else 'low_pending_throttle'))
    return state


def resource_sample(directory):
    memory={line.split()[0].rstrip(':'):int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2}
    vm=dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return dict(available_ram_bytes=memory['MemAvailable'],disk_free_bytes=shutil.disk_usage(directory).free,
                kernel_oom_kills=int(vm.get('oom_kill',0)),load_average=list(os.getloadavg()))


class AdaptiveGuard:
    sample=staticmethod(collector_sample)

    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.last=0.;self.delay_anchor=time.monotonic();self.training_duty=1.
        self.shared=None
        for parent in self.directory.parents:
            path=parent/'cache_full/guard.json'
            if path.exists() and path.parent!=self.directory:self.shared=path;break
        path=self.directory/'guard.json'
        self.state=json.loads(path.read_text()) if path.exists() else dict(samples=[],stopped=False)
        self.check(True)

    def check(self,force=False):
        # Bounded duty-cycle sharing changes wall time, never optimization/RNG.
        elapsed=time.monotonic()-self.delay_anchor
        if self.training_duty<1 and elapsed>=5:
            time.sleep(min(5.,5*(1/self.training_duty-1)));self.delay_anchor=time.monotonic()
        now=time.time()
        if not force and now-self.last<60:return
        self.last=now;resources=resource_sample(self.directory);sample=collector_sample()
        hard=None
        if resources['available_ram_bytes']<48*1024**3:hard='RAM reserve below48GiB'
        if resources['disk_free_bytes']<50*1024**3:hard='disk reserve below50GiB'
        expected=self.state.get('collector_pid',sample['collector_pid'])
        if sample['collector_pid']!=expected or sample['state']!='running' or not 0<=now-sample['heartbeat_timestamp']<=180:
            hard='collector ownership/state/heartbeat changed'
        try:os.kill(sample['collector_pid'],0)
        except (ProcessLookupError,PermissionError):hard='collector process unavailable'
        oom_start=self.state.get('kernel_oom_baseline',resources['kernel_oom_kills'])
        if resources['kernel_oom_kills']>oom_start:hard='new kernel OOM kill; inspect before resuming'
        self.state.update(collector_pid=expected,kernel_oom_baseline=oom_start,resources=resources)
        if self.shared is not None:
            shared=json.loads(self.shared.read_text())
            if shared.get('policy_version')!=VERSION:hard='training-priority shared policy not initialized'
            elif not shared.get('samples') or now-shared['samples'][-1]['timestamp']>180:hard='shared collector evidence stale'
            elif shared['stopped']:hard=shared.get('hard_failure') or 'shared recovery pending'
            self.state.update(policy_version=VERSION,policy=SETTINGS,shared_guard=str(self.shared),
                latest_valid_qph=shared.get('latest_valid_qph'),throttle_level=shared.get('throttle_level',0),
                duty_cycle=shared.get('duty_cycle',1.),throughput_status=shared.get('throughput_status'))
        else:
            if self.state.get('policy_version')!=VERSION:
                self.state['policy_migration']=dict(timestamp=now,previous_stopped=self.state.get('stopped',False),
                    previous_baseline_qph=self.state.get('baseline_qph'),history_samples_preserved=len(self.state['samples']),
                    authorization='User explicitly replaced relative thresholds with absolute qph and adaptive throttling')
                self.state['legacy_resume_pending']=bool(self.state.get('stopped'))
            history=self.state['samples'];eligible=[r for r in history if
                600<=sample['heartbeat_timestamp']-r['heartbeat_timestamp']<=900 and
                r['collector_pid']==sample['collector_pid'] and r['state']=='running' and
                0<=r['timestamp']-r['heartbeat_timestamp']<=180 and sample['queries']>=r['queries']]
            prior=eligible[-1] if eligible else None
            valid=(prior is not None and 0<=now-sample['heartbeat_timestamp']<=180 and
                   (not history or (sample['timestamp']-history[-1]['timestamp']>=60 and
                                    sample['heartbeat_timestamp']>history[-1]['heartbeat_timestamp'])))
            if history and sample['queries']<history[-1]['queries']:hard='collector query counter decreased'
            sample.update(recent_qph=((sample['queries']-prior['queries'])*3600/(sample['heartbeat_timestamp']-prior['heartbeat_timestamp']) if valid else None),
                measurement_valid=valid,window_seconds=sample['heartbeat_timestamp']-prior['heartbeat_timestamp'] if prior else None,
                window_start_timestamp=prior['heartbeat_timestamp'] if prior else None,
                window_start_queries=prior['queries'] if prior else None,
                measurement_note='Counter delta over collector heartbeat timestamps; >=600s window, fresh heartbeat, decisions >=60s apart; not lifetime qph')
            if not history or sample['timestamp']-history[-1]['timestamp']>=60:
                history.append(sample);self.state['samples']=history[-120:]
            throughput_state(self.state)
            if self.state.get('legacy_resume_pending'):hard='legacy pause recovery requires two valid windows above1500qph'
        self.state.update(stopped=hard is not None,hard_failure=hard,last_check_timestamp=now)
        self.training_duty=self.state.get('duty_cycle',1.)
        save_json(self.directory/'guard.json',self.state)
        with (self.directory/'collector_monitor.jsonl').open('a') as log:
            log.write(json.dumps(dict(**sample,policy_version=VERSION,decision=self.state.get('throughput_status'),
                throttle_level=self.state.get('throttle_level',0),duty_cycle=self.training_duty,hard_failure=hard,resources=resources))+'\n')
        if hard:raise RuntimeError('Research stopped: '+hard)


def extraction_recovery(shared):
    evidence=json.loads(Path(shared).read_text())
    if (evidence.get('policy_version')!=VERSION or evidence['stopped'] or not evidence.get('samples') or
        time.time()-evidence['samples'][-1]['timestamp']>180):
        raise RuntimeError('Research stopped: extraction awaits healthy adaptive resource sharing')
    return dict(policy_version=VERSION,latest_valid_qph=evidence['latest_valid_qph'],
                throughput_is_soft_constraint=True,production_actions=[])
