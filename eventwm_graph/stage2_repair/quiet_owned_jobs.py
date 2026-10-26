"""Temporarily suspend only this repair's owned background jobs for profiling.

The primary CUDA0 fit continues. No collector, Stage1 job or external process
is selected. PID start times protect against reuse; all resumptions use finally.
"""
from contextlib import contextmanager
from datetime import datetime,timezone
import os
from pathlib import Path
import signal
import sys
from .diagnose import OUT,read,save


def starttime(path):
    # Linux task names can contain spaces and parentheses; field22 follows
    # the final ') ' delimiter, not a fixed whitespace token in the full line.
    return (path/'stat').read_text().rpartition(') ')[2].split()[19]


@contextmanager
def quiet_jobs(name,pause_primary=False,extra_training_names=()):
    expanded=OUT/'expanded_train_range_jobs_16384.json';owned_preparation={j['pid']for j in read(expanded)}if expanded.exists()else set();paused=[]
    resolution=OUT/'range_resolution256_jobs.json'
    owned_resolution={j['pid']for j in read(resolution)['jobs']}if resolution.exists()else set()
    sensor256=OUT/'sensor256_preparation_jobs.json'
    owned256={j['pid']for j in read(sensor256)['jobs']}if sensor256.exists()else set()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:args=[a.decode()for a in (path/'cmdline').read_bytes().split(b'\0')if a]
        except (OSError,UnicodeError):continue
        if len(args)<3 or args[:2]!=[sys.executable,'scripts/stage2_segment_run.py']:continue
        module=args[2];own=(int(path.name)in owned_preparation and module=='eventwm_graph.stage2_repair.range_train_expand')or(module=='eventwm_graph.stage2_repair.range_online_surface_train'and 'range_surface_online4096'in args)or(module=='eventwm_graph.stage2_repair.online_select'and 'range_online4096'in args)or(module=='eventwm_graph.stage2_repair.range_full_evaluate'and any(n in args for n in ['legal_range_support_pretrained256','legal_range_online4096_epoch01']))or(pause_primary and module=='eventwm_graph.stage2_repair.range_online_train'and 'range_online4096'in args)
        own=own or(module in ['eventwm_graph.stage2_repair.range_online_train','eventwm_graph.stage2_repair.online_select','eventwm_graph.stage2_repair.range_full_evaluate']and any(n in args for n in extra_training_names))or(module=='eventwm_graph.stage2_repair.online_select'and 'online_cardinal256_control'in args)
        own=own or(int(path.name)in owned_resolution and module=='eventwm_graph.stage2_repair.range_branch_replay'and 'range_branches_256'in args)
        own=own or(int(path.name)in owned256 and module in ['eventwm_graph.stage2_repair.range_train_expand','eventwm_graph.stage2_repair.range_full_replay'])
        if not own:continue
        try:
            pid=int(path.name);start=starttime(path);os.kill(pid,signal.SIGSTOP);paused.append(dict(pid=pid,starttime=start,args=args))
        except OSError:pass
    metadata=dict(started_utc=datetime.now(timezone.utc).isoformat(),paused_owned_jobs=paused,pause_primary=pause_primary,unchanged='All unrelated work continues. No collector/Stage1 job selected. Owned primary fit is paused only when explicitly requested.');path=OUT/(name+'_load_isolation.json');save(path,metadata)
    try:yield metadata
    finally:
        for item in paused:
            p=Path('/proc')/str(item['pid'])
            try:
                if p.exists()and starttime(p)==item['starttime']:os.kill(item['pid'],signal.SIGCONT)
            except OSError:pass
        metadata['resumed_utc']=datetime.now(timezone.utc).isoformat();save(path,metadata)
