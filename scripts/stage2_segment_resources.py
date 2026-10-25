"""Read-only process-tree memory observation; never attaches to execution."""
import argparse
import json
from pathlib import Path
import time
from eventwm_graph.stage2_segment.common import OUT,dump


def measure(pid,proportional=False):
    root=Path('/proc')/str(pid);result=dict(pid=pid)
    try:
        for line in (root/'status').read_text().splitlines():
            if line.startswith(('VmRSS:','VmHWM:')):result[line.split(':')[0]]=int(line.split()[1])*1024
        if proportional:
            for line in (root/'smaps_rollup').read_text().splitlines():
                if line.startswith(('Pss:','Private_Clean:','Private_Dirty:')):result[line.split(':')[0]]=int(line.split()[1])*1024
    except (OSError,ProcessLookupError):return None
    return result


def main(pid,name):
    dest=OUT/'resources'/name;dest.mkdir(parents=True,exist_ok=False)
    start=time.time();peak_rss=0;peak_pss=0;highwater={};samples=0;pids={pid}
    with (dest/'samples.jsonl').open('x') as log:
        while Path('/proc/%d'%pid).exists():
            for ancestor in list(pids):
                try:pids.update(map(int,Path('/proc/%d/task/%d/children'%(ancestor,ancestor)).read_text().split()))
                except OSError:pass
            proportional=samples%10==0;rows=[r for p in sorted(pids) if (r:=measure(p,proportional)) is not None]
            rss=sum(r.get('VmRSS',0) for r in rows);peak_rss=max(peak_rss,rss)
            if proportional:peak_pss=max(peak_pss,sum(r.get('Pss',0) for r in rows))
            for row in rows:highwater[row['pid']]=max(highwater.get(row['pid'],0),row.get('VmHWM',0))
            log.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),proportional=proportional,processes=rows))+'\n');log.flush()
            samples+=1;time.sleep(1.)
    dump(dest/'summary.json',dict(pid=pid,processes=len(pids),samples=samples,observation_seconds=time.time()-start,
        observed_peak_aggregate_rss_bytes=peak_rss,observed_peak_aggregate_pss_bytes=peak_pss,
        lifetime_process_rss_highwaters=highwater,sum_lifetime_rss_highwaters_bytes=sum(highwater.values()),
        scope='observation starts during extraction; observed aggregate peaks are lower bounds for the complete run; lifetime per-process VmHWM includes time before observation; summed RSS double-counts shared pages, PSS does not',
        sampling='RSS/high-water1second, aggregate proportional resident memory10seconds'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);parser.add_argument('--name',required=True);main(**vars(parser.parse_args()))
