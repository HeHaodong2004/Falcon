"""Seal descriptive research/collector resource observations, with no control API."""
import collections
import json
from pathlib import Path
import re
import time

import numpy as np

from eventwm_graph.l2_partial.snapshot import sha
from .data import OUT
from .monitor import sample


def quantiles(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),minimum=float(a.min()),p50=float(np.median(a)),
        p95=float(np.quantile(a,.95)),maximum=float(a.max())) if len(a) else None


def main():
    if (OUT/'CAMPAIGN_ACTIVE').exists():
        raise ValueError('finish the research-only monitor before sealing its log')
    path=OUT/'resources.jsonl'
    with path.open('a') as stream:stream.write(json.dumps(sample())+'\n')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    valid=[r for r in rows if 'collector' in r]
    rates=[]
    for a,b in zip(valid,valid[1:]):
        elapsed=b['time']-a['time'];delta=b['collector']['outcomes']-a['collector']['outcomes']
        if elapsed>0 and delta>=0:rates.append(delta/elapsed)
    sources={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in (OUT/'train.npy',OUT/'dev.npy',OUT/'snapshot/manifest.json')}
    total_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    pressure=[float(re.search(r'some avg10=([0-9.]+)',r['cpu_pressure']).group(1)) for r in valid]
    result=dict(created=time.time(),resource_log_sha256=sha(path),samples=len(valid),
        failed_samples=[r for r in rows if 'error' in r],
        first=valid[0],last=valid[-1],elapsed_seconds=valid[-1]['time']-valid[0]['time'],
        collector=dict(worker_counts=sorted({r['collector']['active_worker_count'] for r in valid}),
            configuration_identities=sorted({r['collector']['configuration_identity'] for r in valid}),
            restart_counts=sorted({r['collector']['collector_restart_count'] for r in valid}),
            automatic_training_values=sorted({r['collector']['automatic_training'] for r in valid}),
            status_samples=dict(collections.Counter(r['collector']['collection_status'] for r in valid)),
            outcome_increase=valid[-1]['collector']['outcomes']-valid[0]['collector']['outcomes'],
            root_increase=valid[-1]['collector']['roots']-valid[0]['collector']['roots'],
            interval_outcomes_per_second=quantiles(rates)),
        host=dict(load_1m=quantiles([r['load'][0] for r in valid]),
            cpu_pressure_some_avg10_percent=quantiles(pressure),
            available_ram_bytes=quantiles([int(r['memory_available'].split()[0])*1024 for r in valid]),
            available_disk_bytes=quantiles([r['disk_free'] for r in valid])),
        research_output_bytes_before_archive=total_bytes,input_artifacts=sources,
        semantics='Read-only status/host observations; load is not CPU utilization. '
            'Shared-host rates do not establish a causal absence of collection slowdown. '
            'No collector control or configuration writes; only the research monitor stopped.')
    (OUT/'resource_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(samples=len(valid),collector=result['collector'],
        research_output_bytes_before_archive=total_bytes)),flush=True)


if __name__=='__main__':main()
