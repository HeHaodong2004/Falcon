"""Light read-only status of fixed-duration Stage2 service tests."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--path',required=True);args=parser.parse_args();root=Path(args.path)
    result=dict(utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),complete=(root/'COMPLETE.json').exists(),scenarios=[],failures=[])
    for path in root.rglob('FAILED.json'):result['failures'].append(dict(path=str(path),contents=json.loads(path.read_text())))
    for folder in sorted(root.glob('workers_*')):
        path=folder/'telemetry.jsonl';rows=[]
        if path.exists():
            for line in path.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:continue # an in-flight final line is retried next poll
        summary=folder/'summary.json';entry=dict(workers=int(folder.name.split('_')[1]),completed=summary.exists(),minutes=len(rows),calls=0,graphs=0)
        if rows:
            entry['calls']=sum(v['calls'] for r in rows for v in r['pooled'].values());entry['graphs']=4*entry['calls']
            last=rows[-1];workers=last['workers']
            entry.update(latest_p95_ms={k:v['p95_ms'] for k,v in last['pooled'].items()},
                latest_max_worker_rss_gib=max(w['memory']['Rss'] for w in workers)/2**30,
                latest_service_worker_pss_gib=sum(w['memory']['Pss'] for w in workers)/2**30,
                latest_whole_device_gib=max(w['memory'].get('whole_device_used',0) for w in workers)/2**30,
                exact_proofs=sum(w['counters'].get('exact_proofs',0) for r in rows for w in r['workers']),
                minimum_proof_roots_seen=min(w['proof_roots_seen'] for w in workers),
                gc_policy_unchanged=all(w['gc_enabled'] for r in rows for w in r['workers']))
        result['scenarios'].append(entry)
    result['total_calls']=sum(s['calls'] for s in result['scenarios']);result['total_graphs']=4*result['total_calls']
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
