#!/usr/bin/env python3
"""Metadata-only synthetic benchmark. NEVER a retained-query speed forecast."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import tempfile
import time

import eventwm_combined_runner as old
import eventwm_mass_collection as mass
from eventwm_pipeline_io import StateStore


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    base=dict(roots=[],scenes=[{'name':'synthetic'}],max_completed_roots=910,
              target_primary_queries=30000,collection_seconds=3600.,sessions=[])
    scene=dict(name='synthetic',physical_scene_id='synthetic',family='synthetic',scene_role='train')
    rows=[]
    for count in (128,5000,25000):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'snapshot';b=root/'incremental';a.mkdir();b.mkdir()
            m=copy.deepcopy(base)
            for index in range(count):
                r=mass.schedule_spec(index,[scene])
                r.update(state='failed',_pipeline_ordinal=index)
                r['attempts']=[dict(state='failed',reason='requested_legal_stratum_unavailable')]
                m['roots'].append(r)
            store=StateStore(b)
            store.flush(m,all_roots=True)
            timings={'full_snapshot':[],'incremental':[]}
            for _ in range(7):
                t=time.perf_counter();old.save_manifest(a,m);timings['full_snapshot'].append(time.perf_counter()-t)
                m['roots'][-1]['measured_revision']=_
                t=time.perf_counter();store.touch(m['roots'][-1]);store.flush(m);timings['incremental'].append(time.perf_counter()-t)
            assert store.load()==m
            result=dict(proposals=count,snapshot_bytes=(a/'manifest.json').stat().st_size,
                        snapshot_median_s=statistics.median(timings['full_snapshot']),
                        incremental_median_s=statistics.median(timings['incremental']))
            result['metadata_only_speedup']=result['snapshot_median_s']/result['incremental_median_s']
            rows.append(result);store.close()
    payload=dict(scope='Synthetic metadata persistence only; excludes all Habitat simulation, labels, validation and retained-query throughput',results=rows)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__': main()
