#!/usr/bin/env python3
"""Read-only causal observer replay across all completed scale roots."""
import argparse
import copy
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from collector.core import read
from collector.observation import HumanDetection,LegalHumanTracker,ObservationConfig,TRACK_FIELDS
from eventwm_collect import branches


def check(path):
    cfg=ObservationConfig(**read(path/'legal_observations/observation_config.json'))
    history=read(path/'legal_observations/history.json'); root_time=history[-1]['timestamp']
    bs=branches(path); count=0
    for bid,branch in [('root',None)]+[(b['branch_id'],b) for b in bs]:
        tracker=LegalHumanTracker(cfg); previous={}; rebuilt=[]
        for r in read(path/'audits/workers'/bid/'observation_diagnostics.json'):
            t=r['timestamp']
            ds=[HumanDetection(t,tuple(h['gt_position'])) for h in r['humans'] if h['detected']]
            tracks=tracker.update(t,ds)
            assert tracks==r['legal_tracks'],(bid,t,'tracker replay')
            assert sum(tr['visible_now'] for tr in tracks)==len(ds),(bid,t,'detection count')
            assert len({tr['track_id'] for tr in tracks})==len(tracks)
            for tr in tracks:
                assert set(tr)==set(TRACK_FIELDS)
                old=previous.get(tr['track_id'])
                if old and not tr['visible_now']:
                    assert all(tr[k]==old[k] for k in ('position','estimated_velocity','last_seen_time','num_observations'))
                    assert abs(tr['observation_age']-(t-tr['last_seen_time']))<1e-9
                if old and tr['visible_now']:
                    dt=t-old['last_seen_time']; valid=cfg.min_velocity_dt_s<dt<=cfg.velocity_max_gap_s
                    assert tr['velocity_valid']==valid
                    if valid: np.testing.assert_allclose(tr['estimated_velocity'],(np.array(tr['position'])-old['position'])/dt,atol=1e-12)
                previous[tr['track_id']]=copy.deepcopy(tr)
            rebuilt.append(dict(timestamp=t,human_tracks=tracks)); count+=1
        assert [r for r in rebuilt if r['timestamp']<=root_time]==[dict(timestamp=h['timestamp'],human_tracks=h['human_tracks']) for h in history]
        if branch:
            assert [r for r in rebuilt if r['timestamp']>=root_time]==[dict(timestamp=h['timestamp'],human_tracks=h['human_tracks']) for h in branch['legal_observations']]
    return count


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    manifest=read(args.output/'manifest.json'); total=0
    assert all(r['state']=='completed' for r in manifest['roots'])
    for root in manifest['roots']:
        n=check(args.output/root['path']); total+=n; print('PASS',root['root_id'],n,'observations')
    print('SAVED OBSERVATION AUDIT PASS',total)
