#!/usr/bin/env python3
"""Read-only post-smoke audit: seals, FALCON PSC oracle, native-v2 replay equivalence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import habitat_sim
from falcon.additional_metric import PersonalSpaceCompliance
from collector.core import read,write,digest,load_wm_input
from collector.v3 import social_outcome_labels
from eventwm_scale import verify_root
from eventwm_collect import branches
from eventwm_v3_visualize import ReferencePSC


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--legacy',type=Path,default=REPO/'datasets/raw/eventwm_human_event_v2_smoke_03')
    p.add_argument('--output',type=Path,default=REPO/'output/eventwm_v3_validation/native_audit.json')
    a=p.parse_args();manifest=read(a.dataset/'manifest.json');old=read(a.legacy/'manifest.json')
    cases=read(REPO/'evaluation/controlled50/scenes.json');results=[];checks=0
    class Metric:
        def get_metric(self):return 1
    for root in manifest['roots']:
        assert root['state']=='completed';verify_root(a.dataset,root,deep=True)
        path=a.dataset/root['path'];queries={q['query']['query_id']:q['query'] for q in read(path/'queries/queries.json')}
        raw=branches(path);saved=read(path/'labels/labels.json')
        assert saved==[social_outcome_labels(b,queries[b['query_id']],root['human_count']) for b in raw]
        pf=habitat_sim.PathFinder();scene=next(s for s in cases if s['name']==root['scene'])
        assert pf.load_nav_mesh(str(REPO/'evaluation/controlled50'/scene['navmesh']))
        for b in raw:
            for row in [b['truth'][0],b['truth'][len(b['truth'])//2],b['truth'][-1]]:
                # Compare against the actual FALCON class, not another handwritten predicate.
                for h,evidence in zip(row['humans'],row['social_metric']['distances']):
                    sim=ReferencePSC(pf,np.array(row['robot']['position']),np.array(h['position']))
                    sim.num_articulated_agents=2
                    m=PersonalSpaceCompliance.__new__(PersonalSpaceCompliance)
                    m._sim=sim;m._use_geo_distance=True;m._compliant_steps=0;m._num_steps=0
                    m.update_metric(observations={},episode=SimpleNamespace(info={'human_num':1}),
                        task=SimpleNamespace(measurements=SimpleNamespace(measures={'num_steps':Metric()})))
                    assert (m.get_metric()==0)==(evidence['distance']<1.0)
                    checks+=1
            legal=load_wm_input(path,b['query_id'])
            assert legal['history_images'] is None
            assert not {'social_metric','social_outcome_class','social_distance_violation','native_human_collision'} & set(legal['query'])
        counterpart=next(r for r in old['roots'] if (r['scene'],r['human_count'])==(root['scene'],root['human_count']))
        previous=branches(a.legacy/counterpart['path'])
        assert len(previous)==len(raw)
        for before,after in zip(previous,raw):
            stripped=copy.deepcopy(after['truth'])
            for row in stripped:row.pop('social_metric')
            assert before['truth']==stripped, f"Physics changed: {root['root_id']} {after['branch_id']}"
            assert before['controls']==after['controls']
            assert before['legal_observations']==after['legal_observations']
        assert read(path/'legal_observations/history.json')==read(a.legacy/counterpart['path']/'legal_observations/history.json')
        assert not list(path.rglob('*.png')) and not list(path.rglob('*.mp4')) and not list(path.rglob('*images*.npz'))
        results.append(dict(root_id=root['root_id'],sealed_files=len(root['seals']),
            native_v2_replay_equivalent=True,labels_recomputed=True,structured_loader_pass=True))
    before=read(REPO/'output/eventwm_v3_validation/legacy_hashes_before.json')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
    report=dict(status='PASS',roots=results,actual_falcon_psc_comparisons=checks,
        legacy_manifests_markers_unchanged=len(before),all_native_controls_truth_legal_history_unchanged=True)
    write(a.output,report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
