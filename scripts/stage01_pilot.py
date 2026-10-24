#!/usr/bin/env python
"""Bounded, training-scene integration pilot after Stage0 acceptance."""
import argparse
import concurrent.futures
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def run_case(case,configuration):
    from eventwm_graph.partial.collect import collect_root
    from eventwm_graph.partial.demo import scene_manifest
    from eventwm_graph.benchmark.map import load_certified_map
    from eventwm_graph.stage1.storage import scene_split
    if scene_split(case['scene'])!='train':raise ValueError('Pilot held-out scene refused')
    return collect_root(load_certified_map(scene_manifest(case['scene'])),Path(configuration['output'])/'dataset',
        case['seed'],configuration['stage0_gate'],configuration['queries_per_root'],configuration['max_query_steps'],
        case['node_index'],case['humans'],case['scenario'],pilot=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/stage01_partial_graph_pilot.json')
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet')
    from eventwm_graph.config import read_json,write_json
    from eventwm_graph.partial.collect import verify_gate
    from eventwm_graph.partial.storage import implementation
    cfg=read_json(args.config);verify_gate(cfg['stage0_gate']);versions=implementation();began=time.time();results=[]
    output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run_case,c,cfg):c for c in cfg['cases']}
        for future in concurrent.futures.as_completed(futures):
            case=futures[future]
            try:result={'case':case,'result':future.result()}
            except Exception as error:result={'case':case,'error':repr(error),'traceback':traceback.format_exc()}
            results.append(result);print(case,result.get('error') or result['result']['statistics'],flush=True)
            write_json(output/'pilot.json',{'implementation':versions,'seconds':time.time()-began,'roots':results,
                'source_unchanged':versions==implementation(),'training_started':False})
    if any('error' in r for r in results):raise SystemExit(1)


if __name__=='__main__':main()
