#!/usr/bin/env python
"""Bounded real Stage0 development + static probes; no Stage1 until acceptance."""
import argparse
import concurrent.futures
import multiprocessing
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def run_case(case,output):
    from eventwm_graph.partial.demo import build_demo
    from eventwm_graph.partial.acceptance import native_static_probe
    from eventwm_graph.partial.visualize import stage0
    from eventwm_graph.stage1.storage import scene_split
    if scene_split(case['scene'])!='train':raise ValueError('development_case_is_held_out')
    summary=build_demo(case['scene'],Path(output)/'traces',case['seed'],case['node_index'],case['humans'],
                       case['variant'],True,case['scenario'])
    stage0(summary['trace'],Path(output)/'visualizations')
    result={'case':case,'summary':summary,'probe':None}
    if summary['queries']:
        probe=native_static_probe(summary['trace'],Path(output)/'probes',3)
        result['probe']={k:v for k,v in probe.items() if k!='queries'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/stage01_partial_graph_development.json')
    p.add_argument('--output',default='output/stage01_partial_graph_query_20261020/stage0_current')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--accept',action='store_true')
    args=p.parse_args();os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet')
    from eventwm_graph.config import read_json,write_json
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    cases=read_json(args.config)['cases'];reports=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run_case,case,str(output)):case for case in cases}
        for future in concurrent.futures.as_completed(futures):
            case=futures[future]
            try:report=future.result()
            except Exception as error:report={'case':case,'error':repr(error)}
            reports.append(report);write_json(output/'campaign.json',reports)
            print(case['label'],report.get('error') or {k:report['summary'][k] for k in ('queries','query_sources','human_bands','probe','wall_seconds')},flush=True)
    if args.accept:
        from eventwm_graph.partial.acceptance import accept_stage0
        traces=sorted((output/'traces').glob('*.json.gz'));probes=sorted((output/'probes').glob('*.json.gz'))
        gate=accept_stage0(traces,probes,output/'acceptance');print(gate['status'],gate['checks'],gate['evidence'],flush=True)
        if gate['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
