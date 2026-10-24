#!/usr/bin/env python
"""Observation-limited Stage 0/1 entry point. Never starts model training."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    demo=commands.add_parser('demo')
    demo.add_argument('--scene',default='mp3d_17DRP5sb8fy')
    demo.add_argument('--seed',type=int,default=310201001)
    demo.add_argument('--node-index',type=int,default=4)
    demo.add_argument('--humans',type=int,default=2)
    demo.add_argument('--variant',type=int,default=0)
    demo.add_argument('--scenario',choices=['natural','approach','crossing'],default='natural')
    demo.add_argument('--output',default='output/stage01_partial_graph_query_20261020/stage0')
    demo.add_argument('--no-probe',action='store_true')
    collect=commands.add_parser('collect')
    collect.add_argument('--scene',required=True)
    collect.add_argument('--seed',type=int,required=True)
    collect.add_argument('--node-index',type=int)
    collect.add_argument('--humans',type=int)
    collect.add_argument('--scenario',choices=['natural','approach','crossing'])
    collect.add_argument('--stage0-gate',required=True)
    collect.add_argument('--queries',type=int,default=24)
    collect.add_argument('--horizon',type=int,default=120)
    collect.add_argument('--output',required=True)
    validate=commands.add_parser('validate')
    validate.add_argument('root')
    render=commands.add_parser('outcomes')
    render.add_argument('root');render.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='demo':
        from eventwm_graph.partial.demo import build_demo
        import json
        print(json.dumps(build_demo(args.scene,args.output,args.seed,args.node_index,args.humans,args.variant,not args.no_probe,args.scenario),indent=2),flush=True)
    elif args.command=='collect':
        import json
        from eventwm_graph.partial.demo import scene_manifest
        from eventwm_graph.benchmark.map import load_certified_map
        from eventwm_graph.partial.collect import collect_root
        result=collect_root(load_certified_map(scene_manifest(args.scene)),args.output,args.seed,args.stage0_gate,
            args.queries,args.horizon,args.node_index,args.humans,args.scenario,pilot=True)
        print(json.dumps(result,indent=2),flush=True)
    elif args.command=='validate':
        import json
        from eventwm_graph.partial.storage import validate_root
        print(json.dumps(validate_root(args.root),indent=2),flush=True)
    elif args.command=='outcomes':
        import json
        from eventwm_graph.partial.visualize import stage1
        result=stage1(args.root,args.output)
        print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2),flush=True)


if __name__=='__main__':main()
