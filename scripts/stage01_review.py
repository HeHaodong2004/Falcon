#!/usr/bin/env python
"""Reproduce the separately recorded correctness gates; never trains models."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    a=commands.add_parser('stage0')
    a.add_argument('--directory',required=True)
    a.add_argument('--refresh-probes',action='store_true')
    r=commands.add_parser('reprocess')
    r.add_argument('--source',required=True);r.add_argument('--destination',required=True)
    s=commands.add_parser('stage1');s.add_argument('--stage0-gate',required=True)
    s.add_argument('--dataset',action='append',required=True);s.add_argument('--replay-report',action='append',required=True)
    s.add_argument('--independent-review',required=True);s.add_argument('--output',required=True)
    s.add_argument('--root-lineage',required=True)
    args=p.parse_args()
    if args.command=='stage0':
        from eventwm_graph.partial.acceptance import native_static_probe,accept_stage0
        from eventwm_graph.stage1.storage import read_gzip
        directory=Path(args.directory)
        traces=sorted((directory/'traces').glob('*.json.gz'))
        if args.refresh_probes:
            for trace in traces:
                if read_gzip(trace)['snapshots'][-1]['queries']:
                    print('probe',trace.name,flush=True)
                    native_static_probe(trace,directory/'probes',3)
        gate=accept_stage0(traces,sorted((directory/'probes').glob('*.json.gz')),directory/'acceptance')
        print(gate['status'],gate['checks'],gate['evidence'],flush=True)
        if gate['status']!='PASS':raise SystemExit(1)
    elif args.command=='reprocess':
        from eventwm_graph.partial.acceptance import reprocess_query_traces
        print(reprocess_query_traces(args.source,args.destination),flush=True)
    elif args.command=='stage1':
        from eventwm_graph.partial.stage1_acceptance import accept
        gate=accept(args.stage0_gate,args.dataset,args.replay_report,args.output,args.independent_review,args.root_lineage)
        print(gate['status'],gate['checks'],gate['evidence'],flush=True)
        if gate['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
