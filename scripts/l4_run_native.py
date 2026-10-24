"""Sequential matched native research runs; existing destinations are refused."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import habitat_sim
from eventwm_graph.benchmark.map import load_certified_map
from eventwm_graph.l4.native import run_case,CHECKPOINT


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--variants',default='B0,B1,B2')
    p.add_argument('--case');a=p.parse_args();protocol=json.loads(a.protocol.read_text())
    cases=[c for c in protocol['cases'] if not a.case or a.case==c['case_id']]
    if not cases:raise ValueError('No matching cases')
    variants=a.variants.split(',')
    if 'B3' in variants and variants!=['B3']:raise ValueError('Run visual Falcon in a fresh process')
    a.output.mkdir(parents=True,exist_ok=False)
    sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('eventwm_graph/l4').glob('*.py')}
    receipt=dict(protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),sources=sources,
                 cases=[c['case_id'] for c in cases],variants=variants,threads=1)
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2))
    maps={};model=None
    if 'B2' in variants:
        from eventwm_graph.l2_intermittent.model import IntermittentWorldModel
        model=IntermittentWorldModel.load(CHECKPOINT)
    for case in cases:
        try:
            if case['manifest'] not in maps:maps[case['manifest']]=load_certified_map(case['manifest'])
            hmap=maps[case['manifest']]
        except Exception:
            (a.output/(case['case_id']+'_map_error.txt')).write_text(traceback.format_exc());continue
        for variant in variants:
            folder=a.output/(case['case_id']+'_'+variant)
            try:
                r=run_case(case,variant,folder,hmap,model)
                print(json.dumps({k:r.get(k) for k in ('case_id','variant','status','metrics','compile_error','wall_seconds')}),flush=True)
            except Exception:
                folder.mkdir(exist_ok=True);(folder/'error.txt').write_text(traceback.format_exc())
                print(case['case_id'],variant,'INTEGRATION_ERROR',flush=True)
    receipt['source_unchanged']=all(Path(f).exists() and hashlib.sha256(Path(f).read_bytes()).hexdigest()==sha for f,sha in sources.items())
    (a.output/'completed_receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
