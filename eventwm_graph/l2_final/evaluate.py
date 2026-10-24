"""Source-isolated, unchanged frozen development scoring and accounting."""
import argparse
import json
from .common import BASE,OUT,OLD,PREVIOUS,sha,dump

NAMES={'B':'original_repaired','C':'new','B_public':'original_public','C_public':'public',
       'B_public_final':'original_public_final','C_public_final':'public_final'}


def module():
    from eventwm_graph.l2_scaling_500k import evaluation
    evaluation.BASE=BASE;evaluation.OUT=OUT;evaluation.OLD=OLD;evaluation.PRIOR=PREVIOUS
    return evaluation


def setup(arm):
    root,target=module().setup(NAMES[arm])
    if 'public' in arm:
        path=root/'eventwm_graph/l2_intermittent/evaluate.py';text=path.read_text()
        original='m.from_legacy(initial, seed_for(root, stream + 10000))'
        assert original in text
        text=text.replace(original,"public_initialize(m, legal, root, stream)")
        helper='''
def public_initialize(model,legal,root,stream):
    start=time.perf_counter()
    record=dict(root_id=root,stream=stream,status='attempted')
    try:
        belief=model.initialize(legal['history'],legal['partial_graph'])
        record.update(status='supported_finite_approximation',fallback=belief.diagnostics.get('bootstrap_duplicate_fallback',False),
                      bootstrap_support_status=belief.diagnostics.get('bootstrap_support_status','FINITE_APPROXIMATION'),
                      fingerprint=belief.fingerprint(),components=len(belief.components))
        return belief
    except Exception as error:
        record.update(status='INITIALIZATION_FAILED',error=repr(error))
        raise
    finally:
        record['milliseconds']=(time.perf_counter()-start)*1000
        with (OUT/'public_initialization.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\\n')

'''
        position=text.index('def run(');text=text[:position]+helper+text[position:]
        path.write_text(text)
        dump(root/'public_initialization_protocol.json',dict(roots='prespecified first528 frozen dev roots',
             difference=('restored validated legal-prefix initializer with receipt, lifetime and provenance repairs' if arm.endswith('_final') else 'rejected causal repaired prefix replay instead of common legacy bootstrap'),
             scores='unchanged frozen functions, ancestry and censoring'))
        dump(root/'source_manifest.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*.py')})
    return root,target


def merge(arm):
    from eventwm_graph.l2_scaling_500k import merge as merger
    merger.BASE=BASE;merger.OUT=OUT;merger.OLD=OLD
    merger.run(NAMES[arm])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['setup','run','merge']);parser.add_argument('--arm',choices=NAMES,required=True)
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=2110);a=parser.parse_args()
    if a.command=='setup':setup(a.arm)
    elif a.command=='merge':merge(a.arm)
    else:module().execute(NAMES[a.arm],a.start,a.stop,0)
