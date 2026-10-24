"""Finite reference checks for the unchanged inference/scoring orchestration."""
import collections
import argparse
import gzip
import json
import numbers
from .common import BASE,OUT,sha,dump

def read(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f if json.loads(x)['model']=='D_StopMove']

def run(fixture=False):
    case='evaluation_shard_0157_0158' if fixture else 'evaluation_smoke'
    full=OUT/'evaluation_arm_original'/case;focused=OUT/'evaluation_arm_original_focused'/case
    reports=[]
    for kind in ('forecasts','beliefs','reobservation','update_gains'):
        a=read(full/(kind+'.jsonl.gz'));b=read(focused/(kind+'.jsonl.gz'))
        assert len(a)==len(b);maximum=0.;different=collections.Counter()
        for left,right in zip(a,b):
            assert left.keys()==right.keys()
            for k in left:
                x,y=left[k],right[k]
                if x!=y:
                    assert isinstance(x,numbers.Real) and isinstance(y,numbers.Real),(kind,k,x,y)
                    error=abs(x-y);maximum=max(maximum,error);different[k]+=1
                    assert error<=1e-11,(kind,k,error)
        reports.append(dict(kind=kind,rows=len(a),maximum_numeric_difference=maximum,different_fields=dict(different),
            full_sha256=sha(full/(kind+'.jsonl.gz')),focused_sha256=sha(focused/(kind+'.jsonl.gz'))))
    # Separately connect the deployed implementation to committed original scores.
    key=lambda r:tuple(r.get(k) for k in ('root_id','model','cohort','decision','stream','physical','H'))
    focused_rows={key(r):r for r in read(focused/'forecasts.jsonl.gz')};matched=0;maximum=0.
    archived=BASE/'docs/l2/intermittent_wm_20261022/artifacts/evaluation/forecasts.pre_annotation.jsonl.gz'
    with gzip.open(archived,'rt') as f:
        for line in f:
            left=json.loads(line);k=key(left)
            if k not in focused_rows:continue
            matched+=1;right=focused_rows[k];assert left.keys()==right.keys()
            for field in left:
                x,y=left[field],right[field]
                if x!=y:
                    assert isinstance(x,numbers.Real) and isinstance(y,numbers.Real)
                    error=abs(x-y);maximum=max(maximum,error);assert error<=1e-11
    assert matched==len(focused_rows)
    dump(OUT/('evaluation_reappearance_reproduction.json' if fixture else 'evaluation_reproduction.json'),dict(full_versus_focused=reports,committed_D_rows=matched,
        committed_maximum_numeric_difference=maximum,committed_reference_sha256=sha(archived),
        all_passed=True,tolerance=1e-11,case=case,scope='Same original checkpoint and frozen roots; exact source-only optimization equivalence plus full scientific evaluator vs D-only execution. This validates accounting/control flow, not scientific performance.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reappearance',action='store_true');run(p.parse_args().reappearance)
