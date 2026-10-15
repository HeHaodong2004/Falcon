"""Root-matched intrinsic forecasting comparisons with clustered intervals."""
import argparse
import json
from pathlib import Path
import numpy as np
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventbeliefwm.metrics import paired_root_interval
from eventforecastwm.benchmark import scene_interval


def compare(paths,reference):
    data={name:read(path) for name,path in paths.items()};base=data[reference]
    results={}
    for name,value in data.items():
        if value['protocol_sha256']!=base['protocol_sha256']:raise ValueError('Different development protocol')
        horizons={}
        for h,score in value['scores'].items():
            baseline={r['root_id']:r for r in base['scores'][h]['records']}
            rows=score['records']
            if {r['root_id'] for r in rows}!=set(baseline):raise ValueError('Different evaluation population')
            keys=('energy','nll','mean_ade','mean_fde','map_ade','map_fde')
            pairs={}
            for k in keys:
                if {r['root_id'] for r in rows if k in r}!={rid for rid,r in baseline.items() if k in r}:
                    raise ValueError('Different support masks for '+k)
                selected=[r for r in rows if k in r]
                pairs[k]=dict(root=paired_root_interval([r[k] for r in selected],
                    [baseline[r['root_id']][k] for r in selected],repetitions=1000),scene=scene_interval(rows,k,baseline))
            strata={};extra={r['root_id']:r for r in value['extras'][h]['records']}
            for key in ('lost','dense'):
                for flag in (False,True):
                    rr=[r for r in rows if 'energy' in r and extra[r['root_id']][key]==flag]
                    strata[key+'_'+str(flag)]=dict(roots=len(rr),
                        metrics={k:float(np.mean([r[k] for r in rr])) if rr else None for k in keys})
            horizons[h]=dict(support={k:score[k] for k in ('roots','trajectory_roots','agents','physical_roots')},
                metrics=score['metrics'],extras=value['extras'][h]['metrics'],occupancy=value['extras'][h]['occupancy'],
                paired_difference=pairs,strata=strata)
        results[name]=dict(file=str(paths[name]),sha256=sha256(paths[name]),checkpoint_sha256=value['checkpoint_sha256'],horizons=horizons)
    return dict(reference=reference,models=results,scope='Exploratory train-internal development; root and scene paired intervals; no final-confirmation claim',
        fixed_k_note='Only benchmark schema>=2 has independent fixed-K residual paths',
        benchmark_schemas={name:v['schema_version'] for name,v in data.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',action='append',required=True,help='NAME=path/to/result.json')
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    paths=dict(x.split('=',1) for x in a.result);result=compare(paths,a.reference);save_json(out,result)
    for name,value in result['models'].items():
        for h,v in value['horizons'].items():
            print(json.dumps(dict(model=name,horizon=h,roots=v['support']['trajectory_roots'],metrics=v['metrics'],
                energy_difference=v['paired_difference']['energy'])))


if __name__=='__main__':main()
