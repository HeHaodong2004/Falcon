"""Paired scene-clustered comparisons of already completed development scores."""
import argparse
from pathlib import Path
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import scene_interval
from eventforecastwm.window_evaluate import aggregate


def compare(directory):
    directory=Path(directory);base=read(directory/'incumbent_evaluation/result.json');result={}
    for name in ('cal_origin_cond','cal_all_uncond','cal_all_cond','cal_all_age_cond'):
        path=directory/(name+'_evaluation')/'result.json';candidate=read(path);row={}
        for population in ('origin','all_windows'):
            scores=candidate['scores'] if population=='origin' else candidate['all_windows']['summary']
            reference=base['scores'] if population=='origin' else base['all_windows']['summary']
            row[population]={h:{key:scene_interval(value['records'],key,{r['root_id']:r for r in reference[h]['records']})
                for key in ('energy','nll','mean_ade','mean_fde')} for h,value in scores.items()}
        row['occupancy']={}
        for h,value in candidate['extras'].items():
            other={r['root_id']:dict(root_id=r['root_id'],scene=r['scene'],**r['occupancy']) for r in base['extras'][h]['records'] if 'occupancy' in r}
            records=[dict(root_id=r['root_id'],scene=r['scene'],**r['occupancy']) for r in value['records'] if 'occupancy' in r]
            row['occupancy'][h]={key:scene_interval([r for r in records if r.get(key) is not None and
                other[r['root_id']].get(key) is not None],key,other) for key in ('average_precision','brier','nll')}
        row['result_sha256']=sha256(path);result[name]=row
    path=directory/'cal_all_age_cond_conditional_coverage/result.json';other=directory/'incumbent_conditional_coverage/result.json'
    if path.exists() and other.exists():
        candidate=read(path);baseline=read(other);conditional={}
        for h in (1.,3.):
            conditional[str(h)]={}
            for stratum in sorted({r['stratum'] for r in baseline['records']}):
                select=lambda data:aggregate([r for r in data['records'] if r['horizon']==h and r['stratum']==stratum],('coverage90','nll'))
                left=select(candidate);right={r['root_id']:r for r in select(baseline)}
                conditional[str(h)][stratum]={key:scene_interval(left,key,right) for key in ('coverage90','nll')}
        result['age_aware_conditional_all_windows']=conditional
    return dict(comparisons=result,baseline_sha256=sha256(directory/'incumbent_evaluation/result.json'),
                sign='candidate minus incumbent',source_sha256=sha256(__file__),confirmation=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    save_json(path,compare(args.directory))


if __name__=='__main__':main()
