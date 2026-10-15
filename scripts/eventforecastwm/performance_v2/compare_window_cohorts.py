"""Paired actual-window cohorts for completed physical-control evaluations.

Each selected window contains the named condition; metrics still score all
supported humans in that window. This is explicitly not a per-agent conditional
density rank. Censused labels are evaluation metadata, never inference inputs.
"""
import argparse
from pathlib import Path

from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.window_evaluate import aggregate
from eventforecastwm.benchmark_reuse import CooperativeGuard
from .compare_completed import paired


def compare(cfg,guard=None):
    reference=read(cfg['reference']);census=read(cfg['census'])
    metadata={(r['root_id'],r['window']):r for r in census['records'] if r['split']=='dev'}
    result={};keys=('energy','nll','mean_ade','mean_fde')
    for name,path in cfg['candidates'].items():
        candidate=read(path);result[name]={}
        for horizon in ('1.0','3.0'):
            def grouped(receipt):
                groups={key:[] for key in ('all','age_ge1','age_ge2','age_ge3','no_current_observation','dense_scene','long_history')}
                for row in receipt['all_windows']['scores'][horizon]['records']:
                    if 'energy' not in row:continue
                    meta=metadata[(row['root_id'],row['window_index'])]
                    support=meta['horizons'][str(int(float(horizon)))];ages=support['ages']
                    if len(ages)!=row['agents']:raise ValueError('Census/evaluation target support changed')
                    conditions=dict(all=True,age_ge1=max(ages)>=1.,age_ge2=max(ages)>=2.,age_ge3=max(ages)>=3.,
                        no_current_observation=meta['visible']==0,dense_scene=meta['physical_count_target_only']>=6,long_history=meta['history_s']>=3.)
                    for key,keep in conditions.items():
                        if keep:groups[key].append(row)
                return {key:aggregate(rows,keys) for key,rows in groups.items()}
            baseline=grouped(reference);new=grouped(candidate)
            result[name][horizon]={}
            for cohort,rows in new.items():
                if guard:guard.check()
                result[name][horizon][cohort]={key:paired(rows,baseline[cohort],key) for key in keys}
    return dict(status='window_cohort_comparison_complete',config=cfg,comparisons=result,
        identities={path:sha256(path) for path in (cfg['reference'],cfg['census'],*cfg['candidates'].values())},
        estimand='Equal selected windows within each root, then equal roots; the condition identifies windows, not an agent-only marginal',
        source_sha256=sha256(__file__),confirmation=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);cfg=read(parser.parse_args().config)
    out=Path(cfg['run'])/'result.json'
    if out.exists():raise FileExistsError('Completed cohort comparison is immutable')
    out.parent.mkdir(parents=True,exist_ok=True)
    save_json(out,compare(cfg,CooperativeGuard(out.parent)))


if __name__=='__main__':main()
