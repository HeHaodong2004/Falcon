"""Compare completed development receipts with strict paired support identities."""
import argparse
from pathlib import Path

from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import scene_interval


def paired(rows,reference,key):
    left={r['root_id']:r for r in rows if r.get(key) is not None}
    right={r['root_id']:r for r in reference if r.get(key) is not None}
    if len(left)!=sum(r.get(key) is not None for r in rows):raise ValueError('Repeated root in paired comparison')
    if set(left)!=set(right):raise ValueError('Candidate and control metric support differ: '+key)
    if any(left[rid]['scene']!=right[rid]['scene'] for rid in left):raise ValueError('Scene identity changed')
    return dict(candidate=scene_interval(list(left.values()),key),control=scene_interval(list(right.values()),key),
                delta=scene_interval(list(left.values()),key,right))


def occupancy_rows(receipt,horizon):
    return [dict(root_id=r['root_id'],scene=r['scene'],**r['occupancy'])
            for r in receipt['extras'][horizon]['records'] if 'occupancy' in r]


def compare(reference,candidates):
    base=read(reference);result={}
    for name,path in candidates.items():
        candidate=read(path);row={}
        horizons=[h for h in ('1.0','2.0','3.0') if h in base['scores'] and h in candidate['scores']]
        for horizon in horizons:
            row[horizon]=dict(trajectory={key:paired(candidate['scores'][horizon]['records'],base['scores'][horizon]['records'],key)
                for key in ('energy','nll','mean_ade','mean_fde')},
                occupancy={key:paired(occupancy_rows(candidate,horizon),occupancy_rows(base,horizon),key)
                for key in ('average_precision','brier','nll')})
        if base.get('all_windows') is not None and candidate.get('all_windows') is not None:
            left=candidate['all_windows']['summary'];right=base['all_windows']['summary']
            if set(left)!=set(right):raise ValueError('All-window forecast horizons differ')
            row['all_windows']={h:{key:paired(value['records'],right[h]['records'],key)
                for key in ('energy','nll','mean_ade','mean_fde')} for h,value in left.items()}
        if base.get('uncertainty') is not None and candidate.get('uncertainty') is not None:
            left=candidate['uncertainty'];right=base['uncertainty']
            if (left['draws']!=right['draws'] or
                left['nominal_finite_rank_coverage']!=right['nominal_finite_rank_coverage']):
                raise ValueError('Joint density rank protocols differ')
            row['joint_coverage90']=paired(left['records'],right['records'],'joint_coverage90')
        row['receipt']=str(path);row['receipt_sha256']=sha256(path);result[name]=row
    return dict(reference=str(reference),reference_sha256=sha256(reference),comparisons=result,
                sign='candidate minus reference',confirmation=False,source_sha256=sha256(__file__))


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True)
    p.add_argument('--candidate',action='append',required=True,help='name=completed/result.json');p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    save_json(out,compare(args.reference,dict(x.split('=',1) for x in args.candidate)))


if __name__=='__main__':main()
