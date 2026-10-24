"""Investigate observed regressions from frozen ledgers; no model tuning."""
import collections,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from eventwm_graph.l2_final.common import OUT,PREVIOUS,dump
from eventwm_graph.l2_scaling_500k.compare import rows,ci


def main():
    paths=dict(A=PREVIOUS/'evaluation_arm_new/evaluation',B=OUT/'evaluation_arm_original_repaired/evaluation',C=OUT/'evaluation_arm_new/evaluation')
    reobs={};beliefs={};diagnostics={}
    fields=('association_accuracy','false_split','false_merge','duplicate','missing_antecedent_mass')
    for arm,path in paths.items():
        events={}
        for row in rows(path,'reobservation'):
            key=tuple(row.get(k) for k in ('root_id','decision','stream','physical'))
            if row.get('association_accuracy') is not None:events.setdefault(key,row)
        reobs[arm]=events;beliefs[arm]=rows(path,'beliefs')
    for left,right in [('A','B'),('A','C'),('B','C')]:
        a,b=reobs[left],reobs[right];common=a.keys()&b.keys();records=[]
        for key in common:
            r={k:a[key].get(k) for k in ('root_id','scene_id','decision','stream','physical','gap')}
            for field in fields:r[field]=b[key][field]-a[key][field];r['old_'+field]=a[key][field];r['new_'+field]=b[key][field]
            records.append(r)
        roots={}
        for arm in (left,right):
            root=collections.defaultdict(lambda:dict(zero_count=False,duplicate=False,missing=False))
            for r in beliefs[arm]:
                key=(r['scene_id'],r['root_id']);root[key]['zero_count']|=not r['true_count_support'];root[key]['duplicate']|=r['expected_duplicate_ancestries']>0;root[key]['missing']|=r['expected_missing_preferred_ancestries']>0
            roots[arm]=root
        paired=[dict(scene_id=s,root_id=r,**{k:float(roots[right][(s,r)][k])-float(roots[left][(s,r)][k]) for k in ('zero_count','duplicate','missing')}) for s,r in roots[left].keys()&roots[right].keys()]
        diagnostics[right+'_minus_'+left]=dict(unique_dependent_event_cases=len(common),independent_roots=len({r['root_id'] for r in records}),scenes=len({r['scene_id'] for r in records}),
            event_metrics={field:ci(records,field) for field in fields},
            event_counts=dict(correct_mass_reduced=sum(r['association_accuracy']<0 for r in records),duplicate_mass_increased=sum(r['duplicate']>0 for r in records),
                              both=sum(r['association_accuracy']<0 and r['duplicate']>0 for r in records),antecedent_support_reduced=sum(r['missing_antecedent_mass']>0 for r in records)),
            largest_duplicate_regressions=sorted(records,key=lambda r:r['duplicate'],reverse=True)[:12],
            root_any_state_paired_difference={key:ci(paired,key) for key in ('zero_count','duplicate','missing')})
    dump(OUT/'regression_diagnostics.json',dict(contrasts=diagnostics,
         interpretation='Repaired variants shift represented ancestry toward new birth/duplicate alternatives on real reobservations; this is not explained by arrived-coordinate inconsistency (all variants maximum error0). Complete-assignment correctness for an approximate likelihood does not establish correct physical association. Label-defined physical duplicates differ from illegal duplicate slot keys. Historical bootstrap, miss quadrature, hidden spatial/count prior, learned dynamics and finite-carrier resampling remain possible contributors; this audit does not isolate their causal contributions.',
         decision='Do not promote either repaired variant over source-pinned A; no post-evaluation hyperparameter, seed or uncertainty changes. C training artifact retained as an unpromoted candidate.',
         independence='event/stream repetitions collapsed by root then scene for CI; diagnostic extreme cases are descriptive, not fresh tests'))


if __name__=='__main__':main()
