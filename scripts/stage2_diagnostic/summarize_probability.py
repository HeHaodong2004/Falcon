"""Hierarchical probability/threshold audit, with paired scene bootstrap."""
import argparse
from collections import defaultdict
import gzip
import json
import time
import numpy as np
from .common import OUT,EVALUATION,THRESHOLDS,EDGES,extended_rates,save,digest
from .statistics import rank_metrics,quantiles

RULES=['argmax']+[f'threshold_{t:.2f}' for t in THRESHOLDS]

def metrics(cms,scores,hist,cal,observed,observed_norm=None):
    h=hist[3];binary=rank_metrics(h[0],h[1]+h[2])
    mass=cal[0];calbins=[]
    for j in range(20):
        calbins.append(dict(lower=j/20,upper=(j+1)/20,mass=float(mass[j]),
            probability=float(cal[1,j]/max(mass[j],1e-30)),frequency=float(cal[2,j]/max(mass[j],1e-30))))
    distributions={}
    for gt,label in enumerate(('UNKNOWN','FREE','BLOCKED')):
        distributions[label]={variable:quantiles(hist[i,gt]) for i,variable in enumerate(('P_U','P_F','P_B','P_R'))}
    result=dict(rules={rule:extended_rates(cm) for rule,cm in zip(RULES,cms)},
        probability=dict(multiclass_logloss=float(scores[0]),multiclass_brier=float(scores[1]),
            binary_revelation_logloss=float(scores[2]),binary_revelation_brier=float(scores[3]),
            observed_multiclass_logloss=float(observed[0]/max(observed[2],1e-30)),
            observed_multiclass_brier=float(observed[1]/max(observed[2],1e-30)),
            revelation_rank=binary,revelation_calibration=dict(ece=float(np.abs(cal[1]-cal[2]).sum()),bins=calbins)),
        distributions=distributions)
    result['probability']['observed_score_weighting']='original evaluator: hierarchy of raw revealed-score sums/counts, ratio after aggregation; differs from cell-normalized histogram conditioning'
    if observed_norm is not None:
        result['probability']['observed_cell_normalized_logloss']=float(observed_norm[0]/max(observed_norm[2],1e-30))
        result['probability']['observed_cell_normalized_brier']=float(observed_norm[1]/max(observed_norm[2],1e-30))
    no_reveal=cms[0].sum(1); cm=np.zeros((3,3));cm[:,0]=no_reveal
    prevalence=binary['prevalence']
    result['no_revelation_baseline']=dict(hard=extended_rates(cm),probabilities=[1.,0.,0.],
        multiclass_brier=float(2*prevalence),multiclass_logloss=float(-np.log(1e-9)*prevalence),
        binary_revelation_brier=float(prevalence),binary_revelation_logloss=float(-np.log(1e-9)*prevalence),
        caveat='deliberately degenerate no-growth distribution; not a competitive fitted probability baseline; empty-unionIoU0')
    return result

def bootstrap(scene_cms,indices=(0,),seed=2525):
    rng=np.random.default_rng(seed);n=len(scene_cms);draws=defaultdict(list)
    for _ in range(2000):
        sample=scene_cms[rng.integers(0,n,n)].mean(0)
        base=extended_rates(sample[0])
        for index in indices:
            r=extended_rates(sample[index])
            for key in ('label_aware_delta_iou','revelation_iou','revelation_precision','revelation_recall','revelation_f1',
                        'terminal_label_agreement_on_true_revelations','false_revelation_rate','wrong_free_blocked_rate',
                        'false_free_unknown_fraction','false_free_blocked_fraction'):
                draws[(index,key)].append(r[key]);draws[(index,key,'difference')].append(r[key]-base[key])
            for cls in ('free','blocked'):
                for key in ('precision','recall','f1'):
                    name=cls+'_'+key
                    draws[(index,name)].append(r[cls][key]);draws[(index,name,'difference')].append(r[cls][key]-base[cls][key])
    result={}
    for index in indices:
        result[RULES[index]]={}
        for key,values in draws.items():
            if key[0]!=index:continue
            name=key[1]+('_minus_argmax' if len(key)==3 else '')
            result[RULES[index]][name]=np.quantile(values,[.025,.975]).tolist()
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    from eventwm_graph.stage2.data import source_rows
    rows,_=source_rows('dev');scene_data={};pooled=np.zeros((14,3,3),np.int64);poolhist=np.zeros((4,3,len(EDGES)-1),np.int64)
    rootrecords=[];total=0;start=time.perf_counter();zero_counts=np.zeros((14,2),np.int64)
    cohort_scenes=defaultdict(lambda:defaultdict(list))
    for row in rows:
        path=OUT/'probability/roots'/(row['root_id']+'.json.gz')
        if not path.exists():
            if args.partial:continue
            raise FileNotFoundError(path)
        with gzip.open(path,'rt') as f:r=json.load(f)
        records=r['records'];rootrecords.append((r['scene_id'],r['root_id'],records))
        total+=len(records);cms=np.array([v['confusion'] for v in records],np.int64);pooled+=cms.sum(0)
        rootcm=cms.mean(0);scores=np.mean([v['scores'] for v in records],axis=0);obs=np.mean([v['observed_scores'] for v in records],axis=0)
        obsnorm=np.mean([np.asarray(v['observed_scores'])/max(v['candidates'],1) for v in records],axis=0)
        scene=r['scene_id']
        if scene not in scene_data:
            scene_data[scene]=dict(roots=0,cm=np.zeros_like(rootcm),scores=np.zeros(4),obs=np.zeros(3),obsnorm=np.zeros(3),
                hist=np.zeros((4,3,len(EDGES)-1)),cal=np.zeros((3,20)))
        dest=scene_data[scene];dest['roots']+=1;dest['cm']+=rootcm;dest['scores']+=scores;dest['obs']+=obs;dest['obsnorm']+=obsnorm
        with np.load(path.parent/(r['root_id']+'.hist.npz')) as h:
            dest['hist']+=h['histogram'];dest['cal']+=h['calibration'];poolhist+=h['pooled']
        for rec in records:
            if rec['zero_new']:
                cm=np.array(rec['confusion']);predicted=cm[:,:,1:].sum((1,2));zero_counts[:,0]+=(predicted==0);zero_counts[:,1]+=predicted
        for group in {g for rec in records for g in rec['groups']}:
            cohort_scenes[group][scene].append(np.mean([rec['confusion'] for rec in records if group in rec['groups']],axis=0))
    names=sorted(scene_data);arrays={}
    for key in ('cm','scores','obs','obsnorm','hist','cal'):
        arrays[key]=np.stack([scene_data[s][key]/scene_data[s]['roots'] for s in names])
    avg={k:v.mean(0) for k,v in arrays.items()}
    result=metrics(avg['cm'],avg['scores'],avg['hist'],avg['cal'],avg['obs'],avg['obsnorm'])
    result.update(roots=len(rootrecords),scenes=len(names),segments=total,
        pooled_confusions={rule:cm.tolist() for rule,cm in zip(RULES,pooled)},
        pooled_revelation_rank=rank_metrics(poolhist[3,0],poolhist[3,1]+poolhist[3,2]),
        zero_new_observation={rule:dict(exact_empty=int(v[0]),false_cells=int(v[1])) for rule,v in zip(RULES,zero_counts)},
        protocol_sha256=digest(OUT/'protocol.json'),precision_contract_sha256=digest(OUT/'precision_contract.json'))
    iou=np.array([result['rules'][r]['label_aware_delta_iou'] for r in RULES[1:]])
    f1=np.array([result['rules'][r]['revelation_f1'] for r in RULES[1:]])
    selected=[int(iou.argmax())+1,int(f1.argmax())+1]
    result['development_selected']=dict(graphdelta_iou=RULES[selected[0]],revelation_f1=RULES[selected[1]],
        caveat='selected on these same development outcomes; bootstrap CI conditional/exploratory, not independent validation')
    result['paired_scene_ci95']=bootstrap(arrays['cm'],tuple(sorted(set([0]+selected))))
    protocol=json.loads((OUT/'protocol.json').read_text());selection=set(protocol['threshold_selection_scenes']);validation=set(protocol['diagnostic_validation_scenes'])
    selection_indices=[j for j,s in enumerate(names) if s in selection];validation_indices=[j for j,s in enumerate(names) if s in validation]
    if selection_indices and validation_indices:
        sc=arrays['cm'][selection_indices].mean(0)
        siou=np.array([extended_rates(cm)['label_aware_delta_iou'] for cm in sc[1:]])
        sf1=np.array([extended_rates(cm)['revelation_f1'] for cm in sc[1:]])
        heldselected=[int(siou.argmax())+1,int(sf1.argmax())+1]
        valavg={k:v[validation_indices].mean(0) for k,v in arrays.items()}
        vm=metrics(valavg['cm'],valavg['scores'],valavg['hist'],valavg['cal'],valavg['obs'],valavg['obsnorm'])
        fitted_prior=arrays['hist'][selection_indices,3].mean(0).sum(1)
        fitted_prior/=fitted_prior.sum();validation_prior=valavg['hist'][3].sum(1);validation_prior/=validation_prior.sum()
        vm['fitted_constant_probability_baseline']=dict(probabilities=fitted_prior.tolist(),
            multiclass_logloss=float(-np.sum(validation_prior*np.log(np.clip(fitted_prior,1e-9,1)))),
            multiclass_brier=float(1+np.sum(fitted_prior**2)-2*np.dot(fitted_prior,validation_prior)),
            binary_revelation_logloss=float(-validation_prior[0]*np.log(fitted_prior[0])-(1-validation_prior[0])*np.log(1-fitted_prior[0])),
            binary_revelation_brier=float((1-validation_prior[0])*fitted_prior[0]**2+validation_prior[0]*(1-fitted_prior[0])**2),
            auroc=.5,auprc=float(1-validation_prior[0]),fit='70selectionscenes only; no inputs, no neural training')
        result['diagnostic_scene_validation']=dict(selection_scenes=len(selection_indices),validation_scenes=len(validation_indices),
            selected_rules=dict(graphdelta_iou=RULES[heldselected[0]],revelation_f1=RULES[heldselected[1]]),
            metrics=vm,paired_scene_ci95=bootstrap(arrays['cm'][validation_indices],tuple(sorted(set([0]+heldselected)))),
            caveat=protocol['split_caveat'])
    result['cohorts']={group:{rule:extended_rates(cm) for rule,cm in zip(RULES,np.mean([np.mean(rs,axis=0) for rs in scene_roots.values()],axis=0))}
        for group,scene_roots in cohort_scenes.items()}
    if not args.partial:
        original=json.loads((EVALUATION/'metrics.json').read_text())['candidate']
        np.testing.assert_array_equal(pooled[0],original['pooled_confusion'])
        for key in ('label_aware_delta_iou','revelation_iou'):
            np.testing.assert_allclose(result['rules']['argmax'][key],original['scene_balanced'][key],atol=1e-14,rtol=0)
        if total!=47128 or len(rootrecords)!=2110:raise ValueError('incomplete frozen evaluation')
    destination=OUT/'probability'/('summary_partial.json' if args.partial else 'summary.json')
    save(destination,result)
    np.savez_compressed(OUT/'probability'/('scene_arrays_partial.npz' if args.partial else 'scene_arrays.npz'),
        scene_names=np.array(names),confusions=arrays['cm'],histograms=arrays['hist'],scores=arrays['scores'],calibration=arrays['cal'])
    print(json.dumps({'roots':len(rootrecords),'segments':total,'selected':result['development_selected'],
        'probability':result['probability'],'seconds':time.perf_counter()-start}),flush=True)

if __name__=='__main__':main()
