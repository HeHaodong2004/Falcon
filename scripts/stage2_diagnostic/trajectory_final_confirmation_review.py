"""Independent final confirmation review of metadata and DERIVED outputs only.

Never opens protected legal/privileged payloads or imports model/data modules.
"""
from collections import defaultdict
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/stage2_diagnostic_20261025';BASE=OUT/'final_confirmation';META=OUT/'final_confirmation_metadata';DEST=BASE/'independent_review'
OLD=Path('/home/haodong/workspace/projects/eventwm-falcon-l2-partial-20261021/output/l2_graph_wm_120k_20261021/snapshot/manifest.json');FIT=Path('/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023/output/l2_two_stage_20261023/final_collection_20261024/manifest.json')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ratio(a,b):return float(a/b)if b else 0.
def cmrates(c):
 result={};c=np.asarray(c,float)
 for i,k in enumerate(('unknown','free','blocked')):result[k]=dict(precision=ratio(c[i,i],c[:,i].sum()),recall=ratio(c[i,i],c[i].sum()),f1=ratio(2*c[i,i],c[i].sum()+c[:,i].sum()))
 actual=c[1:,:].sum();pred=c[:,1:].sum();exact=c[1,1]+c[2,2];both=c[1:,1:].sum()
 result.update(label_aware_delta_iou=ratio(exact,actual+pred-exact),revelation_iou=ratio(both,actual+pred-both),revelation_precision=ratio(both,pred),revelation_recall=ratio(both,actual),revelation_f1=ratio(2*both,actual+pred),terminal_label_agreement_on_true_revelations=ratio(exact,actual),missed_revelations=float(c[1:,0].sum()),incorrect_free_blocked=float(c[1,2]+c[2,1]),false_revelations=float(c[0,1:].sum()),false_revelation_rate=ratio(c[0,1:].sum(),c[0].sum()),wrong_free_blocked_rate=ratio(c[1,2]+c[2,1],actual),false_free_unknown_count=float(c[0,1]),false_free_blocked_count=float(c[2,1]),false_free_unknown_fraction=ratio(c[0,1],c[:,1].sum()),false_free_blocked_fraction=ratio(c[2,1],c[:,1].sum()),false_free_on_actual_blocked_rate=ratio(c[2,1],c[2].sum()))
 return result
def check(actual,expected,tol=1e-10):
 ds=[]
 for k,v in actual.items():
  if isinstance(v,dict):ds+=check(v,expected[k],tol)
  else:
   d=abs(v-expected[k]);assert d<=tol,(k,d);ds.append(d)
 return ds
def moments(x,y):return np.stack([abs(x-y),x-y,x,y,x*x,y*y,x*y])
def count_rates(m):
 a,b,x,y,xx,yy,xy=m;vx=np.maximum(xx-x*x,0);vy=np.maximum(yy-y*y,0);valid=(vx>1e-12*np.maximum(xx,1))&(vy>1e-12*np.maximum(yy,1));corr=np.divide(xy-x*y,np.sqrt(vx*vy),out=np.zeros(3),where=valid)
 return {k:dict(mae=float(a[j]),bias=float(b[j]),mean_prediction=float(x[j]),mean_truth=float(y[j]),pearson=float(corr[j])if valid[j]else None)for j,k in enumerate(('revelation','free','blocked'))}
def main():
 DEST.mkdir(exist_ok=False);config=read(BASE/'protocol.json');summary=read(BASE/'summary.json');complete=read(BASE/'complete.json');selected=read(META/'selected_manifest.json')['rows'];exposure=read(BASE/'exposure_receipt.json')
 protocol_sha=sha(BASE/'protocol.json');assert protocol_sha==complete['identity']['protocol_sha256']==summary['protocol_sha256']==exposure['protocol_sha256']
 assert sha(META/'selected_manifest.json')==config['selected_manifest_sha256']=='f29a343e745005710a3a9e3e5a8fd4f7524b88e7fb6cce4065bac2f247691b46'
 assert sha(META/'receipt.json')==config['metadata_receipt_sha256']
 assert config['current_source_base_commit']=='82eb2dabb3401528c262d1de384b83b4098808aa'and config['threshold']==.2 and config['policies']==['released_argmax','frozen_revelation_0.20']
 for p,h in config['source'].items():assert sha(ROOT/p)==h
 for p,h in config['frozen_development_sources'].items():assert sha(OUT/p)==h
 checkpoint=ROOT/'output/stage2_segment_20261024/training/final/last.pt';assert sha(checkpoint)==config['checkpoint_sha256']==summary['checkpoint_sha256']
 a=read(OUT/'probability/summary.json');d=read(OUT/'decision_relevance/counts.json');g=read(OUT/'visibility/soft_paired_v1/summary.json')
 assert config['prior']==a['diagnostic_scene_validation']['metrics']['fitted_constant_probability_baseline']['probabilities'];assert config['duration_baselines']==d['fitted_baselines'];assert config['geometry_rates']==g['rates']
 old=read(OLD);fit=read(FIT);fscenes={r['scene_id']for r in fit['rows']};dev={r['scene_id']for r in old['rows']if r['split']=='dev'};opened={r['scene_id']for r in old['protected']if r['exclusion']=='previously_opened_fresh_scene'};protected={r['scene_id']for r in old['protected']if r['exclusion']=='protected_research_holdout'and r['source_split']=='train'}
 scenes={r['scene_id']for r in selected};assert len(selected)==222 and len(scenes)==111 and scenes<=protected and scenes.isdisjoint(fscenes|dev|opened)
 assert all(int(hashlib.sha256(('l2-explore-20261021:'+s).encode()).hexdigest()[:8],16)%100>=85 for s in scenes)
 assert set(exposure['roots'])=={r['root_id']for r in selected};lookup={r['root_id']:r for r in selected};grouped=defaultdict(list);receipt=[];zero=np.zeros((2,2),int);ood=0;known=0;audits=defaultdict(int);geometry_hist=np.zeros((33,2));h=None;graphs=0;prior_max=0.;cm_max=0.
 for p in sorted((BASE/'roots').glob('*.json')):
  r=read(p);rid=r['root_id'];meta=lookup[rid];assert r['scene_id']==meta['scene_id']and r['identity']==complete['identity']and r['member_sha256']==meta['files'];assert len(r['records'])==r['audit']['candidate_segments'];assert r['audit']['queries']==meta['queries']
  seen=set();count=[];prior=np.asarray(config['prior'])
  for v in r['records']:
   key=(v['start_node'],v['end_node']);assert key not in seen;seen.add(key);assert len(v['confusions'])==2and len(v['scores'])==4
   mats=np.asarray(v['confusions']);n=v['candidates']+v['out_of_domain'];true=np.asarray(v['true_counts']);truth=np.array([n-true[0],true[1],true[2]])
   np.testing.assert_array_equal(mats.sum(2),np.broadcast_to(truth,(2,3)));assert np.all(mats>=0)and mats.sum()==2*n
   assert all(v['full_future_graphs_verified']);graphs+=2
   if not true[0]:
    counts=mats[:,:,1:].sum((1,2));zero[:,0]+=(counts==0);zero[:,1]+=counts
   ood+=v['out_of_domain'];known+=v['known_terminal_differences']
   prior_ce=-np.dot(truth,np.log(prior))/n;prior_br=1+np.square(prior).sum()-2*np.dot(truth/n,prior)
   error=max(abs(prior_ce-v['baseline_scores'][0]),abs(prior_br-v['baseline_scores'][1]));assert error<1e-6;prior_max=max(prior_max,error)
   duration=v['duration'];band='duration:'+('<=0.5'if duration<=.5000001 else'0.5-1'if duration<=1.0000001 else'1-2'if duration<=2.0000001 else'>2');median=np.asarray(config['duration_baselines'][band]['median_counts'])
   count.append([moments(np.asarray(v['expected_counts']),true),moments(median,true)])
  np.testing.assert_allclose(np.mean(count,axis=0),np.asarray(r['count_moments'])[:2],atol=2e-8,rtol=0)
  hist=p.with_suffix('.npz');assert sha(hist)==r['hist_sha256']
  with np.load(hist,allow_pickle=False)as z:
   rh=z['revelation_histogram'];assert np.isfinite(rh).all()and np.all(rh>=0)and abs(rh.sum()-1)<1e-12
   if h is None:h=np.zeros_like(rh)
   h+=rh/222;geometry_hist+=z['geometry_histogram']/222;assert z['geometry_certain_unknown_ood_positive_mass']==0
  for k,v in r['audit'].items():audits[k]+=v
  grouped[r['scene_id']].append(r);receipt.append(dict(root_id=rid,scene_id=r['scene_id'],record_sha256=sha(p),hist_sha256=r['hist_sha256']))
 assert set(grouped)==scenes and all(len(v)==2for v in grouped.values())and len(receipt)==222and ood==0
 names=sorted(grouped);cms=np.asarray([np.mean([np.mean([v['confusions']for v in r['records']],axis=0)for r in grouped[s]],axis=0)for s in names]);scores=np.asarray([np.mean([np.mean([v['scores']for v in r['records']],axis=0)for r in grouped[s]],axis=0)for s in names]);baselines=np.asarray([np.mean([np.mean([v['baseline_scores']for v in r['records']],axis=0)for r in grouped[s]],axis=0)for s in names]);mom=np.asarray([np.mean([r['count_moments']for r in grouped[s]],axis=0)for s in names])
 differences=[]
 for policy,matrix in zip(config['policies'],cms.mean(0)):differences+=check(cmrates(matrix),summary['policies'][policy])
 for k,v in zip(('multiclass_logloss','multiclass_brier','binary_logloss','binary_brier'),scores.mean(0)):differences+=check({k:float(v)},summary['probability'])
 for k,v in zip(('constant_prior_multiclass_logloss','constant_prior_multiclass_brier','geometry_binary_logloss','geometry_binary_brier'),baselines.mean(0)):differences+=check({k:float(v)},summary['frozen_baselines'])
 for name,matrix in zip(('model','frozen_dev_duration_median','geometry_R_only'),mom.mean(0)):
  actual=count_rates(matrix)
  if name=='geometry_R_only':actual={'revelation':actual['revelation']}
  differences+=check(actual,summary['counts'][name])
 rate=np.asarray(config['geometry_rates']);geo_ll=-np.sum(geometry_hist[:,0]*np.log(1-rate)+geometry_hist[:,1]*np.log(rate));geo_br=np.sum(geometry_hist[:,0]*rate**2+geometry_hist[:,1]*(1-rate)**2)
 np.testing.assert_allclose([geo_ll,geo_br],[summary['frozen_baselines']['geometry_binary_logloss'],summary['frozen_baselines']['geometry_binary_brier']],atol=1e-14,rtol=0)
 # Independent rank equations from equal-scene/Root cell-normalized histogram.
 neg=h[0];pos=h[1:].sum(0);N=neg.sum();P=pos.sum();auc=np.dot(pos,np.cumsum(neg)-.5*neg)/(N*P);tp=np.cumsum(pos[::-1]);fp=np.cumsum(neg[::-1]);ap=np.sum(pos[::-1]/P*tp/np.maximum(tp+fp,1e-30))
 differences+=check({'prevalence':float(P/(N+P)),'auroc':float(auc),'auprc_ap':float(ap)},summary['probability']['revelation_rank'])
 rng=np.random.default_rng(2525);draw=defaultdict(list)
 for _ in range(2000):
  i=rng.integers(0,111,111);rr=[cmrates(m)for m in cms[i].mean(0)]
  for k in ('label_aware_delta_iou','revelation_f1','terminal_label_agreement_on_true_revelations'):draw[k+'_difference'].append(rr[1][k]-rr[0][k])
  draw['model_minus_geometry_binary_logloss'].append(scores[i,2].mean()-baselines[i,2].mean());m=mom[i].mean(0);draw['model_count_mae'].append(m[0,0,0]);draw['model_minus_duration_median_mae'].append(m[0,0,0]-m[1,0,0])
 cierror=0.
 for k,values in draw.items():
  ci=np.quantile(values,[.025,.975]);d=float(np.max(abs(ci-summary['scene_ci95'][k])));assert d<1e-12;cierror=max(cierror,d)
 for j,k in enumerate(config['policies']):assert {'exact_empty':int(zero[j,0]),'false_cells':int(zero[j,1])}==summary['zero_new'][k]
 assert dict(audits)==summary['source_audit']and known==summary['overlapping_window_known_terminal_differences']and graphs==2*summary['segments']==9134and summary['segments']==4567
 # Ordinary loader guards and all preexisting frozen files match the dev commit.
 git_matches=[]
 for rel in config['source']:
  if rel=='scripts/stage2_diagnostic/final_confirmation.py':continue
  committed=subprocess.check_output(['git','show','82eb2dabb3401528c262d1de384b83b4098808aa:'+rel],cwd=ROOT)
  assert hashlib.sha256(committed).hexdigest()==config['source'][rel];git_matches.append(rel)
 result=dict(status='PASS_WITH_DECLARED_SCOPE',review_source_sha256=sha(__file__),protocol_sha256=protocol_sha,final_source_sha256=config['source']['scripts/stage2_diagnostic/final_confirmation.py'],selected_manifest_sha256=sha(META/'selected_manifest.json'),summary_sha256=sha(BASE/'summary.json'),checkpoint_sha256=sha(checkpoint),roots=222,scenes=111,queries=1429,segments=4567,full_future_graph_receipts=graphs,exact_cohort_membership=True,protected_scene_fit_dev_opened_isolation=True,all_recorded_member_hash_identities_match_selection=True,all_frozen_development_parameters_and_artifact_hashes_exact=True,preexisting_source_files_matching_frozen_development_commit=len(git_matches),all_source_hashes_match_frozen_protocol=True,no_test_configuration_selection_in_source=True,maximum_aggregate_metric_difference=max(differences),maximum_ci95_difference=cierror,maximum_prior_recomputation_difference=prior_max,frozen_geometry_proper_scores_independently_recomputed_from_bin_counts=True,all_zero_new_counts_exact=True,all_source_audit_counts_exact=True,out_of_domain_positive=ood,review_read_scope='Selected metadata, historical boundary metadata, frozen source/checkpoint hashes, derived records/histograms only; no raw legal/privileged payload reopened and no model execution.',limitations=['Sampled2Rootsperscene; confirms this bounded historical-planner/source distribution, not all protected data or cross-planner transfer.','No dense protected p cache was created; per-cell threshold decisions cannot be independently recomputed without new inference. Review verifies immutable source rule, identities, derived confusions, and full-graph check receipts.','Full future graph checks are executed in the frozen harness and recorded; this review does not independently reconstruct raw protected terminal graphs.','Proper-score and count baselines remain development-fitted, scene CIs conditional on fixed fits.','Higher threshold improves overlap but hard FREE precision remains limited; no automatic production/policy promotion.','Selected222Root legal payloads are now final-evaluation-exposed. All remaining protected membership remains unchanged; historical statements of sealed status describe earlier phases.'])
 with (DEST/'proof.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 with (DEST/'root_receipts.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
