"""Independent cached D count/order review. No neural model or production import."""
from __future__ import annotations
from collections import defaultdict
import gzip,hashlib,json
from pathlib import Path
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/stage2_diagnostic_20261025';DEST=OUT/'trajectory/decision_independent_review_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)
def dump(p,v):
 with open(p,'x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def stats(x,y,w):
 w=w/w.sum();mx=np.sum(w[:,None]*x,axis=0);my=np.sum(w[:,None]*y,axis=0)
 dx=x-mx;dy=y-my;vx=np.sum(w[:,None]*dx*dx,axis=0);vy=np.sum(w[:,None]*dy*dy,axis=0)
 cor=np.sum(w[:,None]*dx*dy,axis=0)/np.sqrt(np.maximum(vx*vy,1e-60))
 return np.array([np.sum(w[:,None]*abs(x-y),axis=0),np.sum(w[:,None]*(x-y),axis=0),mx,my,cor])
def main():
 DEST.mkdir(exist_ok=False);start=__import__('time').perf_counter()
 protocol=read(OUT/'protocol.json');selection=set(protocol['threshold_selection_scenes']);validation=set(protocol['diagnostic_validation_scenes'])
 reported=read(OUT/'decision_relevance/counts.json');paired=read(OUT/'decision_relevance/pairwise_counts.json');pairrecords=read(OUT/'decision_relevance/pairwise_count_records.json')
 soft=read(OUT/'visibility/soft_paired_v1/branch_expected_counts.json');soft={r['root_id']:r for r in soft['records']}
 cs=read(OUT/'visibility/full_v1/summary.json');category_rates=np.array(cs['calibration']['category_rates'])
 roots=[];category_proofs=0;legal_root_checks=[];cmass=defaultdict(list);cwindows={}
 for p in sorted((OUT/'probability/roots').glob('*.json.gz')):
  a=read(p);rid=a['root_id'];c=read(OUT/'visibility/full_v1/roots'/(rid+'.json.gz'))
  assert a['root_id']==c['root_id'] and a['scene_id']==c['scene_id']
  amap={(r['start_node'],r['end_node']):r for r in a['records']};cw={}
  for r in c['records']:
   key=(r['start_node'],r['end_node']);ar=amap[key];counts=np.asarray(r['category_label_counts'],np.int64);strict,frustum=r['raw_opportunity_counts'];n=r['candidates']
   # Category cardinalities are legal-input geometry only. Target columns sum out.
   np.testing.assert_array_equal(counts.sum(1),[n-frustum,frustum-strict,strict])
   np.testing.assert_array_equal(counts.sum(0),np.asarray(ar['confusion'][0]).sum(1))
   assert counts.sum()==n==ar['candidates'] and r['domain_sha256']==ar['domain_sha256']
   assert r['true_count']==ar['true_counts'][0]
   category_proofs+=1;cw[key]=r
  assert set(cw)==set(amap)
  cmass[a['scene_id']].append(np.mean([np.asarray(r['category_label_counts'])/r['candidates'] for r in c['records']],axis=0))
  # Compact A data and a minimal paired lookup only; no probability arrays loaded.
  compact=[]
  for r in a['records']:
   compact.append(dict(y=r['true_counts'],x=r['expected_counts'],n=r['candidates'],band=next(g for g in r['groups']if g.startswith('duration:')),start=r['start_node'],end=r['end_node']))
  roots.append(dict(root_id=rid,scene=a['scene_id'],records=compact));cwindows[rid]=cw
  legal_root_checks.append((rid,sha(p),sha(OUT/'visibility/full_v1/roots'/(rid+'.json.gz'))))
 assert len(roots)==2110 and category_proofs==47128
 mass=np.mean([np.mean(cmass[s],axis=0) for s in sorted(selection)],axis=0);rr=mass[:,1:].sum(1)/mass.sum(1)
 np.testing.assert_allclose(rr,category_rates,atol=1e-14,rtol=0)
 scene_roots=defaultdict(list)
 for r in roots:scene_roots[r['scene']].append(r)
 train=[];trainw=[]
 for s in sorted(selection):
  for r in scene_roots[s]:
   for v in r['records']:train.append(v);trainw.append(1/(len(selection)*len(scene_roots[s])*len(r['records'])))
 trainw=np.asarray(trainw);baseline={};maxdiff=0.
 for group in ('all','duration:<=0.5','duration:0.5-1','duration:1-2','duration:>2'):
  idx=np.array([i for i,r in enumerate(train)if group=='all' or r['band']==group]);w=trainw[idx];ys=np.asarray([train[i]['y']for i in idx]);ns=np.asarray([train[i]['n']for i in idx]);mean=np.average(ys,weights=w,axis=0)
  median=[]
  for j in range(3):
   order=np.argsort(ys[:,j]);median.append(float(ys[order[np.searchsorted(w[order].cumsum(),.5*w.sum())],j]))
  area=np.sum(w[:,None]*ys,axis=0)/np.sum(w*ns);baseline[group]=dict(mean=mean,median=np.asarray(median),area=area)
  for v,k in [(mean,'mean_counts'),(median,'median_counts'),(area,'area_prevalence')]:np.testing.assert_allclose(v,reported['fitted_baselines'][group][k],atol=2e-10,rtol=0)
 names=['model','zero','global_mean','duration_mean','duration_median','duration_candidate_area'];outcomes=['revelation','free','blocked'];scenes=sorted(validation)
 sx=[];sy=[];sw=[];scene_m=[];valwindows=0
 for s in scenes:
  rms=[]
  for r in scene_roots[s]:
   ms=[]
   for v in r['records']:
    y=np.asarray(v['y'],float);b=baseline[v['band']];xs=np.asarray([v['x'],np.zeros(3),baseline['all']['mean'],b['mean'],b['median'],b['area']*v['n']])
    ms.append(np.stack([abs(xs-y),xs-y,xs,np.broadcast_to(y,xs.shape),xs**2,np.broadcast_to(y*y,xs.shape),xs*y],axis=1))
    sx.append(xs);sy.append(y);sw.append(1/(38*len(scene_roots[s])*len(r['records'])));valwindows+=1
   rms.append(np.mean(ms,axis=0))
  scene_m.append(np.mean(rms,axis=0))
 sx=np.asarray(sx);sy=np.asarray(sy);sw=np.asarray(sw);scene_m=np.asarray(scene_m)
 for j,name in enumerate(names):
  m=stats(sx[:,j],sy,sw)
  for k,out in enumerate(outcomes):
   for i,field in enumerate(('mae','bias','mean_prediction','mean_truth','pearson')):
    expected=reported['metrics'][name][out][field]
    if expected is None:continue
    d=abs(m[i,k]-expected);assert d<2e-9,(name,out,field,d);maxdiff=max(maxdiff,d)
 # Independent resampling of scene sufficient statistics; evaluate the same population Pearson.
 rng=np.random.default_rng(2525);indices=rng.integers(0,38,(2000,38));boot=scene_m[indices].mean(1)
 mae,bias,px,py,xx,yy,xy=np.moveaxis(boot,2,0);vx=np.maximum(xx-px*px,0);vy=np.maximum(yy-py*py,0);valid=(vx>1e-12*np.maximum(xx,1))&(vy>1e-12*np.maximum(yy,1));corr=np.divide(xy-px*py,np.sqrt(vx*vy),out=np.zeros_like(px),where=valid)
 maxci=0.
 for j,name in enumerate(names):
  for k,out in enumerate(outcomes):
   for arr,field in [(mae,'mae'),(bias,'bias'),(corr,'pearson')]:
    if field=='pearson' and not valid[:,j,k].any():continue
    actual=np.quantile(arr[:,j,k]if field!='pearson' else arr[valid[:,j,k],j,k],[.025,.975]);expected=reported['scene_ci95'][f'{name}/{out}/{field}']
    d=np.max(abs(actual-expected));assert d<2e-9;maxci=max(maxci,float(d))
    key=f'{name}/{out}/{field}/model_minus_baseline'
    if key in reported['scene_ci95']:
     good=(valid[:,j,k]&valid[:,0,k])if field=='pearson'else np.ones(2000,bool)
     actual=np.quantile(arr[good,0,k]-arr[good,j,k],[.025,.975]);d=np.max(abs(actual-reported['scene_ci95'][key]));assert d<2e-9;maxci=max(maxci,float(d))
 rootlookup={r['root_id']:r for r in roots};pair_branch_checks=0
 for p in pairrecords:
  rid=p['root_id'];b=read(OUT/'trajectory/primary_v1/roots'/(rid+'.json'));s=soft[rid];a=rootlookup[rid];lookup={(r['start'],r['end']):r for r in a['records']}
  assert s['scene_id']==p['scene_id']==a['scene'] and p['start_node']==b['pair']['start_node']
  expected_true=np.asarray([lookup[(p['start_node'],p[end])]['y'] for end in ('end_a','end_b')]);np.testing.assert_array_equal(expected_true,p['true_counts']);np.testing.assert_array_equal(expected_true[:,0],s['true_counts'])
  np.testing.assert_array_equal(expected_true,[b['true_counts_a'],b['true_counts_b']])
  base=p['baseline_revelation_counts'];np.testing.assert_allclose(base['soft_geometry_fixed_union'],s['geometry_expected_counts'],atol=0,rtol=0)
  np.testing.assert_allclose(base['stage2_fixed_union'],s['stage2_direct_expected_counts'],atol=2e-10,rtol=0)
  np.testing.assert_allclose(base['stage2'],np.asarray(p['expected_counts'])[:,0],atol=0,rtol=0)
  np.testing.assert_allclose(base['stage2'],s['stage2_production_expected_counts'],atol=2e-10,rtol=0)
  cats=[cwindows[rid][(p['start_node'],p[end])]for end in ('end_a','end_b')]
  legal_sizes=[np.asarray([r['candidates']-r['raw_opportunity_counts'][1],r['raw_opportunity_counts'][1]-r['raw_opportunity_counts'][0],r['raw_opportunity_counts'][0]])for r in cats]
  np.testing.assert_allclose(base['calibrated_geometry'],np.asarray(legal_sizes)@category_rates,atol=1e-12,rtol=0)
  pair_branch_checks+=1
 pairdiff=0.
 for subset,records in [('all',pairrecords),('diagnostic_validation38',[r for r in pairrecords if r['scene_id']in validation])]:
  for out,j in zip(outcomes,range(3)):
   for margin in (0,10,50):
    eligible=[r for r in records if abs(r['true_counts'][0][j]-r['true_counts'][1][j])>margin]
    if not eligible:continue
    reported_pair=paired['subsets'][subset][out][f'actual_difference_gt_{margin}'];assert len(eligible)==reported_pair['pairs']
    assert sum(r['true_counts'][0][j]==r['true_counts'][1][j]for r in records)==reported_pair['exact_actual_ties_excluded']
    nms=list(eligible[0]['baseline_revelation_counts'])if j==0 else ['stage2'];groups=defaultdict(list)
    for r in eligible:
     truth=r['true_counts'][0][j]-r['true_counts'][1][j];row=[]
     for name in nms:
      a,b=r['baseline_revelation_counts'][name]if j==0 else np.asarray(r['expected_counts'])[:,j];d=a-b
      row.append(.5 if abs(d)<=1e-9 else float((d>0)==(truth>0)))
     groups[r['scene_id']].append(row)
    arr=np.array([np.mean(v,axis=0)for s,v in sorted(groups.items())]);assert len(arr)==reported_pair['scenes'];means=arr.mean(0);draw=np.random.default_rng(2525).integers(0,len(arr),(2000,len(arr)));bs=arr[draw].mean(1)
    for k,name in enumerate(nms):
     rr=reported_pair['accuracy'][name]
     for actual,expect in [(means[k],rr['value']),(np.quantile(bs[:,k],[.025,.975]),rr['ci95']),(np.quantile(bs[:,0]-bs[:,k],[.025,.975]),rr['stage2_minus_baseline_ci95'])]:
      d=float(np.max(abs(np.asarray(actual)-expect)));assert d<1e-12;pairdiff=max(pairdiff,d)
 identity={'review_source_sha256':sha(__file__),'counts_sha256':sha(OUT/'decision_relevance/counts.json'),'pairwise_counts_sha256':sha(OUT/'decision_relevance/pairwise_counts.json'),'pair_records_sha256':sha(OUT/'decision_relevance/pairwise_count_records.json'),'soft_branch_counts_sha256':sha(OUT/'visibility/soft_paired_v1/branch_expected_counts.json'),'C_summary_sha256':sha(OUT/'visibility/full_v1/summary.json'),'protocol_sha256':sha(OUT/'protocol.json')}
 result=dict(status='PASS_WITH_DECLARED_SCOPE',identity=identity,roots=len(roots),C_windows_proved_target_partition_invariant=category_proofs,paired_branch_alignment_checks=pair_branch_checks,validation_roots=sum(len(scene_roots[s])for s in validation),validation_scenes=38,validation_windows=valwindows,fit_scenes=70,category_calibration_reproduced=True,independent_count_point_max_absolute_error=maxdiff,independent_count_ci95_max_absolute_error=maxci,independent_pairwise_point_ci95_max_absolute_error=pairdiff,all_actual_ties_margins_predicted_half_ties_weights_verified=True,soft_branch_targets_and_domains_matched=True,no_target_leak_from_C_category_row_sums='Verified all records: sum_j category_label_counts[k,j] == [N-frustum,frustum-strict,strict][k]. This erases all target labels and equals geometry-only cardinalities; rates fitted only on70scenes.',elapsed_seconds=time.perf_counter()-start,limitations=['Expected observation counts/order are descriptive frozen-model diagnostics, not a Query-ranking model or model-selection target.','Margin>10 and>50 strata condition on outcome difference, not an online filtering rule; performance there is conditional.','38 scenes were historically scored; not untouched confirmation.','Conditional scene bootstrap holds70scene baseline/calibration fits fixed.','Counts across correlated/overlapping windows are averaged with scene/Root weights; never summed as independent information gain.','Count correlation and conditional ordering do not establish reliable occupancy decisions or downstream utility.'])
 dump(DEST/'proof.json',result);dump(DEST/'all_root_compact_receipts.json',legal_root_checks)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
