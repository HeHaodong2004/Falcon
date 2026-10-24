"""Reproducible scene-cluster summaries and standalone diagnostic figures."""
import argparse,collections,json,pathlib
import numpy as np
from .evaluate import scene_ci

def root_mean(records,key):
    groups=collections.defaultdict(list)
    for r in records:groups[r['root_id']].append(r)
    return [{'root_id':k,'scene_id':rs[0]['scene_id'],key:float(np.mean([r[key] for r in rs]))} for k,rs in groups.items()]

def metric(records,key):return scene_ci(root_mean(records,key),key)

def summarize(data,dev):
    raw=data['queries'];queries=[r for r in raw if r.get('risk_evaluable',True)]
    risks={}
    for name,pred,baseline in [('collision','pred_collision','cv_collision'),('psc','pred_psc','cv_psc'),('appearance','pred_appearance',None)]:
        if not queries or pred not in queries[0]:continue
        prior_rows=[r for r in dev['queries'] if r.get('risk_evaluable',True)]
        prevalence=metric(prior_rows,name)['mean']
        records=[{**r,'brier':(r[pred]-r[name])**2,'gap':r[pred]-r[name],
                  'constant_brier':(prevalence-r[name])**2,
                  'baseline_brier':(r[baseline]-r[name])**2 if baseline else (prevalence-r[name])**2} for r in queries]
        bins=[]
        for lo,hi in zip(np.arange(0,1,.1),np.arange(.1,1.1,.1)):
            subset=[r for r in records if lo<=r[pred] and (r[pred]<hi or hi>.99)]
            if subset:bins.append({'low':float(lo),'high':float(hi),'queries':len(subset),'predicted':metric(subset,pred),'observed':metric(subset,name)})
        groups=collections.defaultdict(list)
        for r in records:groups[r['root_id']].append(r)
        ranking=[]
        for root,rs in groups.items():
            pos=[r for r in rs if r[name]];neg=[r for r in rs if not r[name]]
            if pos and neg:
                score=[float(a[pred]>b[pred])+.5*float(a[pred]==b[pred]) for a in pos for b in neg]
                ranking.append({'root_id':root,'scene_id':rs[0]['scene_id'],'concordance':float(np.mean(score))})
        risks[name]={'brier':metric(records,'brier'),'constant_dev_prevalence_brier':metric(records,'constant_brier'),
                     'baseline_brier':metric(records,'baseline_brier'),'observed':metric(records,name),'predicted':metric(records,pred),
                     'prediction_minus_observed':metric(records,'gap'),'within_root_ranking':scene_ci(ranking,'concordance'),
                     'calibration':bins,'queries':len(records),'fixed_dev_prevalence':prevalence}
    groups=collections.defaultdict(list)
    for r in raw:groups[r['root_id']].append(r)
    choices=[]
    for root,rs in groups.items():
        if len(rs)<2 or not all(r.get('risk_evaluable',True) for r in rs):continue
        best=max(r['actual_reward'] for r in rs);row={'root_id':root,'scene_id':rs[0]['scene_id'],'oracle':best,'candidates':len(rs)}
        for model,key in [('structured','pred_reward'),('memory_cv','cv_reward'),('depth0','depth0_reward')]:
            chosen=max(rs,key=lambda r:(r[key],r['query_id']))
            row[model]=chosen['actual_reward'];row[model+'_regret']=best-chosen['actual_reward'];row[model+'_collision']=chosen['collision'];row[model+'_psc']=chosen['psc']
        row['structured_minus_depth0']=row['structured']-row['depth0'];row['structured_minus_cv']=row['structured']-row['memory_cv'];choices.append(row)
    decision={key:scene_ci(choices,key) for key in ('oracle','structured','memory_cv','depth0','structured_regret','memory_cv_regret','depth0_regret','structured_minus_depth0','structured_minus_cv','structured_collision','depth0_collision','structured_psc','depth0_psc')}
    regression=[]
    for r in queries:regression.append({**r,'duration_mae':abs(r['pred_duration']-r['duration']),'progress_mae':abs(r['pred_progress']-r['progress']),'blockage_brier':(r['blockage']-r['actual_blockage'])**2})
    forecast={}
    for name in data['models']:
        forecast[name]={}
        for horizon in (.5,1,2,3):
            rs=[r for r in data['forecasts'] if r['model']==name and r['horizon']==horizon]
            forecast[name][str(horizon)]={k:scene_ci(rs,k) for k in ('energy_score','mean_error','coverage90','radius90','unresolved_mass','conditional_scored_mass')}
    paired={}
    full={(r['root_id'],r['horizon']):r for r in data['forecasts'] if r['model']=='structured_learned'}
    for name in ('visible_cv','memory_cv','structured_kinematic','learned_graph'):
        paired[name]={}
        for h in (.5,1,2,3):
            rs=[]
            for r in data['forecasts']:
                key=(r['root_id'],h)
                if r['model']==name and r['horizon']==h and key in full:rs.append({**r,'delta':full[key]['energy_score']-r['energy_score']})
            paired[name][str(h)]=scene_ci(rs,'delta')
    common_ids={r['root_id'] for r in data['forecasts'] if r['model']=='structured_learned' and r['horizon']==3}
    common_cohort={name:{str(h):scene_ci([r for r in data['forecasts'] if r['model']==name and r['horizon']==h and r['root_id'] in common_ids],'energy_score') for h in (.5,1,2,3)} for name in data['models']}
    return {'split':data['split'],'roots':data['roots'],'forecast':forecast,'forecast_paired_deltas':paired,'same_3s_cohort':common_cohort,
            'conditioning':data['observation_conditioning'],'risk':risks,'decision':decision,'decision_rows':choices,
            'regression':{k:metric(regression,k) for k in ('duration_mae','progress_mae','blockage_brier')},
            'query_support':{'all':len(raw),'eligible':len(queries),'excluded_administrative_censoring':len(raw)-len(queries)},
            'unresolved_query_quantiles':dict(zip(('p50','p90','p95','p99','max'),map(float,np.percentile([r['unresolved'] for r in raw],[50,90,95,99,100])))),
            'consistency_max':{k:max((r[k] for r in data['consistency']),default=0) for k in ('branch_probability_error','count_marginal_error_retained')},
            'estimand':'equal-weight scenes of equal-weight independent roots; sibling query metrics averaged within root',
            'forecast_scope':'offline-aligned originally-visible targets, conditional retained mass; not a random-finite-set or full-belief score',
            'event_calibration_caveat':'frozen event-head intercept was calibrated at observed stopping times; heldout evaluation uses corrected full planned horizons'}

def plot(outdir,data,summary,latency,rollout):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':240})
    outdir=pathlib.Path(outdir);outdir.mkdir(exist_ok=True,parents=True)
    def save(fig,name):
        fig.tight_layout();fig.savefig(outdir/(name+'.pdf'),bbox_inches='tight');fig.savefig(outdir/(name+'.png'),bbox_inches='tight');plt.close(fig)
    colors=['#7f7f7f','#0072B2','#E69F00','#009E73','#CC79A7'];labels=['Visible CV','Memory CV','Structured kinematics','Learned graph','Structured learned']
    fig,ax=plt.subplots(1,2,figsize=(10,3.9));hs=[.5,1,2,3]
    for name,label,c in zip(data['models'],labels,colors):
        vals=[summary['forecast'][name][str(h)] for h in hs];mu=[v['energy_score']['mean'] for v in vals]
        ax[0].plot(hs,mu,'o-',color=c,label=label);ax[0].fill_between(hs,[v['energy_score']['lo'] for v in vals],[v['energy_score']['hi'] for v in vals],color=c,alpha=.12)
        ax[1].plot(hs,[v['coverage90']['mean'] for v in vals],'o-',color=c)
    ax[0].set(xlabel='Physical horizon (s)',ylabel='Energy Score (m; lower is better)',title='Offline-aligned visible-target forecasts');ax[0].legend(fontsize=8)
    ax[1].axhline(.9,ls='--',color='black',lw=1);ax[1].set(xlabel='Physical horizon (s)',ylabel='Radial 90% region coverage',ylim=(0,1),title='Finite retained-world uncertainty')
    save(fig,'01_forecast_horizon')
    cats=['overall','positive','occluded_or_ambiguous','reappearance_proxy','aged_tracks','dense'];values=[summary['conditioning'][c] for c in cats]
    fig,ax=plt.subplots(figsize=(9,4.5));yy=np.arange(len(cats))
    for i,v in enumerate(values):
        if v['mean'] is not None:ax.errorbar(v['mean'],i,xerr=[[v['mean']-v['lo']],[v['hi']-v['mean']]],fmt='o',color='#0072B2',capsize=3)
    ax.axvline(0,color='black',lw=1);ax.axvline(.02,ls='--',color='#D55E00',label='Predeclared +0.020 m margin')
    ax.set_yticks(yy);ax.set_yticklabels([c.replace('_',' ')+' (n='+str(v['roots'])+', scenes='+str(v['scenes'])+')' for c,v in zip(cats,values)])
    ax.set(xlabel='Posterior − propagated prior Energy Score (m)',title='Same absolute 2 s target; 95% scene-bootstrap intervals');ax.legend(fontsize=8);save(fig,'02_observation_gain')
    fig,ax=plt.subplots(1,3,figsize=(11,3.7));ax[0].bar(['Mean','Covariance'],[max(1e-16,rollout['composition_mean_max_error']),max(1e-16,rollout['composition_covariance_max_error'])]);ax[0].set(yscale='log',title='Analytic OU composition',ylabel='Max absolute discrepancy')
    rr=rollout['recursive_fixed_3s'];ax[1].errorbar([r['depth'] for r in rr],[r['mean_error'] for r in rr],yerr=[r['mean_error_std'] for r in rr],fmt='o-');ax[1].set(xlabel='Decision-free quadrature steps',ylabel='Mean error to analytic mean (m)',title='Fixed 3 s, 32 seeds')
    ax[2].bar(['Branch mass','Retained count'],[max(1e-16,summary['consistency_max'][k]) for k in ('branch_probability_error','count_marginal_error_retained')]);ax[2].set(yscale='log',title='Within-call identities only');fig.suptitle('Observation segmentation still fails: P(no detection) 0.25 direct vs 0.0625 split',fontsize=11);save(fig,'03_recursive_consistency')
    fig,ax=plt.subplots(1,3,figsize=(11,3.5))
    for a,(name,v) in zip(ax,summary['risk'].items()):
        bins=v['calibration'];a.plot([0,1],[0,1],'--',color='gray');a.errorbar([r['predicted']['mean'] for r in bins],[r['observed']['mean'] for r in bins],yerr=[[r['observed']['mean']-r['observed']['lo'] for r in bins],[r['observed']['hi']-r['observed']['mean'] for r in bins]],fmt='o-',capsize=2,color='#0072B2');a.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted probability',ylabel='Native event frequency',title=name+(' (new-ID proxy)' if name=='appearance' else ''))
    fig.suptitle('Planned-horizon calibration; censored administrative outcomes excluded',fontsize=11);save(fig,'04_native_calibration')
    fig,ax=plt.subplots(1,3,figsize=(11,3.5));names=['depth0','memory_cv','structured'];ds=summary['decision'];vals=[ds[n+'_regret'] for n in names]
    ax[0].bar(names,[v['mean'] for v in vals],color=['gray','#0072B2','#CC79A7']);ax[0].errorbar(range(3),[v['mean'] for v in vals],yerr=[[v['mean']-v['lo'] for v in vals],[v['hi']-v['mean'] for v in vals]],fmt='none',color='black',capsize=3);ax[0].set(title='Fixed-rule candidate selection',ylabel='Regret to hindsight oracle')
    rs=[r for r in data['queries'] if r.get('risk_evaluable',True)]
    for a,key,label in zip(ax[1:],['duration','progress'],['Executed duration (s)','Reference-path travel (m)']):
        a.hexbin([r[key] for r in rs],[r['pred_'+key] for r in rs],gridsize=25,mincnt=1,bins='log',cmap='viridis');mx=max(max(r[key] for r in rs),max(r['pred_'+key] for r in rs));a.plot([0,mx],[0,mx],'--',color='gray');a.set(xlabel='Actual '+label,ylabel='Predicted '+label)
    save(fig,'05_query_decisions')
    fig,ax=plt.subplots(1,3,figsize=(12,3.7));keys=['observe','forecast','queries','complete'];xx=np.arange(len(keys))
    for i,p in enumerate(['p50','p95','p99']):ax[0].bar(xx+(i-1)*.24,[latency['summary']['warm'][k][p] for k in keys],width=.24,label=p)
    ax[0].set_xticks(xx);ax[0].set_xticklabels(keys,rotation=20);ax[0].set(ylabel='Wall-clock ms',title='Warm full path');ax[0].axhline(65,color='black',ls='--');ax[0].legend(fontsize=8)
    for mode,c in [('warm','#0072B2'),('cold_support_cache','#D55E00')]:
        rr=[r for r in latency['records'] if r['mode']==mode];ax[1].scatter([r['humans'] for r in rr],[r['timings_ms']['complete'] for r in rr],s=8,alpha=.3,label=mode,color=c);ax[2].scatter([r['graph_cells'] for r in rr],[r['timings_ms']['complete'] for r in rr],s=8,alpha=.3,color=c)
    ax[1].set(xlabel='Inferred human slots',ylabel='Complete call ms');ax[1].legend(fontsize=7);ax[2].set(xlabel='Observed graph cells',ylabel='Complete call ms');fig.suptitle('3 s · 16 worlds · 24 candidate rows (duplicates labeled) · 4 refined',fontsize=11);save(fig,'06_full_latency')
    root=outdir.parent;histories=[]
    for name in ('learned_218','learned_436','learned_s11_fixed','learned_posdiff'):
        path=root/name/'learning.json'
        if path.exists():histories.append((name,json.loads(path.read_text())))
    fig,ax=plt.subplots(1,2,figsize=(10,3.7))
    for name,h in histories:
        history=h['history'] if isinstance(h,dict) and 'history' in h else h
        if isinstance(history,list) and history:
            keys=history[0].keys();score=next((k for k in keys if 'energy' in k or k=='dev_es'),None)
            if score:ax[0].plot([r.get('epoch',i) for i,r in enumerate(history)],[r[score] for r in history],label=name.replace('learned_',''))
    ax[0].set(xlabel='Epoch',ylabel='Dev selector Energy Score',title='Training curves (selector estimand)');ax[0].legend(fontsize=8)
    size_scores=[json.loads((root/n/'training.json').read_text())['selected_dev_energy_score'] for n in ('learned_218','learned_436','learned_s11_fixed')];final_score=json.loads((root/'learned_posdiff'/'training.json').read_text())['selected_dev_energy_score'];ax[1].plot([218,436,872],size_scores,'o-',label='Original process law');ax[1].scatter([872],[final_score],marker='*',s=90,label='Positional diffusion revision');ax[1].set(xlabel='Independent supervised train roots',ylabel='Best dev selector Energy Score',title='20 / 20 / 30 epochs; one seed');ax[1].legend(fontsize=8)
    fig.suptitle('Fixed-cohort conditioning/calibration curves are shown in Figure 7b',fontsize=11);save(fig,'07_learning_curves')
    dd=rollout['depth'];fig,ax=plt.subplots(1,2,figsize=(9,3.6));ax[0].plot([r['depth'] for r in dd],[r['expected_utility'] for r in dd],'o-');ax[0].set(xlabel='Decision depth',ylabel='Expected synthetic utility',title='Two-route observation-opportunity oracle');ax[1].plot([r['depth'] for r in dd],[r['latency_p50_ms'] for r in dd],'o-',label='P50');ax[1].plot([r['depth'] for r in dd],[r['latency_p95_ms'] for r in dd],'s--',label='P95');ax[1].set(xlabel='Decision depth',ylabel='Wall-clock ms');ax[1].legend();fig.suptitle('Synthetic self-model utility only; no native depth-2 decision evidence',fontsize=11);save(fig,'08_compute_value')

def run(directory):
    root=pathlib.Path(directory);data=json.loads((root/'holdout_evaluation.json').read_text());dev=json.loads((root/'dev_evaluation_corrected.json').read_text())
    value=summarize(data,dev);(root/'release_metrics.json').write_text(json.dumps(value,indent=2))
    latency=json.loads((root/'latency_release.json' if (root/'latency_release.json').exists() else root/'latency_optimized.json').read_text());rollout=json.loads((root/'rollout_diagnostics.json').read_text())
    plot(root/'figures',data,value,latency,rollout);print(json.dumps({k:value[k] for k in ('query_support','decision','regression','unresolved_query_quantiles')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');run(**vars(p.parse_args()))
