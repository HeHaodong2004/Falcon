"""Plot recorded repair evidence only; no simulation, fitting or data access.

Usage: python scripts/l2_partial_repair_report.py <repair-output-directory>
Every required input must exist. Missing diagnostics are reported explicitly;
an unavailable quantity never becomes a zero-valued scientific result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {'legacy':'Released belief', 'association_only':'Association repair',
          'full':'Complete repair', 'full_no_maneuver':'No maneuver',
          'full_kinematic':'Kinematic repair', 'full_unfitted_optics':'Unfitted optics'}
COLORS = dict(zip(LABELS, ['#777777','#e69f00','#0072b2','#cc79a7','#009e73','#d55e00']))
EVENT_COLORS = ['#777777','#e69f00','#cc79a7','#0072b2']


def load(path):
    return json.loads(path.read_text())


def point(ci):
    return ci.get('mean')


def interval(ax, x, ci, color, label=None, marker='o', horizontal=False):
    value = point(ci)
    if value is None:
        return
    lo, hi = ci.get('lo'), ci.get('hi')
    err = None if lo is None or hi is None else [[max(0., value-lo)], [max(0., hi-value)]]
    kwargs = dict(fmt=marker, color=color, capsize=2, label=label, ms=4)
    if horizontal:
        ax.errorbar(value, x, xerr=err, **kwargs)
    else:
        ax.errorbar(x, value, yerr=err, **kwargs)


def save(fig, directory, name):
    fig.tight_layout()
    for suffix in ('pdf','png'):
        fig.savefig(directory/(name+'.'+suffix), dpi=240, bbox_inches='tight')
    plt.close(fig)


def population_fig(evaluation, directory):
    summary=evaluation['summary'];names=list(summary);x=np.arange(len(names))
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,key,title in zip(axes, ['count_absolute_error','gospa','unresolved_mass'],
                            ['Population count MAE','Localized GOSPA (m)','Unresolved mass']):
        for i,name in enumerate(names):
            interval(ax,i,summary[name]['population'][key],COLORS[name])
        ax.set_xticks(x);ax.set_xticklabels([LABELS[n] for n in names], rotation=45,ha='right')
        ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed development cohort; equal-scene means and scene-bootstrap 95% intervals',fontsize=10)
    save(fig,directory,'01_population')


def forecast_fig(evaluation,directory,trajectory):
    fig,axes=plt.subplots(1,3,figsize=(13,3.8));horizons=[.5,1.,2.,3.]
    for name,summary in evaluation['summary'].items():
        scores=[summary['forecast'][str(h)]['ancestry_energy_score'] for h in horizons]
        x=[h for h,c in zip(horizons,scores) if point(c) is not None]
        y=[point(c) for c in scores if point(c) is not None]
        axes[0].plot(x,y,'o-',ms=3,color=COLORS[name],label=LABELS[name])
        for h,c in zip(horizons,scores):interval(axes[0],h,c,COLORS[name])
        coverage=[summary['forecast'][str(h)]['ancestry_coverage90'] for h in horizons]
        axes[1].plot([h for h,c in zip(horizons,coverage) if point(c) is not None],
                     [point(c) for c in coverage if point(c) is not None], 'o-',ms=3,color=COLORS[name])
        joint=[trajectory['summary'][name][str(h)] for h in horizons]
        axes[2].plot([h for h,c in zip(horizons,joint) if point(c) is not None],
                     [point(c) for c in joint if point(c) is not None], 'o-',ms=3,color=COLORS[name])
        for h,c in zip(horizons,joint):interval(axes[2],h,c,COLORS[name])
    axes[0].set_ylabel('Energy Score (m; lower is better)');axes[1].set_ylabel('Coverage of 90% sampled region')
    axes[2].set_ylabel('Joint-time RMS-scaled Energy Score (m)')
    axes[2].set_title('Post-freeze metric supplement: 2–4 scored roots',fontsize=8)
    axes[1].axhline(.9,color='black',ls=':',lw=1);axes[1].set_ylim(0,1.02)
    for ax in axes:ax.set_xlabel('Absolute forecast horizon (s)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Fixed ancestry: all-cohort endpoints and selected-cohort joint-time diagnostic; conditional on represented mass',fontsize=9)
    save(fig,directory,'02_forecast_horizon')


def conditioning_fig(evaluation,directory):
    categories=['overall','positive','reappearance_physical','aged_tracks','dense','association_ambiguous','occluded_or_ambiguous']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,horizon in zip(axes,[1.,2.]):
        for j,name in enumerate(['legacy','association_only','full']):
            summary=evaluation['summary'][name]['conditioning_by_horizon'][str(horizon)]
            for i,category in enumerate(categories):
                ci=summary.get(category)
                if ci is None:continue
                interval(ax,i+(j-1)*.2,ci,COLORS[name],LABELS[name] if i==0 else None,horizontal=True)
                if name=='full' and ci.get('roots'):
                    ax.annotate(str(ci['roots'])+' roots',(point(ci),i+.26),fontsize=6,color=COLORS[name])
        ax.axvline(0,color='black',lw=.8);ax.axvline(.020,color='#aa3333',ls=':',lw=1)
        ax.set_title(('Secondary' if horizon==1 else 'Primary')+' target: root + '+str(horizon)+' s')
        ax.set_xlabel('Posterior − prior Energy Score (m)');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(categories)));axes[0].set_yticklabels([c.replace('_',' ') for c in categories])
    axes[0].invert_yaxis();axes[0].legend(fontsize=7,loc='lower left')
    fig.suptitle('Same observation time, target and survival evidence; scene-bootstrap 95% intervals',fontsize=10)
    save(fig,directory,'03_observation_conditioning')


def risk_fig(report,directory):
    fig,axes=plt.subplots(1,3,figsize=(12,3.7));styles=EVENT_COLORS
    for ax,event,title in zip(axes,['collision','psc','untracked_appearance_proxy'],
                             ['Native collision','Native PSC','Untracked appearance proxy']):
        for (name,model),color in zip(report['models'].items(),styles):
            bins=model['events'][event]['calibration_bins']
            valid=[b for b in bins if point(b['predicted']) is not None and point(b['observed']) is not None]
            ax.plot([point(b['predicted']) for b in valid],[point(b['observed']) for b in valid],
                    'o-',ms=3,label=name,color=color)
            for b in valid:interval(ax,point(b['predicted']),b['observed'],color)
        ax.plot([0,1],[0,1],':',color='black',lw=1);ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.set_title(title);ax.set_xlabel('Predicted probability');ax.grid(alpha=.2)
    axes[0].set_ylabel('Observed event frequency');axes[0].legend(fontsize=7)
    fig.suptitle('Complete-followup planned-horizon development calibration; scene intervals',fontsize=10)
    save(fig,directory,'05_native_risk_calibration')


def decision_fig(report,directory):
    models=report['models'];names=list(models);fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for j,(name,model) in enumerate(models.items()):
        for i,event in enumerate(['collision','psc']):
            interval(axes[0],i+(j-(len(models)-1)/2)*.17,model['events'][event]['within_root_auc']['summary'],
                     EVENT_COLORS[j],name if i==0 else None)
    axes[0].set_xticks([0,1]);axes[0].set_xticklabels(['Collision','PSC']);axes[0].set_ylim(0,1.03)
    axes[0].set_ylabel('Within-root concordance');axes[0].legend(fontsize=7)
    final=models[names[-1]]['decision']
    rules=list(final['rules'])
    for i,rule in enumerate(rules):interval(axes[1],i,final['rules'][rule]['regret'],'#0072b2')
    axes[1].set_xticks(range(len(rules)));axes[1].set_xticklabels(['Expected utility','Residual upper risk','Depth 0'],rotation=25,ha='right')
    axes[1].set_ylabel('Hindsight regret (fixed utility)')
    for i,key in enumerate(['duration_absolute_error','progress_absolute_error']):interval(axes[2],i,final[key],'#0072b2')
    axes[2].set_xticks([0,1]);axes[2].set_xticklabels(['Duration (s)','Traveled distance (m)'])
    axes[2].set_ylabel('Absolute prediction error')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Observed curriculum candidate sets; hindsight is diagnostic, no closed-loop navigation claim',fontsize=10)
    save(fig,directory,'06_query_decision_information')


def latency_fig(profile,directory):
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    modes=['warm','fully_cold','graph_revalidation'];quantiles=['p50','p95','p99']
    for j,q in enumerate(quantiles):
        axes[0].bar(np.arange(3)+(j-1)*.24,
                    [profile['summary'][mode]['timings_ms']['outer_complete'][q] for mode in modes],
                    width=.23,label=q.upper(),color=['#56b4e9','#0072b2','#08306b'][j])
    axes[0].set_xticks(range(3));axes[0].set_xticklabels(['Warm','Cold','Revalidation'],rotation=20)
    for level,color in [(65,'#009e73'),(100,'#d55e00')]:axes[0].axhline(level,color=color,ls=':',lw=1)
    axes[0].set_ylabel('Complete online call (ms)');axes[0].legend(fontsize=7)
    for field,ax,label in [('tracked_slots_max',axes[1],'Maximum tracked slots'),('graph_cells',axes[2],'Certified graph cells')]:
        groups={}
        for r in profile['records']:
            if r['mode']=='warm':groups.setdefault(r['root_id'],[]).append(r)
        for group in groups.values():
            ax.scatter(group[0][field],np.percentile([r['timings_ms']['outer_complete'] for r in group],95),s=15,alpha=.6,color='#0072b2')
        ax.set_xlabel(label);ax.set_ylabel('Per-root warm P95 (ms)');ax.axhline(100,color='#d55e00',ls=':',lw=1)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('16 worlds, 3 s, 24 candidate rows, four refined queries; active collector',fontsize=10)
    save(fig,directory,'07_full_call_latency')


def learning_fig(event,optical,directory):
    fig,axes=plt.subplots(1,2,figsize=(10,3.7));history=event['history']
    axes[0].plot([r['epoch'] for r in history],[r['dev_censored_nll'] for r in history],'o-',color='#0072b2',label='Dev censored NLL')
    rows=[r for r in history if r['train_objective'] is not None]
    axes[0].plot([r['epoch'] for r in rows],[r['train_objective'] for r in rows],'o-',color='#e69f00',label='Train objective + regularizer')
    axes[0].set_xlabel('Event-head epoch');axes[0].set_ylabel('Root-balanced score');axes[0].legend(fontsize=7)
    for j,split in enumerate(['train','dev']):
        axes[1].plot([0,1],[optical[split+'_initial_root_nll'],optical[split+'_fitted_root_nll']],
                     'o-',label=split,color=['#e69f00','#0072b2'][j])
    axes[1].set_xticks([0,1]);axes[1].set_xticklabels(['Initial','Fitted']);axes[1].set_ylabel('Observation root NLL')
    axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Bounded fitting on existing data; graph dynamics checkpoint held fixed',fontsize=10)
    save(fig,directory,'08_bounded_learning')


def probability_fig(diagnostics,directory):
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    synthetic=diagnostics['synthetic_segmentation']['records'][0]
    histories=synthetic['histories'];x=np.arange(len(histories))
    axes[0].bar(x-.16,[r['oracle_probability'] for r in histories],width=.3,color='#777777',label='Analytic')
    axes[0].bar(x+.16,[r['recursive_probability'] for r in histories],width=.3,color='#0072b2',label='Recursive API')
    axes[0].set_xticks(x);axes[0].set_xticklabels([r['history'] for r in histories])
    axes[0].set_ylabel('Joint observation probability');axes[0].legend(fontsize=7)
    axes[0].set_title('Known supported synthetic law')
    for name,color in [('quadrature_default','#777777'),('quadrature_fitted','#0072b2')]:
        q=diagnostics[name]
        rows=q['tracked_misses']
        if rows:
            axes[1].loglog([r['position_variance'] for r in rows],
                          [max(1e-16,r['mass_absolute_error']) for r in rows],'o-',color=color,label=name.replace('quadrature_',''))
    axes[1].set_xlabel('Position variance (m²)');axes[1].set_ylabel('Miss-probability quadrature error')
    if axes[1].lines:axes[1].legend(fontsize=7)
    summaries=[r for r in diagnostics['real']['depth_drift_summary'] if r['regularization']=='configured']
    for key,label,color in [('resolved_terminal','Contact','#d55e00'),('resolved_survived_no_detection','Survived, no detection','#0072b2')]:
        axes[2].plot([r['depth_vs_one'] for r in summaries],
                     [r['metrics'][key]['mean_absolute_difference'] for r in summaries],'o-',label=label,color=color)
    axes[2].set_xlabel('Decision stages over fixed physical interval');axes[2].set_ylabel('Mean absolute probability drift vs depth 1')
    axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Exact oracle identities and measured real-scene approximation error are separate',fontsize=10)
    save(fig,directory,'04_recursive_probability')


def computation_depth_fig(diagnostics,directory):
    records=[r for r in diagnostics['real']['recursive_api'] if r['regularization']=='configured']
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for depth in [1,2,3]:
        rows=[r for r in records if r['depth']==depth]
        times=[r['wall_ms'] for r in rows]
        if times:
            axes[0].plot([depth]*3,np.percentile(times,[50,95,99]),'o-',color='#0072b2')
            axes[0].annotate('P50/95/99',(depth,min(times)),fontsize=6)
    axes[0].set_ylabel('Recursive one-query computation (ms)')
    summaries=[r for r in diagnostics['real']['depth_drift_summary'] if r['regularization']=='configured']
    axes[1].plot([r['depth_vs_one'] for r in summaries],
                 [r['metrics']['resolved_terminal']['maximum_absolute_difference'] for r in summaries],'o-',color='#d55e00')
    axes[1].set_ylabel('Maximum contact-probability drift')
    speed=diagnostics['real']['speed_plausibility'];horizons=[.5,1.,2.,3.]
    for key,label,color in [('probability_above_3mps','Speed >3 m/s','#e69f00'),('probability_above_5mps','Speed >5 m/s','#d55e00')]:
        means=[]
        for h in horizons:
            values=[v['all_instantaneous'][key] for root in speed for v in root['model_records']
                    if abs(v['horizon_s']-h)<1e-8 and v['all_instantaneous'].get(key) is not None]
            means.append(float(np.mean(values)) if values else np.nan)
        axes[2].plot(horizons,means,'o-',label=label,color=color)
    axes[2].set_xlabel('Physical forecast horizon (s)');axes[2].set_ylabel('Mean retained-person exceedance probability')
    axes[2].legend(fontsize=7)
    for ax in axes[:2]:ax.set_xlabel('Decision stages, fixed physical duration');ax.set_xticks([1,2,3])
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Development diagnostic only: additional computation is not established decision utility',fontsize=10)
    save(fig,directory,'09_computation_depth_and_plausibility')


def ghost_fig(diagnostics,reproduced,directory):
    trace=diagnostics['ghost']['trace'];old=reproduced['ghost_population']['replay']
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].plot([r['frame'] for r in old],[r['retained_expected_total'] for r in old],'o-',color='#777777',label='Released model')
    axes[0].plot([r['frame'] for r in trace],[r['posterior_population']['conditional_expected_total'] for r in trace],
                 'o-',color='#0072b2',label='Complete repair')
    axes[0].axhline(6,color='black',ls=':',label='Offline truth (evaluation only)')
    axes[0].set_ylabel('Resolved expected population');axes[0].legend(fontsize=7)
    for key,label,color in [('birth_probability','Any first localization','#e69f00'),
                            ('pruned_posterior_mass','Pruned posterior mass','#cc79a7')]:
        axes[1].semilogy([r['frame'] for r in trace],[max(1e-18,r[key]) for r in trace],'o-',label=label,color=color)
    axes[1].set_ylabel('Probability (log scale)');axes[1].legend(fontsize=7)
    for ax in axes:ax.set_xlabel('Arrived legal observation frame');ax.grid(alpha=.2)
    fig.suptitle('Preserved six-person ghost failure; no track deletion or online true-count input',fontsize=10)
    save(fig,directory,'10_ghost_track_replay')


def observation_law_fig(report,directory):
    fig,axes=plt.subplots(1,3,figsize=(12,3.8));names=list(report['models'])
    for ax,kind,title in zip(axes,['contact','survived_no_detections','survived_any_detections'],
                            ['Absorbing contact','Survived, no detection','Survived, any detection']):
        for i,(name,model) in enumerate(report['models'].items()):
            summary=model['classes'][kind]['summary']
            lo=point(summary['predicted_lower']);hi=point(summary['predicted_upper'])
            # The horizontal segment is a probability interval over unspecified
            # residual optical mass. Its midpoint is not a point prediction.
            ax.hlines(i,lo,hi,color=EVENT_COLORS[i],lw=5,label='Model mass interval' if i==0 else None)
            ax.plot([lo,hi],[i,i],'|',color=EVENT_COLORS[i],ms=9)
            interval(ax,i+.12,summary['event_rate'],'black','Observed ± scene CI' if i==0 else None,
                     marker='d',horizontal=True)
        ax.set_xlim(0,1);ax.set_yticks(range(len(names)));ax.set_yticklabels(names)
        ax.set_title(title);ax.set_xlabel('Scene-macro probability');ax.grid(axis='x',alpha=.2)
    axes[0].legend(fontsize=6,loc='upper right')
    fig.suptitle('Generative observation law: residual-mass bounds and statistical intervals are distinct',fontsize=10)
    save(fig,directory,'11_generative_observation_events')


def main(directory):
    directory=Path(directory).resolve();figures=directory/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    inputs={name:directory/file for name,file in {
        'population':'population_evaluation.json','risk':'risk_scene_calibrated.json',
        'latency':'latency_v5_active.json','event':'native_events_v5_scene.json',
        'optical':'observation_fit.json','probability':'probability_diagnostics_v5_final.json',
        'trajectory':'trajectory_diagnostic.json',
        'observation_events':'observation_calibration_scene.json',
        'reproduced':'reproduced_release_failures.json'}.items()}
    data={name:load(path) for name,path in inputs.items()}
    population_fig(data['population'],figures);forecast_fig(data['population'],figures,data['trajectory'])
    conditioning_fig(data['population'],figures);risk_fig(data['risk'],figures)
    decision_fig(data['risk'],figures);latency_fig(data['latency'],figures)
    learning_fig(data['event'],data['optical'],figures)
    probability_fig(data['probability'],figures);computation_depth_fig(data['probability'],figures)
    ghost_fig(data['probability'],data['reproduced'],figures)
    observation_law_fig(data['observation_events'],figures)
    manifest={'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.values()},
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'figures':[p.name for p in sorted(figures.glob('*'))],
              'scope':'recorded development data only; no regenerated or fabricated measurements'}
    (figures/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory')
    main(parser.parse_args().directory)
