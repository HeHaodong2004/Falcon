"""Standalone publication artifacts; truth is an offline overlay only."""
import argparse
import importlib.util
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .common import OUT, OLD, SNAP, DOC, read, canonical, truth_at, sha, dump, seed_for, AGE_BINS
from .model import IntermittentWorldModel
from .evaluate import exact_ids, extend_ancestry, lineage_slot, mask_advance
from eventwm_graph.l2_120k.model import load_model
from eventwm_graph.l2_120k.evaluate import restore

DEST=DOC/'figures'
VERIFIED_EVALUATOR_SHA='a278381a264b564eb51683469c89d307a41268805292ac856cec611434f3c184'


def install_verified_ancestry():
    """Use exactly the executed OFFLINE lifetime accounting for annotations."""
    global extend_ancestry,lineage_slot
    path=OUT/'primary_agent_notes/corrected_science_runtime/eventwm_graph/l2_intermittent/evaluate.py'
    if not path.exists():path=DOC/'artifacts/verified_execution/sources/eventwm_graph/l2_intermittent/evaluate.py'
    if sha(path)!=VERIFIED_EVALUATOR_SHA:raise ValueError('incorrect final figure ancestry accounting')
    spec=importlib.util.spec_from_file_location('eventwm_graph.l2_intermittent._figure_verified_evaluator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    extend_ancestry=module.extend_ancestry;lineage_slot=module.lineage_slot


def save(fig,name):
    fig.savefig(DEST/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(DEST/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def paths(model,b,ancestry,p,seed,horizon=2.):
    f=model.predict_shared(b,horizon,32,seed);xx=[];mode=[]
    target=int(np.argmin(abs(f.times-b.timestamp-min(1.,horizon))))
    for w,c in enumerate(f.endpoints[0]):
        i=lineage_slot(c,ancestry,p)
        if i is not None:xx.append(f.positions[w,:,i]);mode.append(bool(f.stopped[w,target,i]))
    return f,np.asarray(xx),mode


def panel(ax,model,b,ancestry,p,ts,truth,seed,title,show_last_measurement=True):
    f,x,modes=paths(model,b,ancestry,p,seed)
    for path,stop in zip(x,modes):ax.plot(path[:,0],path[:,1],color='#d07b20' if stop else '#247b9e',alpha=.25,lw=.8)
    if len(x):ax.plot(x[:,:,0].mean(0),x[:,:,1].mean(0),color='#204254',lw=2,label='sample mean')
    y=np.array([truth_at(ts,truth,float(t))[p] for t in f.times])
    ax.plot(y[:,0],y[:,1],'k--',lw=1.7,label='offline truth overlay')
    pairs=[(c,i) for c in b.components if (i:=lineage_slot(c,ancestry,p)) is not None]
    if pairs:
        c,i=pairs[0]
        if show_last_measurement:ax.scatter(*c.last_position[i],color='#257d45',s=36,zorder=5)
        age=sum(c.weight*(b.timestamp-c.last_seen[i]) for c,i in pairs)/sum(c.weight for c,i in pairs)
    else:age=None
    age_label='%.3fs'%age if age is not None else 'unsupported'
    ax.set_title(title+'\nage %s; lineage support %d/32'%(age_label,len(x)),fontsize=10)
    ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.16);ax.set_xlabel('x (m)');ax.set_ylabel('z (m)')
    return dict(timestamp=b.timestamp,age=age,supported_worlds=len(x),future=f,paths=x,modes=modes)


def masked_example(model,bootstrap,rows):
    chosen=None
    for row in rows:
        if not row['components'] or row['times'][-1]-row['times'][0]<2.5:continue
        folder=SNAP/'roots'/row['root_id'];legal=read(folder/'legal_root.json.gz');_,obs=canonical(folder)
        with np.load(folder/'privileged_future.npz') as f:ts=f['timestamps'].copy();truth=f['positions'][...,[0,2]].copy()
        now=row['timestamp']
        if ts[-1]<now+4.1:continue
        ids=exact_ids(legal['history'][-1],ts,truth)
        for p in sorted(ids):
            moving_root=any(np.linalg.norm(h.mean[i,2:])>.3 and np.linalg.norm(h.mean[i,:2]-ids[p])<2e-4
                            for h in row['filtered_hypotheses'] for i in range(h.n) if abs(h.last_seen[i]-now)<1e-7)
            if not moving_root:continue
            choices={age:min(range(len(obs)),key=lambda j:abs(obs[j]['timestamp']-now-age)) for age in (.5,1.,2.)}
            if any(abs(obs[j]['timestamp']-now-age)>.06 for age,j in choices.items()):continue
            j=next((j for j in range(choices[2.],len(obs)) if p in exact_ids(obs[j],ts,truth)),None)
            if j is not None and ts[-1]>=obs[j]['timestamp']+2.:
                chosen=(row,folder,legal,obs,ts,truth,p,choices,j);break
        if chosen:break
    if chosen is None:raise ValueError('no input-supported masked illustration')
    row,folder,legal,obs,ts,truth,p,choices,reob=chosen
    b=model.from_legacy(restore(bootstrap,row,legal['partial_graph']),seed_for(row['root_id'],10041));ancestry={};extend_ancestry(b,ancestry,ts,truth)
    fig,axes=plt.subplots(2,3,figsize=(15,10));panels=[]
    panels.append(panel(axes[0,0],model,b,ancestry,p,ts,truth,41,'Genuine root measurement'))
    for axis,age in zip((axes[0,1],axes[0,2],axes[1,0]),(.5,1.,2.)):
        aged=mask_advance(model,b,obs,choices[age]);panels.append(panel(axis,model,aged,ancestry,p,ts,truth,41,'Artificial mask: %.1fs requested'%age))
    prior=mask_advance(model,b,obs,reob)
    # Independently legal static updates were already applied by mask_advance.
    posterior=model.observe(prior,{**obs[reob],'static_updates':[]})
    panels.append(panel(axes[1,1],model,posterior,ancestry,p,ts,truth,41,'Actual legal reobservation'))
    data=panels[2];f=data['future'];x=data['paths']
    for path,stop in zip(x,data['modes']):axes[1,2].plot(f.times-f.times[0],np.linalg.norm(path-path[0],axis=-1),color='#d07b20' if stop else '#247b9e',alpha=.35,lw=.9)
    axes[1,2].set(xlabel='future time (s)',ylabel='displacement from current sample (m)',title='1 s-old belief: motion alternatives')
    axes[1,2].grid(alpha=.16)
    # Preserve spatial scale across observation ages and posterior correction.
    plotted=[q['paths'].reshape(-1,2) for q in panels if q['paths'].size]
    plotted.append(truth[(ts>=b.timestamp)&(ts<=posterior.timestamp+2.),p])
    points=np.concatenate(plotted);lo=points.min(0)-.25;hi=points.max(0)+.25
    for ax in (axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,1]):
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal',adjustable='box')
    legend=[Line2D([0],[0],color='#247b9e',label='Move mode at future +1s'),Line2D([0],[0],color='#d07b20',label='Stop mode at future +1s'),
            Line2D([0],[0],color='black',ls='--',label='offline truth overlay'),Line2D([0],[0],marker='o',color='#257d45',ls='',label='last legal measured center')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.025),ncol=4,fontsize=9)
    fig.suptitle('Artificially unavailable human measurements — not native occlusion\n'+row['scene_id']+' / '+row['root_id'][:12],fontsize=13)
    fig.subplots_adjust(left=.065,right=.985,bottom=.14,top=.84,wspace=.32,hspace=.65)
    save(fig,'01_masked_trajectory_sequence')
    return dict(root_id=row['root_id'],scene_id=row['scene_id'],physical_index_OFFLINE_ONLY=p,experiment='artificial sensor unavailability, actual legal reobservation',
        selection='first metadata root/person with legal-history speed>.3m/s, root positive, legal positive near2s, and >=2s subsequent truth; no score selection',
        panels=[{k:v for k,v in q.items() if k not in ('future','paths','modes')} for q in panels],
        masked_timestamps=[o['timestamp'] for o in obs[:reob+1]],reobservation_timestamp=obs[reob]['timestamp'],
        legal_sha256=sha(folder/'legal_root.json.gz'),future_legal_sha256=sha(folder/'future_observations.json.gz'),offline_truth_sha256=sha(folder/'privileged_future.npz'))


def native_example(model,bootstrap,rows):
    events=json.loads((OUT/'evaluation/reappearance_eligibility_verified.json').read_text());lookup={r['root_id']:r for r in rows}
    chosen=None
    for event in events:
        if event['stream']!=41 or not .75<=event['gap']<=1.25 or not event['future_available']['2.0']:continue
        chosen=event;break
    if chosen is None:return dict(status='native one-second illustration unsupported by frozen eligibility/truth')
    e=chosen;row=lookup[e['root_id']];folder=SNAP/'roots'/e['root_id'];legal=read(folder/'legal_root.json.gz');_,obs=canonical(folder)
    with np.load(folder/'privileged_future.npz') as f:ts=f['timestamps'].copy();truth=f['positions'][...,[0,2]].copy()
    b=model.from_legacy(restore(bootstrap,row,legal['partial_graph']),seed_for(row['root_id'],10041));ancestry={};extend_ancestry(b,ancestry,ts,truth)
    root_anchor_supported=(e['physical'] in exact_ids(legal['history'][-1],ts,truth) and
        all(lineage_slot(c,ancestry,e['physical']) is not None for c in b.components if c.weight>0))
    correspondence_path=DOC/'reviews/reviewer_b_final_canonical_correspondence.json'
    correspondence=json.loads(correspondence_path.read_text()) if correspondence_path.exists() else None
    flagged=correspondence is None or any((r['root_id'],r['index'],r['physical'])==
        (e['root_id'],e['index'],e['physical']) for r in correspondence['flagged_reappearances'])
    qualified=False
    try:
        for ob in obs[:e['index']]:b=model.observe(b,ob);extend_ancestry(b,ancestry,ts,truth)
        ob=obs[e['index']];prior=model.observe(b,{**ob,'tracks':[],'human_sensor_available':False});post=model.observe(b,ob)
        qualified=(not flagged and all(lineage_slot(c,ancestry,e['physical']) is not None
            for state in (prior,post) for c in state.components if c.weight>0))
        fig,axes=plt.subplots(1,2,figsize=(10,4.5));p=e['physical']
        aa=panel(axes[0],model,prior,ancestry,p,ts,truth,107,'Before native reappearance',qualified)
        bb=panel(axes[1],model,post,ancestry,p,ts,truth,107,'After actual anonymous positive',qualified)
        plotted=[q['paths'].reshape(-1,2) for q in (aa,bb) if q['paths'].size]
        plotted.append(truth[(ts>=prior.timestamp)&(ts<=prior.timestamp+2.),p])
        points=np.concatenate(plotted);lo=points.min(0)-.25;hi=points.max(0)+.25
        for ax in axes:
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal',adjustable='box')
        qualification=('Birth-qualified fixed ancestry; unambiguous correspondence' if qualified else
            'Descriptive example: lifetime or correspondence unverified')
        fig.suptitle('Native gap %.3fs; same absolute future target\n%s\n%s / %s'%
            (e['gap'],qualification,e['scene_id'],e['root_id'][:12]),fontsize=11)
        legend=[Line2D([0],[0],color='#247b9e',label='Move mode at future +1s'),
                Line2D([0],[0],color='#d07b20',label='Stop mode at future +1s'),
                Line2D([0],[0],color='black',ls='--',label='offline truth overlay'),
                Line2D([0],[0],marker='o',color='#257d45',ls='',label='last measurement of preferred lineage')]
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,0.),ncol=2,fontsize=8)
        fig.tight_layout(rect=[0,.15,1,.9]);save(fig,'02_native_reappearance')
        return dict(event=e,status='plotted',selection='first metadata annotation-verified frozen event with age[.75,1.25] and2s truth; no score selection',
            root_anchor_supported=root_anchor_supported,correspondence_ambiguous_or_unaudited=flagged,
            fixed_ancestry_supported=qualified,
            prior_support=aa['supported_worlds'],posterior_support=bb['supported_worlds'],
            prior_age=aa['age'],posterior_age=bb['age'],legal_sha256=sha(folder/'future_observations.json.gz'))
    except Exception as exc:return dict(event=e,status='selected_example_model_failure',error=repr(exc))


def summary_figures():
    learn=json.loads((OUT/'training/learning.json').read_text());fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([r['epoch'] for r in learn],[r['dev_es'] for r in learn],label='fixed development path ES')
    ax.plot([r['epoch'] for r in learn if r['train_es'] is not None],[r['train_es'] for r in learn if r['train_es'] is not None],label='stochastic training path ES',alpha=.7)
    best=min((r for r in learn if r['epoch']>0),key=lambda r:r['dev_es'])['epoch']
    ax.axvline(best,color='grey',ls='--',lw=1,label='selected epoch%d'%best);ax.set(xlabel='epoch',ylabel='component-conditional path Energy Score');ax.legend();ax.grid(alpha=.2);save(fig,'03_learning')
    summary=json.loads((OUT/'evaluation/summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    groups=[name for name,_,_ in AGE_BINS]
    for ax,cohort in zip(axes,('native','masked')):
        matrix=np.full((len(groups),4),np.nan);support=np.zeros_like(matrix,int)
        for r in summary['cells']:
            if r['model']=='D_StopMove' and r['cohort']==cohort and r['age_bin'] in groups:
                i=groups.index(r['age_bin']);j=[.5,1.,2.,3.].index(r['H']);m=r['metrics']['path_es'];matrix[i,j]=m['mean'] if m['mean'] is not None else np.nan;support[i,j]=m['roots']
        image=ax.imshow(matrix,aspect='auto',cmap='YlGnBu',vmin=0,vmax=2)
        for i in range(len(groups)):
            for j in range(4):ax.text(j,i,('%.2f\nn=%d'%(matrix[i,j],support[i,j])) if np.isfinite(matrix[i,j]) else 'unsupported',ha='center',va='center',fontsize=8,color='white' if np.isfinite(matrix[i,j]) and matrix[i,j]>1.1 else 'black')
        ax.set_xticks(range(4));ax.set_xticklabels([.5,1.,2.,3.]);ax.set_yticks(range(len(groups)));ax.set_yticklabels(groups);ax.set(xlabel='future horizon (s)',ylabel='last-observed age bin (s)',title=cohort+' — D path ES / unique roots')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='RMS-scaled path ES (m)',shrink=.8);save(fig,'04_age_horizon_support')


def latency_figure():
    names=('reviewer_c_systems_final_v2','reviewer_c_systems_speed004_v1',
           'reviewer_c_systems_speed007_v1','reviewer_c_systems_speed008_v1')
    benchmarks=[json.loads((OUT/name/'benchmark.json').read_text()) for name in names]
    if len({b['checkpoint_sha256'] for b in benchmarks})!=1:raise ValueError('latency checkpoint changed')
    if len({b['cohort_sha256'] for b in benchmarks})!=1:raise ValueError('latency cohort changed')
    labels=('Frozen initial implementation','Compiled beam and trajectory loops',
            'Batched filtering and repeated inputs','Exact keyed RNG setup (final)')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    y=np.arange(4);colors=('#247ba0','#e88b3a')
    for ax,key,title in zip(axes,('representative_first60_quantiles','dense_quantiles'),
                           ('Representative first 60 roots','Dense stress 12 roots')):
        for k,cache in enumerate(('warm','dynamic_cold')):
            values=[b[key][repr((cache,16,'primary'))]['metrics']['combined_ms']['p95_ms'] for b in benchmarks]
            bars=ax.barh(y+(k-.5)*.32,values,height=.3,color=colors[k],label=cache.replace('_',' '))
            for bar,value in zip(bars,values):ax.text(value+5,bar.get_y()+bar.get_height()/2,'%.1f'%value,va='center',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.2))
        ax.axvline(100,color='#ba332d',ls='--',lw=1.2,label='100 ms target')
        ax.axvline(65,color='grey',ls=':',lw=1,label='65 ms preferred')
        ax.set(title=title,xlabel='Complete-call P95 latency (ms)',xlim=(0,600))
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_yticks(y);axes[0].set_yticklabels(labels);axes[0].invert_yaxis()
    axes[1].legend(loc='lower right',fontsize=8)
    fig.suptitle('Same checkpoint · 32 components · 3 s / 16 worlds / 64 timestamps · optics and successor',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'05_latency')
    dump(DEST/'latency_sources.json',dict(files={name:sha(OUT/name/'benchmark.json') for name in names},
        scope='Measured fixed-cohort complete-call P95; startup is separate; dense stress still exceeds100ms. No prediction support or conditional work reduced.'))


def run():
    import torch
    torch.set_num_threads(1);DEST.mkdir(exist_ok=True)
    install_verified_ancestry()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    model=IntermittentWorldModel.load(OUT/'training/checkpoint.pt');bootstrap=load_model();rows=list(np.load(OLD/'dev.npy',allow_pickle=True))
    manifest=dict(masked=masked_example(model,bootstrap,rows),native=native_example(model,bootstrap,rows),model_version=model.model_version,
        offline_accounting_sha256=VERIFIED_EVALUATOR_SHA,
        checkpoint_sha256=sha(OUT/'training/checkpoint.pt'),semantics='All truth is separate OFFLINE evaluation overlay; no truth or physical index enters inference; illustration criteria never change scores')
    summary_figures();latency_figure();manifest['files']={p.name:sha(p) for p in sorted(DEST.iterdir()) if p.suffix in ('.png','.pdf')};dump(DEST/'manifest.json',manifest)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--latency-only',action='store_true');a=p.parse_args()
    DEST.mkdir(exist_ok=True);latency_figure() if a.latency_only else run()
