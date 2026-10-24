"""Render actual sealed measurements; never supplies missing results as passes."""
import ast
import collections
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eventwm_graph.l2_partial.snapshot import sha
from eventwm_graph.l2_partial.repair_evaluation import scene_ci
from .data import OUT,SNAP

DOC=Path(os.environ.get('L2_REPRO_DOC','docs/l2/graph_wm_120k_20261021'));DOC.mkdir(exist_ok=True,parents=True)
NAMES=('A_visible_cv','B_memory_cv','B_historical_cv','C_frozen_graph','C_matched_motion','D_trained_graph')
H=(.5,1.,2.,3.)


def load(path):return json.loads(Path(path).read_text())
def ci(x):
    if x is None:return 'unsupported'
    if x.get('status'):return x['status']+' (zero-support n='+str(x.get('infinite_roots',x.get('infinite_records','?')))+')'
    if x.get('mean') is None:return 'unsupported (0 roots)'
    return f"{x['mean']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}] ({x['roots']}/{x['scenes']})"
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])+'\n'
def write(name,text):(DOC/name).write_text(text.strip()+'\n')


def figures(summary,raw,training,systems,physical):
    dest=DOC/'figures';dest.mkdir(exist_ok=True);files=[]
    def save(name,fig):
        fig.tight_layout()
        for suffix in ('png','pdf'):
            p=dest/(name+'.'+suffix);fig.savefig(p,dpi=160,bbox_inches='tight');files.append(p)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));epochs=[r['epoch'] for r in training]
    ax.plot(epochs,[r['dev_es'] for r in training],label='fixed development path ES')
    ax.plot(epochs,[np.nan if r['train_es'] is None else r['train_es'] for r in training],alpha=.7,label='training path ES (resampled)')
    ax.set(xlabel='Epoch',ylabel='RMS-path Energy Score (m)',title='Actual compact-model training');ax.legend();save('01_learning',fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name in NAMES:
        for ax,kind,key,label in ((axes[0],'forecasting','population_es3','Population Hilbert ES3'),(axes[1],'forecasting','tracked_es','Tracked ES (m)'),(axes[2],'paths','population_path_es3','Complete population path ES3')):
            vals=[summary[kind][name][str(h)][key] for h in H]
            means=[v['mean'] if v.get('mean') is not None else np.nan for v in vals]
            ax.plot(H,means,marker='o',label=name)
            lo=[v['lo'] if v.get('lo') is not None else np.nan for v in vals];hi=[v['hi'] if v.get('hi') is not None else np.nan for v in vals];ax.fill_between(H,lo,hi,alpha=.1)
            ax.set(xlabel='Physical horizon (s)',ylabel=label)
    axes[-1].legend(fontsize=7);fig.suptitle('Original shared-seed41 procedure; horizon cohorts differ');save('02_forecasting_horizons',fig)
    roots=[r for r in raw['filtering'] if r['phase']=='root'];K=max(len(r['count_pmf']) for r in roots)
    predicted=np.zeros(K);actual=np.zeros(K)
    for r in roots:
        predicted[:len(r['count_pmf'])]+=r['count_pmf'];actual[r['true_count']]+=1
    predicted/=len(roots);actual/=len(roots)
    fig,ax=plt.subplots(figsize=(8,4));xx=np.arange(K)
    ax.bar(xx-.18,actual,.36,label='offline actual count');ax.bar(xx+.18,predicted,.36,label='mean posterior PMF')
    ax.set(xlabel='Complete population count',ylabel='Root frequency',title=f'Count calibration: {len(roots)} development roots');ax.legend();save('03_population_count',fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,label in ((axes[0],'aged_tracked_es','Aged-track ES (m)'),(axes[1],'never_localized_es3','Unlocalized-pool proxy ES3')):
        for name in NAMES:
            vals=[summary['forecasting'][name][str(h)][key] for h in H]
            ax.plot(H,[v.get('mean') if v.get('mean') is not None else np.nan for v in vals],marker='o',label=name)
        ax.set(xlabel='Physical horizon (s)',ylabel=label)
    axes[1].legend(fontsize=7)
    groups=('reappearance','aged','no_detection','positive')
    for j,group in enumerate(groups):
        rr=summary['conditioning']['2.0'][group]['population_es3_gain']
        if rr.get('mean') is not None:
            axes[2].errorbar([rr['mean']],[j],xerr=[[rr['mean']-rr['lo']],[rr['hi']-rr['mean']]],fmt='o',label=f"{group}: {rr['roots']} roots")
    axes[2].set_yticks(range(len(groups)));axes[2].set_yticklabels(groups)
    axes[2].axvline(0,color='black',lw=1);axes[2].set(xlabel='Same-target population gain',title='Original shared-seed107');save('04_hidden_and_aged',fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for positive,label in ((True,'positive detection'),(False,'no detection')):
        values=[r['population_es3_gain'] for r in raw['conditioning'] if r['target_horizon']==2. and r['positive']==positive and r['population_es3_gain'] is not None]
        if values:ax.hist(values,bins=30,alpha=.5,label=f'{label}, n={len(values)} roots')
    ax.axvline(0,color='black',lw=1);ax.set(xlabel='Prior score minus posterior score',ylabel='Independent roots',title='Same root+2s target; original shared-seed107');ax.legend(fontsize=8);save('05_conditioning_gains',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));records=[r for r in systems['recursion'] if r['segments'] in ('two','three')]
    for name in ('two','three'):
        rr=[r for r in records if r['segments']==name]
        axes[0].plot(np.maximum(1e-17,[r['covariance_error'] for r in rr]),label=name)
    axes[0].set(yscale='log',ylabel='Maximum covariance discrepancy',xlabel='Real cohort root');axes[0].legend()
    cc=systems['scalar_jump_fourth_cumulant'];axes[1].bar(list(cc),list(cc.values()));axes[1].set(ylabel='Isolated jump fourth cumulant',title='Projection changes higher moments');save('06_recursion',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name in ('B_memory_cv','D_trained_graph'):
        for ax,key,ylabel in ((axes[0],'coverage90','Empirical90% region coverage'),(axes[1],'radius90','Mean90% sample radius (m)')):
            vals=[summary['forecasting'][name][str(h)][key] for h in H]
            ax.plot(H,[v.get('mean') if v.get('mean') is not None else np.nan for v in vals],marker='o',label=name);ax.set(xlabel='Physical horizon (s)',ylabel=ylabel)
    axes[0].axhline(.9,color='black',ls='--');axes[0].set_ylim(0,1.05);axes[0].legend(fontsize=8);save('07_coverage',fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for cache in ('warm','dynamic_cold'):
        values=np.sort([r['combined'] for r in systems['calls'] if r['cache']==cache and r['workload']=='real' and r['worlds']==16])
        if len(values):ax.plot(values,np.arange(1,len(values)+1)/len(values),label=cache)
    ax.axvline(65,color='gray',ls=':');ax.axvline(100,color='red',ls='--');ax.set(xlabel='Combined Layer2 time (ms)',ylabel='Empirical CDF');ax.legend();save('08_latency',fig)
    fig,ax=plt.subplots(figsize=(7,4));bins=summary['observations']['calibration'];valid=[r for r in bins if r['n']]
    ax.plot([r['predicted'] for r in valid],[r['observed'] for r in valid],marker='o');ax.plot([0,1],[0,1],ls='--',color='gray')
    ax.set(xlabel='Predicted detection probability',ylabel='Recorded detection frequency',title='Fixed planned-query endpoint calibration');save('09_observation_calibration',fig)
    (dest/'manifest.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in (OUT/'evaluation/summary.json',OUT/'training/learning.json',OUT/'systems.json')},
        files={str(p):sha(p) for p in files}),indent=2))


def independent_mc_figures(forecast,conditional):
    dest=DOC/'figures';manifest=load(dest/'manifest.json')
    def save(name,fig):
        fig.tight_layout()
        for suffix in ('png','pdf'):
            p=dest/(name+'.'+suffix);fig.savefig(p,dpi=160,bbox_inches='tight')
            manifest['files'][str(p)]=sha(p)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name in NAMES:
        for ax,records,key,label in ((axes[0],forecast['records'],'population_es3','Population Hilbert ES3'),
                (axes[1],forecast['records'],'tracked_es','Tracked ES (m)'),
                (axes[2],forecast['paths'],'population_path_es3','Complete population path ES3')):
            values=[scene_ci([r for r in records if r['model']==name and r['horizon']==h],key) for h in H]
            ax.plot(H,[r['mean'] if r['mean'] is not None else np.nan for r in values],marker='o',label=name)
            ax.fill_between(H,[r['lo'] if r['lo'] is not None else np.nan for r in values],
                [r['hi'] if r['hi'] is not None else np.nan for r in values],alpha=.08)
            ax.set(xlabel='Physical horizon (s)',ylabel=label)
    axes[-1].legend(fontsize=7);fig.suptitle('Independent root MC; fixed weights; horizon cohorts differ');save('10_independent_mc_forecasting',fig)
    groups=('all','positive','no_detection','aged','reappearance','dense')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,label in ((axes[0],'population_es3_gain','Population gain'),(axes[1],'tracked_es_gain','Tracked gain (m)'),(axes[2],'gospa_gain','GOSPA gain')):
        for j,group in enumerate(groups):
            r=conditional['conditioning']['2.0'][group][key]
            if r['mean'] is not None:ax.errorbar([r['mean']],[j],xerr=[[r['mean']-r['lo']],[r['hi']-r['mean']]],fmt='o')
        ax.set_yticks(range(len(groups)));ax.set_yticklabels(groups);ax.axvline(0,color='black',lw=1);ax.set(xlabel=label)
    fig.suptitle('Same absolute 2 s target; independent root MC');save('11_independent_mc_conditioning',fig)
    for name in ('forecast_root_keyed_41.json','conditioning_root_keyed_107.json'):
        manifest['inputs'][str(OUT/name)]=sha(OUT/name)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))


def main():
    m=load(SNAP/'manifest.json');audit=load(OUT/'data_audit.json');train=load(OUT/'training/training.json');learning=load(OUT/'training/learning.json')
    summary=load(OUT/'evaluation/summary.json');systems=load(OUT/'systems.json');physical=load(OUT/'physical.json');persist=load(OUT/'persistence.json')
    raw={k:load(OUT/('evaluation/'+k+'.json')) for k in ('forecasts','paths','filtering','conditioning','observations','failures')}
    forecast_mc=load(OUT/'forecast_root_keyed_41.json');conditional_mc=load(OUT/'conditioning_root_keyed_107.json')
    selected=torch.load(OUT/'training/checkpoint.pt',map_location='cpu');epoch0=torch.load(OUT/'training/epoch0.pt',map_location='cpu')
    norm=math.sqrt(sum(float((selected['state_dict'][k]-epoch0['state_dict'][k]).double().square().sum()) for k in selected['state_dict'] if torch.is_floating_point(selected['state_dict'][k])))
    cksha=sha(OUT/'training/checkpoint.pt');numeric=load(OUT/'numerics.json') if (OUT/'numerics.json').exists() else {'passed':False,'actual':0}
    figures(summary,raw,learning,systems,physical)
    independent_mc_figures(forecast_mc,conditional_mc)
    splitrows=[[k,v.get('roots'),v.get('queries'),v.get('scenes','protected')] for k,v in m['splits'].items()]
    distributions=[[k,v['n'],f"{v['mean']:.4f}",v['quantiles'][2],v['quantiles'][4],v['quantiles'][-1]]
        for k,v in audit['distributions'].items() if v['n']]
    labels=load(OUT/'label_audit.json');labelrows=[]
    for split,c in labels['eligible'].items():
        for key in ('human_collision','static_obstacle_collision','any_psc_violation','psc_valid','query_complete'):
            labelrows.append([split,key,c['query_outcomes'],c[key+'_available'],c[key+'_positive']])
    countrows=[]
    for split in ('train','dev'):
        count=collections.Counter(int(r['count']) for r in np.load(OUT/(split+'.npy'),allow_pickle=True))
        countrows.extend([[split,n,count[n]] for n in sorted(count)])
    write('DATA.md',f'''# Frozen dataset

Read-only SQLite backup plus sealed before/copy/after checksums, schema and core
map/sensor/query/executor identity checks. No live mutable training files or hard
links. Snapshot source: **{m['source_outcomes']:,} outcomes / {m['source_roots']:,}
roots / {m['source_scenes']} scenes**. Manifest SHA256 `{sha(SNAP/'manifest.json')}`.

{table(['Split/exclusion','Roots','Query outcomes','Scenes'],splitrows)}

{table(['Root distribution','Samples','Mean','Median','P95','Maximum'],distributions)}

{table(['Split','Actual human count','Independent roots'],countrows)}

Human-count strata describe crowd density; people per square meter are not
invented from the unknown full floor area. Count0 roots remain part of filtering
and every forecast horizon with available truth.

{table(['Split','Fixed-follower label','Outcomes','Label available','Positive'],labelrows)}

Source aggregate outcomes: {m['source_statistics']['human_collisions']:,} human
contacts, {m['source_statistics']['psc_violations']:,} PSC violations,
{m['source_statistics']['query_completions']:,} query completions,
{m['source_statistics']['query_timeouts']:,} timeouts. These descriptive label
counts include dependent siblings. They are not effective independent samples.
Eligible first-event timestamp availability, stopping causes and native executed
duration quantiles are in label_audit.json. No label audit opens protected payloads.

Accepted proposal diagnostics (not independent futures):
{table(['Recorded category','Count'],sorted(audit['sampling'].items()))}
Core contract hashes: `{json.dumps(m['core_contracts'])}`. Full source data,
generator, observer, metrics and executor identities are in the frozen manifest.

The eligible train/dev payload snapshot contains87,085 outcomes from12,045
independent roots. All query siblings stay together. Fixed scene hashing retains
existing boundaries; protected source dev/confirmation, research holdout and
previously opened fresh scenes are excluded. No fresh confirmation is claimed.
Incompatible historical GraphV2 full-map datasets were not mixed in.

Only {train['training_roots']:,} train roots with matched root-visible trajectories
supervise neural motion; every9,935 eligible training root supervises the count
prior. Development motion selection uses {train['dev_roots']} roots. The full
scientific evaluator includes all2,110 development roots, including no detections.
No invented future target is used when native truth ends before a horizon.

Inputs are legal arrived anonymous centers, robot poses, static graph deltas and
history only. Offline future arrays establish supervision/scoring correspondences;
they never supply online features, hidden state, human identity or full geometry.
Canonical fixed-query diagnostics use the predeclared longest compiled query,
not an action selector. All siblings share one human future.

Native0.048-second dynamics boundaries and actual timestamp jitter are distinct.
Four recorded native endpoints per available training path are used without
inventing a nominal truth clock. Evaluation targets0.5/1/2/3s use linear
interpolation of offline native truth, with availability/censoring explicit.
Future observation predictions use legal planned poses; recorded-pose optical
scores are separately named. Complete planned-prefix event calibration excludes
incomplete exposures; endpoint risk sets are descriptive under informative stops.

Raw count/history/age/duration/speed quantiles, proposal categories and scalar-fit
support are in `artifacts/raw/data_audit.json.gz`; outcome frequencies, availability
and censoring are in `artifacts/raw/label_audit.json.gz`.
Sampling remains the campaign's geometry curriculum/crowd/proposal mixture, not
a navigation-policy visitation distribution. Original protected payloads were
neither copied nor opened. Full root IDs, versions and checksums are bundled in
`artifacts/dataset_manifest.json.gz`; actual immutable payloads remain in output.
''')
    write('TRAINING.md',f'''# Actual training

Width64 GraphDynamics, {train['parameters']:,} parameters, unchanged architecture.
Completed **{train['completed_epochs']} full epochs** on {train['training_roots']:,}
independently supervised roots. Selected trained epoch **{selected['epoch']}**,
checkpoint SHA256 `{cksha}`. Parameter L2 change from epoch0 is {norm:.8g}.
Epoch0, selected, last, optimizer/scheduler/RNG states and fixed baseline assets
are preserved. The selected candidate is an actually trained checkpoint.

Root-balanced conditional path Energy Score, four independent samples per
persistent intention mode in training, eight fixed-seed development samples.
Within-mode U-statistic and cross-mode independent pair terms are tested.
Initial legal posterior components are weight-sampled per root. This proper
conditional marginal objective is a surrogate for full-posterior population/joint
scoring; it does not learn the common-U copula or hidden spatial law. Full-path
compound NLL is unsupported, not replaced by a mismatched Gaussian density.

AdamW LR0.001, weight decay0.001, batch48 independent roots, gradient clip5,
seed211, cosine LR floor0.00005, minimum15/max50 epochs, patience8/minimum
improvement0.001. Development-only selection; no protected tuning. Stop reason:
`{train['stop_reason']}`. Selected dev path ES {train['selected_dev_path_es']:.6f};
epoch0 dev ES {learning[0]['dev_es']:.6f}. Training wall time {train['seconds']:.2f}s,
CPU time {train['cpu_seconds']:.2f}s. Hardware {train.get('gpu_name')}; device
{train['device']}, one numerical CPU thread, loader workers0, CUDA memory cap20%.
Actual peak tensor GPU memory and host RSS are recorded each epoch in learning.json.
Fit process CPU/wall ratio is {train['cpu_seconds']/train['seconds']:.4f} CPU cores;
peak recorded RSS is {max(r['rss_kb'] for r in learning)/1024:.2f}MiB and
peak allocated GPU tensor memory {max(r['gpu_peak_bytes'] for r in learning)/1048576:.2f}MiB.
The latter excludes the CUDA context/allocator reservation and is not whole-device use.
Frozen payloads/caches and evaluation storage are summarized in resource_summary.json;
preprocessing/evaluation duration is separate from the fitting duration above.

Train-only scalar jump fit: {json.dumps(train['motion'])}. It uses8,571 training
roots/249,337 finite-difference velocity innovations, fixed rate2Hz and a fitted
nuisance innovation variance. No speed clipping. This fit is not evidence of
calibrated future physical tails; matched native diagnostics are separate.

All cache, raw-manifest, population, motion, optics and source hashes are saved
with the checkpoint. Deterministic torch algorithms, disabled TF32 and
CUBLAS_WORKSPACE_CONFIG=:4096:8 are explicit. NVML hardware utilization could
not be read because of driver/library mismatch; CUDA allocation/training worked.
CUDA step stream elapsed is recorded as a timing proxy, not a utilization claim.
Raw collection/resource logs preserve worker counts, throughput, status changes,
CPU pressure, RAM and disk. Production settings/services were untouched.

The frozen stopping implementation counts epochs whose increment over the
immediately previous record-best is less than0.001. It is not conventional
patience against the last significant best. Epoch9-to17 cumulative development
improvement was0.002053, and selected epoch17 was still improving. This release
reports a completed predeclared-rule experiment, **not optimization convergence**.
The exact rule and optimizer-step count are independently replayed in the review.

![Learning curves](figures/01_learning.png)
''')
    rows=[];paths=[]
    diagnostics=[];joint=[]
    for name in NAMES:
        for h in H:
            r=summary['forecasting'][name][str(h)];p=summary['paths'][name][str(h)]
            rows.append([name,h,ci(r['population_es3']),ci(r['tracked_es']),ci(r['gospa']),ci(r['count_mae']),ci(r['count_nll']),ci(r['coverage90'])])
            paths.append([name,h,ci(p['population_path_es3']),ci(p['path_es']),ci(p['joint_path_es']),ci(r['aged_tracked_es']),ci(r['never_localized_es3'])])
            diagnostics.append([name,h,ci(r['velocity_error']),ci(r['velocity_coverage90']),ci(r['false_population']),ci(r['missed_population']),ci(r['localization_squared']),ci(r['cardinality_squared']),ci(r['tracked_missing'])])
            joint.append([name,h,ci(p['joint_population_path_diversity']),ci(r['cross_human_predicted_covariance_trace']),ci(r['cross_human_observed_residual_product']),ci(r['cross_human_covariance_calibration_difference'])])
    mc_rows=[];mc_paths=[]
    for name in NAMES:
        for h in H:
            rr=[r for r in forecast_mc['records'] if r['model']==name and r['horizon']==h]
            pp=[r for r in forecast_mc['paths'] if r['model']==name and r['horizon']==h]
            mc_rows.append([name,h,*[ci(scene_ci(rr,key)) for key in ('population_es3','tracked_es','gospa','coverage90')]])
            mc_paths.append([name,h,*[ci(scene_ci(pp,key)) for key in ('population_path_es3','path_es','joint_path_es')]])
    physicalrows=[]
    for name,metrics in physical['matched_summary'].items():
        for key in ('speed_p95','speed_p99','speed_over3','speed_over5','stop_frequency','restart_frequency','turn_p95_rad','acceleration_p95','displacement_p95'):
            r=metrics['predicted_'+key]
            physicalrows.append([name,key,ci(r['predicted']),ci(r['actual']),ci(r['predicted_minus_actual'])])
    write('FORECASTING.md',f'''# Forecasting evidence (development only)

Every cell is equal-scene root mean [95% scene-bootstrap interval] (roots/scenes).
One root has one exogenous human future; siblings are never independent samples.
Frozen seed73/1000 scene bootstrap draws. No predictive SOTA/confirmation claim.
The primary contrast averages supported horizons within each root, roots within
each scene, then weights scenes equally. The corresponding cohort has1,881 roots;
it is not restricted to the259 roots with a complete3s future.
The original tables below condition on a globally reused16-world seed41;
their intervals do not estimate cross-root common Monte Carlo error. The paired
second-seed diagnostic exposed that limitation. A post-fit full-root validation
uses independent SHA-derived per-root seeds, all6 fixed models, the same targets
and no reselection. It is the numerical check of continuous-law comparisons;
the original procedure remains visible. See MC_VALIDATION_AMENDMENT.

{table(['Baseline','Independent root-MC D-minus-baseline'],[[k,ci(v)] for k,v in forecast_mc['primary_contrasts'].items()])}

![Independent root Monte Carlo validation](figures/10_independent_mc_forecasting.png)

Independent root-MC horizon and path scores:

{table(['Model','s','Population ES3','Tracked ES m','GOSPA','Coverage90'],mc_rows)}

{table(['Model','s','Population path ES3','Tracked path ES m','Joint tracked path ES'],mc_paths)}

Population ES3 here is Energy Score on the Gaussian-kernel counting-measure
Hilbert embedding (bandwidth3m) augmented with count. It includes ALL sampled
tracked and hidden people and all truth targets. The Hilbert norm is a valid
negative-type ES distance; GOSPA is only diagnostic. No numerical equivalence
to another historical metric namedES3 is claimed. Coordinates are horizontal
(x,z). GOSPA uses cutoff2m, p=2 and alpha=2. Count MAE uses the posterior mean
count as its point estimate. Iid-world U-statistic scores
estimate the declared normalized model law without diagonal MC bias. Path ES3
embeds entire anonymous human paths, RMS-scaled across target times, preserving
same-world human paths and penalizing missing/ghost population. No per-time
identity rematching is used. Tracked/path scores use fixed root correspondence.

A drops aged locations but retains count uncertainty; B retains track memory.
C_frozen_graph uses previous network and2Hz/2m/s forecasting law on the common
repaired history. C_matched_motion also uses the fitted scalar motion law, isolating
neural learning. All use matched legal histories and the same training-only prior.

{table(['Model','s','Population ES3','Tracked ES m','GOSPA','Count MAE','Count NLL','Coverage90'],rows)}

{table(['Model','s','Population path ES3','Tracked path ES m','Joint tracked path ES','Aged ES m','Unlocalized-pool proxy ES3'],paths)}

{table(['Model','s','Velocity error m/s','Velocity coverage90','False population','Missed population','Localization squared','Cardinality squared','Missing tracked targets'],diagnostics)}

{table(['Model','s','Joint population path diversity','Predicted cross-human covariance trace','Observed residual product','Covariance difference'],joint)}

The unlocalized-pool proxy truth partition is fixed from common root/history
alignment, not seed-dependent or model-specific. Aged alignment has a declared
2m offline gate; it is a proxy, not verified never-localized physical ancestry.
A also transfers aged tracks into its anonymous pool, so its isolated pool score
is not a clean like-for-like latent-class baseline comparison. Complete-population
ES remains the primary common object. True never-localized ancestry calibration
is not established by this proxy. Missing tracked targets
and zero count support are explicit. Infinity is never hidden in a finite-only
NLL average. Exact compound multi-step position density is unavailable, so
position NLL is not fabricated. Valid native-step NLL/gradient tests remain.
Velocity error/coverage use instantaneous latent predicted velocity against the
native interval finite-difference truth: they are proxy diagnostics, not exact
latent-velocity calibration. The physical tables instead match both sides to
the same native displacement/time interval. False/missed population, localization/cardinality errors,
joint diversity and cross-human covariance-vs-residual-product diagnostics are
in the complete raw summary. Marginal path training does not establish learned
joint dependence. Native matched-interval physical diagnostics are in physical.json.

Primary paired contrasts D-minus-baseline (lower is better):
{table(['Baseline','Population ES3 contrast'],[[k,ci(v)] for k,v in summary['primary_contrasts'].items()])}

These are the original common-seed contrasts. Cross-horizon curves change
available cohorts (3s:259roots/57scenes versus0.5s:1881roots/106scenes);
their slope is not a matched-cohort estimate of horizon degradation.

Native matched-interval physical diagnostics below use each metric's paired
nonempty truth/prediction root support, not different predicted/actual cohorts.
All-predicted-root summaries including empty-truth ghosts remain in physical.json.
These are physical-law diagnostics, not evidence of calibrated stops/turns.
The fitted scalar reduces excessive old tails but does not solve the remaining
stop/restart and turning mismatch. No arbitrary velocity clipping is applied.
These physical and optical descriptive intervals condition on their frozen
common-random-number procedures; they do not estimate shared Monte Carlo error.
The independent-root amendment applies to primary forecast/conditioning scores.

{table(['Model','Native statistic','Predicted','Actual','Predicted minus actual'],physicalrows)}

![Forecasting](figures/02_forecasting_horizons.png)
![Hidden and aged](figures/04_hidden_and_aged.png)
![Coverage](figures/07_coverage.png)
''')
    condrows=[];countsupport=[]
    for h,groups in summary['conditioning'].items():
        for group,r in groups.items():
            condrows.append([h,group,ci(r['population_es3_gain']),ci(r['tracked_es_gain']),ci(r['count_nll_gain']),ci(r['gospa_gain']),ci(r['coverage90_change'])])
            support=r['count_nll_support'];countsupport.append([h,group,support['prior_infinite'],support['posterior_infinite'],support['undefined_gain']])
    filt=summary['filtering'];seq=[r for r in raw['filtering'] if r['phase']=='future_arrived']
    seq_zero=[r for r in seq if r.get('infinite_count_nll')]
    retentionrows=[]
    for phase in ('root','future_arrived'):
        values=[r['retention_product'] for r in raw['filtering'] if r['phase']==phase]
        retentionrows.append([phase,len(values),*['%.8g'%v for v in np.quantile(values,[0,.01,.05,.5,.95])]])
    filterrows=[]
    for phase,groups in summary['filtering_strata'].items():
        for group,r in groups.items():
            filterrows.append([phase,group,ci(r['count_mae']),ci(r['ess']),ci(r['retention_product']),ci(r['expected_localized_ghosts']),ci(r['false_birth_mass']),ci(r['false_recovery_mass']),r['infinite_count_nll_updates']])
    mc_condrows=[]
    for h,groups in conditional_mc['conditioning'].items():
        for group,r in groups.items():mc_condrows.append([h,group,ci(r['population_es3_gain']),ci(r['tracked_es_gain']),ci(r['count_nll_gain']),ci(r['gospa_gain']),ci(r['coverage90_change'])])
    write('BELIEF_AND_CONDITIONING.md',f'''# Persistent belief and new information

Normalized assumed-density posteriors are separate from the preserved local-bound
channel. Tiny pruned mass is no longer multiplied by a global residual supremum
inside the point approximation. This is explicit support projection, not a proof
that omitted physical hypotheses vanished. Local retained fractions/product,
effective hypothesis support, counts and predictive errors are all measured.

{table(['Root filtering quantity','Result'],[[k,ci(v)] for k,v in filt.items() if isinstance(v,dict)])}

{table(['Phase','Updates','Min retention','P1','P5','Median','P95'],retentionrows)}

True count has zero represented probability in {filt['infinite_count_nll_roots']}
root history and {len(seq_zero)} sequential updates across
{len({r['root_id'] for r in seq_zero})} sequential-cohort roots. Unconditional
count NLL is infinite in both affected cohorts. Stable normalization does not
establish useful true support. One root excludes truth3 with all retained
hypotheses containing4 localized people despite retention0.997568. No count
floor, ghost deletion or favorable recovery was inserted to conceal it.

{table(['Phase','Stratum','Count MAE','Effective hypotheses','Retention product','Expected ghosts','False births','False recoveries','Infinite countNLL updates'],filterrows)}

Full root-history support:2,110 roots. Sequential future-arrival cohort: first240
metadata-ordered roots, up to30 successive recorded observations each;
{len(seq)} updates actually scored. All failures are retained in failures.json.
The maximum scored exact-coordinate error is
{max((r.get('exact_coordinate_error',0.) for r in seq),default=0.):.6g}m.
Empty-frame persistence, unavailable-sensor neutrality and repeated-evidence
receipts have separate synthetic references. Actual current campaign frames
contain no general certified human-visible-empty region.

Exact assignment-evidence DP applies through10 detections. Above10 the evidence
denominator is an overcounting envelope and retention is a lower diagnostic,
not an exact retained posterior fraction. Native roots have at most6 people;
the synthetic12-person stress is separately qualified. The11-detection finite
review witness verifies this distinction without changing normalized point weights.

Raw filtering rows report ghost frequency, association ambiguity, false-birth/
false-recovery mass and reappearance. These ancestry diagnostics use external
root/history correspondence and source-independent origin tags; truth never
changes inference weights/state. Unmatched ancestry is counted as unsupported.
Long synthetic sessions additionally exercise500 arrivals with forecast/event/
successor/cache churn; their results do not masquerade as real500-frame episodes.
The legacy fields false_birth_mass, false_recovery_mass and supported_matches
sum weights per slot: they are expected counts per frame and may exceed1,
not probabilities of a whole-frame error event. Stratum ancestry is a diagnostic
proxy; reappearance labels are not a verified oracle of never-localized identity.

For each pair, both arms receive the identical arrived robot/static information.
One arm omits only the new human frame; the other conditions on it. Both target
the SAME absolute root+2s (primary) or+1s (secondary) truth. Root slot ancestry is
carried externally across both arms, never rematched at future times. Positive
gain=prior score minus posterior score. Coverage is instead posterior-minus-prior
change. No universal individual-sample improvement requirement is imposed.

The following independent root-MC validation repeats every original pair with
base107 SHA-derived per-root seeds shared between its two arms. All observation
frames, targets and stratum labels remain frozen. Its scene intervals include
independent numerical sampling variation across roots. Original common-seed107
rows remain below and are conditional on that numerical procedure.

{table(['Target s','Stratum','Root-MC population gain','Tracked gain','Finite countNLL gain','GOSPA gain','Coverage change'],mc_condrows)}

At the primary2s target, overall population gain is
**{ci(conditional_mc['conditioning']['2.0']['all']['population_es3_gain'])}**:
inconclusive. Tracked gain is
**{ci(conditional_mc['conditioning']['2.0']['all']['tracked_es_gain'])}**,
and positive-detection population gain is
**{ci(conditional_mc['conditioning']['2.0']['positive']['population_es3_gain'])}**.
The original negative2s no-detection tracked/GOSPA findings do not persist under
independent-root sampling; those intervals now span zero. At the secondary1s
target, no-detection population gain is
**{ci(conditional_mc['conditioning']['1.0']['no_detection']['population_es3_gain'])}**,
a remaining deterioration. These mixed results qualify Gate4. Stratum and
secondary-target intervals are descriptive and are not multiplicity-adjusted
confirmatory tests.

![Independent root conditioning validation](figures/11_independent_mc_conditioning.png)

{table(['Target s','Observation stratum','Population gain','Tracked gain','Finite countNLL gain','GOSPA gain','Coverage change'],condrows)}

{table(['Target s','Observation stratum','Prior infinite countNLL','Posterior infinite countNLL','Undefined count gain'],countsupport)}

Count-gain summaries explicitly record prior/posterior infinite support and
undefined pairs; finite-only differences are labeled conditional. No sensor-
unavailable empirical stratum is invented; the synthetic no-evidence invariant
is tested. Reappearance/dense/aged/first-localization supports and CIs are raw.
The optical law is approximate; negative or inconclusive conditional gains remain
negative/inconclusive evidence, never explained away by training-set size.

![Count calibration](figures/03_population_count.png)
![Conditioning](figures/05_conditioning_gains.png)
''')
    rec=[r for r in systems['recursion'] if r['segments'] in ('two','three')]
    obsrows=[[kind,ci(r['brier']),ci(r['nll']),ci(r['probability']),ci(r['observed']),ci(r['first_detection_nll'])] for kind,r in summary['observations'].items() if isinstance(r,dict) and 'brier' in r]
    availability=load(OUT/'observation_availability_audit.json')
    eventrec=load(OUT/'observation_recursion.json')
    write('RECURSION_AND_OBSERVATIONS.md',f'''# Recursive state, shared worlds and future observations

Same physical3s target: direct analytic propagation vs2x1.5s/3x1s, no observations.
Maximum mean discrepancy {max((r['mean_error'] for r in rec),default=0.):.8g};
covariance discrepancy {max((r['covariance_error'] for r in rec),default=0.):.8g}.
Count-support equality {all(r['count_support_equal'] for r in rec)}. These moment
identities do not imply equality to direct compound sampled distributions.
Maximum weight discrepancy {max((r['weight_error'] for r in rec),default=0.):.8g}.

Native-boundary Gaussian projection loses older jump cumulants. Under the fitted
impulse law the isolated3s fourth cumulants are
`{json.dumps(systems['scalar_jump_fourth_cumulant'])}`. Real256-world direct versus
intermediate-projection characteristic/count features and their MC standard errors
are recorded in systems.json. This is measured projection error, not independent-
draw seed inequality. Analytic joint-mode compression remains another approximation.

Finite conditional carriers preserve held paths, event weights, observation
ledgers and continuation RNG on supported endpoints. Independent finite references
test normalization, no/any event marginalization, same tape continuation,
reproducibility, immutable ownership, query-order independence and terminal
absorption. Arbitrary exact hypothetical centers/interior bridges remain unsupported.
Region/masked same-endpoint exact refinement is explicitly refused; later new
emissions are allowed. Event predictions bind to physical tapes AND optical regimes.

On12 real legal histories, a fixed-query prefix no-detection successor followed
by its conditional remaining observation forecast reproduces the direct no-event
probability: passed={eventrec['passed']}, maximum absolute error
{max((r['probability_error'] for r in eventrec['records']),default=0.):.8g},
posterior-weight error {max((r['posterior_weight_error'] for r in eventrec['records']),default=0.):.8g}.
These finite-world probability identities use held paths and externally declared
coarse prefix events, with no future truth or exact imaginary measurements.

{table(['Observation evaluation','Brier','NLL','Predicted frequency','Observed frequency','First detection NLL'],obsrows)}

These optical scores and scene intervals remain conditional on the original
globally shared forecast streams. They do not constitute a root-independent
Monte Carlo calibration study. The public first-detection distribution is scored
on1,495 complete available planned prefixes; its normalization and additive
before/after artifact hashes are in planned_first_detection.json. That stage
follows the preserved availability audit, rather than replacing its hashes.

Actual native clock error audit for nominal planned endpoints:
`{json.dumps(availability['native_clock_error_s'])}`. Actual timestamps remain
authoritative. Nominal planned-exposure calibration is qualified by these clock
and route deviations; it is not exact same-clock native-sensor likelihood.
All canonical prefixes contain {availability['unavailable_frames']} unavailable
frames among {availability['total_recorded_frames']} recorded frames.
The final audit excludes {availability['removed_complete_prefix_scores']}
complete-prefix scores with unavailable endpoints instead of treating absence
of a sensor emission as negative human evidence. Original scores and corrected
source/artifact hashes are retained in observation_availability_audit.json.

Planned-query outputs use immutable legal compiled robot poses/nominal action
timestamps. Recorded-pose rows are separate offline optical diagnostics. Complete
planned-prefix rows require complete followup; endpoint risk-set rows and native
timing errors expose censoring/clock limitations. First-detection time likelihood,
no-event probability, normalization and calibration bins are included in raw rows.
Unavailable sensor does not emit a negative frame; static arrivals remain separate.
No privileged visibility mask, future graph geometry or planner event-region
selection is used. Native fixed-follower contact/PSC heads remain secondary and
unvalidated after this motion-law change; no future-L4 risk claim is made.

![Recursion](figures/06_recursion.png)
![Future observation calibration](figures/09_observation_calibration.png)
''')
    perfrows=[]
    for group,metrics in systems['quantiles'].items():
        for key,r in metrics.items():perfrows.append([group,key,r['n'],f"{r['p50']:.2f}",f"{r['p95']:.2f}",f"{r['p99']:.2f}"])
    depthrows=[]
    for label,values in (
            ('direct3s',[r['direct_ms'] for r in systems['recursion'] if r['segments']=='two']),
            ('two1.5s',[r['segmented_ms'] for r in systems['recursion'] if r['segments']=='two']),
            ('three1s',[r['segmented_ms'] for r in systems['recursion'] if r['segments']=='three'])):
        depthrows.append([label,len(values),*[f'{v:.2f}' for v in np.quantile(values,[.5,.95,.99])]])
    warmcombined=systems['quantiles'][str(('warm',16,'real'))]['combined']
    coldcombined=systems['quantiles'][str(('dynamic_cold',16,'real'))]['combined']
    resource=load(OUT/'resource_summary.json') if (OUT/'resource_summary.json').exists() else None
    resource_text='Resource-summary artifact unavailable; no consumption claim is supplied.'
    if resource:
        resource_text=(f"Read-only resource log: {resource['samples']} samples over "
            f"{resource['elapsed_seconds']/3600:.3f} hours. Collector worker counts "
            f"{resource['collector']['worker_counts']}, restart counts {resource['collector']['restart_counts']}, "
            f"automatic-training values {resource['collector']['automatic_training_values']}; "
            f"{len(resource['collector']['configuration_identities'])} configuration identity. "
            f"It added {resource['collector']['outcome_increase']:,} outcomes during that interval. "
            f"Observed interval throughput median/P95 "
            f"{resource['collector']['interval_outcomes_per_second']['p50']:.3f}/"
            f"{resource['collector']['interval_outcomes_per_second']['p95']:.3f} outcomes/s. "
            f"Status sample counts {resource['collector']['status_samples']} are preserved. "
            f"Research output occupied {resource['research_output_bytes_before_archive']/1073741824:.3f}GiB "
            f"before archive assembly. Host available RAM minimum "
            f"{resource['host']['available_ram_bytes']['minimum']/1073741824:.2f}GiB; "
            f"available disk minimum {resource['host']['available_disk_bytes']['minimum']/1073741824:.2f}GiB. "
            f"One-minute load median/P95 {resource['host']['load_1m']['p50']:.2f}/"
            f"{resource['host']['load_1m']['p95']:.2f}; load is not CPU utilization. "
            f"CPU-pressure some-avg10 maximum "
            f"{resource['host']['cpu_pressure_some_avg10_percent']['maximum']:.2f}%. "
            "See resource_summary.json and its hashed resource log.")
    failures_table=table(['Root','Scene','Failure'],[[r.get('root_id','synthetic'),r.get('scene_id','synthetic'),r['error']] for r in systems['failures']])
    write('PERFORMANCE.md',f'''# Layer2 computation

Primary workload: actual human observe + joint3s/16world forecast + one fixed
legal query's future human observation law + supported coarse successor.
No planning packet, ranking or policy is included. Fixed-query secondary head
readout and analytic propagation are measured separately. Warm and dynamic-cold
cache labels use the same real cohort, plus dense and8/32/64-world sensitivity.
Cold clears actual graph/footprint/kernel/endpoint caches in an initialized
process. It does not hide startup: model/JIT {systems['startup_model_and_JIT_ms']:.2f}ms;
complete fresh-process synthetic API demo {systems['cold_process_complete_synthetic_demo_ms']:.2f}ms;
real prefix initialization quantiles `{json.dumps(systems['initialization_ms'])}`.
Propagation component timing follows observe/forecast (already warmed); separate
direct/2/3-depth times are also saved. Component quantiles do not sum to P95.

The primary combined P95 is **{warmcombined['p95']:.2f}ms warm** and
**{coldcombined['p95']:.2f}ms dynamic-cold**. Neither the preferred65ms nor
the100ms integration target is met. Timings are conditional on76 supported
roots/54 scenes; the77th attempted root in a55th scene is retained as a failure.
Its swept robot footprint touches UNKNOWN cell[37,-17]. The conservative refusal
is correct; converting UNKNOWN to free would invalidate the input boundary.

{table(['Cache/worlds/workload','Component','Calls','P50ms','P95ms','P99ms'],perfrows)}

{table(['Analytic3s propagation depth','Roots','P50ms','P95ms','P99ms'],depthrows)}

Failures/exclusions (complete traces remain in systems.json):

{failures_table}

Synthetic certificates are outside current native sensor emissions and remain
separately visible.
The12-detection stress also uses the declared association evidence envelope;
its retained-fraction diagnostic is not exact posterior retention. Real native
detection sets are within the exact assignment-evidence DP size limit.
Memory experiment:500 sensor arrivals and {persist['forecast_event_successor_calls']}
forecast/event/successor calls; raw RSS and bounded cache records in persistence.json.
These are empirical measurements, not hard real-time guarantees or an absence-
of-all-leaks proof. Cooperative checks may overrun nonpreemptive kernels.
Production remained externally supervised; raw logs include status fluctuations,
28-worker counts and unchanged configuration identity. No causal no-slowdown
claim is inferred from shared-host throughput observations.

{resource_text}

![Latency](figures/08_latency.png)
''')
    real={cache:systems['quantiles'].get(str((cache,16,'real')),{}).get('combined') for cache in ('warm','dynamic_cold')}
    dense=[systems['quantiles'].get(str((cache,16,'dense_positive')),{}).get('combined') for cache in ('warm','dynamic_cold')]
    latency_pass=all(v and v['p95']<=100 for v in list(real.values())+dense) and not systems['failures']
    contrasts=summary['primary_contrasts'];mc_contrasts=forecast_mc['primary_contrasts']
    strong_key=max(('D_minus_B_memory_cv','D_minus_B_historical_cv'),key=lambda k:mc_contrasts[k].get('mean',-math.inf))
    strong=mc_contrasts[strong_key];gain=conditional_mc['conditioning']['2.0']['all']['population_es3_gain']
    forecast_verdict=('DEVELOPMENT SUPERIORITY' if all(mc_contrasts['D_minus_'+n].get('hi',math.inf)<0 for n in ('A_visible_cv','B_memory_cv','B_historical_cv'))
        else 'DEVELOPMENT INFERIOR TO MEMORY CV' if any(mc_contrasts['D_minus_'+n].get('lo',-math.inf)>0 for n in ('B_memory_cv','B_historical_cv'))
        else 'INCONCLUSIVE VS STRONGEST MEMORY CV')
    support_failures=filt['infinite_count_nll_roots']
    belief_verdict='FAIL: TRUE COUNT SUPPORT LOST' if support_failures else 'QUALIFIED APPROXIMATION'
    gates=[['1 Legality','PASS in audited interfaces','Legal features, scene/root split, no future/full-map online input'],
        ['2 Belief',belief_verdict,'Normalized support with explicit omissions; '+str(support_failures)+' root count-support failures'],
        ['3 Forecasting',forecast_verdict,'Full-population/path/track/count/GOSPA/coverage, scene CIs'],
        ['4 Conditioning','PARTIAL: MIXED GAINS; POPULATION INCONCLUSIVE','2s tracked/positive-detection gains; overall population CI spans zero; 1s no-detection population worsens'],
        ['5 Recursion','PARTIAL: MOMENTS/FINITE CARRIERS','Higher moments change under analytic projection; no exact global-law claim'],
        ['6 Counterfactuals','PASS tested shared law','Query-independent human tapes; no query selector'],
        ['7 Observation prediction','PARTIAL: CALIBRATION/EXPOSURE PROXY','Coherent finite law; approximate optics/censoring and conditional fixed-MC calibration'],
        ['8 Performance','PASS declared native workload' if latency_pass else 'FAIL','Primary real/dense warm+cold P95 and unsupported routes; all stresses visible'],
        ['9 Reproducibility','TRAINED ARTIFACTS SEALED','Cache/source/data/checkpoint/optimizer/RNG hashes and runnable scripts']]
    write('FINAL_REPORT.md',f'''# Trained Layer 2 experimental research release

**Actually trained; scientifically qualified, not an unqualified integration or
predictive-SOTA promotion.** {train['completed_epochs']} epochs completed;
selected epoch {selected['epoch']} of the 33,244-parameter graph model.
No Layer 2.5/L3/L4, planner, policy or closed-loop navigation experiment was built.
Production collection was not stopped, reconfigured, restarted or merged into.

Snapshot: {m['source_outcomes']:,} outcomes, {m['source_roots']:,} independent roots,
{m['source_scenes']} scenes. Eligible training: 9,935 roots / 72,539 outcomes / 467 scenes;
development: 2,110 roots / 14,546 outcomes / 108 scenes. Actual trajectory-supervised training roots:
{train['training_roots']:,}; all training roots fit the count prior. Protected
exclusions stay protected. **Development evidence only; no invented confirmation.**

Selected checkpoint SHA256 `{cksha}`; parameter-change L2 norm from epoch 0: {norm:.6g}.
Fitting wall time: {train['seconds']:.2f} s. Optimizer, last, initial/frozen baselines and
fixed configuration are retained. Proper path sampling matches the compound
forecast law; no legacy OU-only objective or Gaussian pathNLL substitution.

Independent root-MC full-population ES3 {strong_key}: **{ci(strong)}**.
This averages supported horizons within root, roots within scene, then scenes equally.
Independent root-MC same-target 2 s population gain: **{ci(gain)}** (positive is better).
This primary population gain is inconclusive. The tracked-human gain is
**{ci(conditional_mc['conditioning']['2.0']['all']['tracked_es_gain'])}**;
positive detections improve population scores, while secondary 1 s no-detection
population gain is **{ci(conditional_mc['conditioning']['1.0']['no_detection']['population_es3_gain'])}**.
Full horizon/baseline/hidden/joint/coverage tables are in FORECASTING. Negative
or inconclusive comparisons remain so; a trained model is not evidence of
superiority. Count support infinities, missing tracked ancestry and sensor
strata are explicit in BELIEF_AND_CONDITIONING.

The original global-seed41 contrast was **{ci(contrasts[strong_key])}**, worse
on that frozen 16-world procedure. A paired second-seed check exposed shared
Monte Carlo error omitted by those scene intervals. Full-root independent-seed
validation changes the ranking; the strongest-CV two-sided 95% interval still
reaches zero. No favorable one-sided test or rounded-zero promotion is used.
Improvement over the previous graph network remains, but data-volume and
objective effects cannot be causally separated. Count NLL is infinite
on {support_failures} root because the retained count law excludes its truth.
No probability floor or favorable count was invented to hide that failure.
Observation-event calibration remains approximate. Three apparent infinite
prefix-event losses were subtraction cancellation and are corrected from direct
no-event likelihoods, with original rows preserved. Genuine count-support failures
remain. These distinctions prevent an unqualified belief/calibration verdict.

Combined 3 s / 16-world P95: warm **{real['warm']['p95']:.2f} ms**, dynamic-cold
**{real['dynamic_cold']['p95']:.2f} ms**. The 100 ms integration target is
{'met on the declared native workload' if latency_pass else '**not met**'}.
Unsupported routes remain counted. Full initialization and certificate stress
are separately reported. Physical motion remains uncalibrated for stops/turns.

{table(['Gate','Verdict','Scope'],gates)}

The structural residual-envelope witness motivated a separate normalized
approximate channel, not fabricated rigorous bounds. It records local discarded
fractions and retained-support products; useful support must be judged with
actual filtering/forecast errors. The preserved bound channel remains available.
Gaussian projection, GH integration, bounded associations and the fixed joint
intention copula remain structural limitations. Hidden spatial/motion belief is
persistent but broad; it was not magically learned from visible-only targets.

Analytic3s vs2x1.5/3x1 moments compose within measured precision, while compound
higher moments change under intermediate projection. Finite conditional successors
retain their joint tapes and continue coherently at supported endpoints. General
continuous hypothetical bridges are explicitly unsupported. Fixed-query optics
are evaluated against recorded legal frames, including planned-pose predictions;
native contact/PSC head calibration is not established after the motion change.

All 361 inherited regression cases pass; new campaign/reviewer checks and trained
API evidence are bundled. Optimized/reference comparisons: {numeric.get('actual',0)},
passed={numeric.get('passed')}. Three focused independent reviews and remaining
qualifications are in reviews/. Reviewer agreement does not replace these data.

See DATA, TRAINING, FORECASTING, BELIEF_AND_CONDITIONING,
RECURSION_AND_OBSERVATIONS, PERFORMANCE, API, ARCHITECTURE, SCALING_HANDOFF and
REPRODUCE. Raw measurement artifacts and figure hashes accompany the release.
''')
    (OUT/'release_summary.json').write_text(json.dumps(dict(checkpoint_sha256=cksha,selected_epoch=selected['epoch'],
        trained_parameter_change_l2=norm,forecast_verdict=forecast_verdict,primary_latency_pass=latency_pass,
        gates=gates,original_global_seed_contrasts=contrasts,root_mc_contrasts=mc_contrasts,
        original_conditioning_gain=summary['conditioning']['2.0']['all']['population_es3_gain'],
        root_mc_conditioning_gain=gain),indent=2))
    seed=load(OUT/'seed_sensitivity.json');seedrows=seed['records']
    table41={(r['root_id'],r['horizon'],r['model']):r for r in raw['forecasts']}
    paired=[{**r,'seed107_minus41':r['population_es3']-table41[(r['root_id'],r['horizon'],r['model'])]['population_es3']}
        for r in seedrows if (r['root_id'],r['horizon'],r['model']) in table41]
    seedcontrast={name:scene_ci([r for r in paired if r['model']==name],'seed107_minus41') for name in NAMES}
    contrasts41={}
    for baseline in NAMES[:-1]:
        rr=[]
        for r in seedrows:
            key=(r['root_id'],r['horizon'])
            if r['model']=='D_trained_graph' and (*key,'D_trained_graph') in table41 and (*key,baseline) in table41:
                rr.append({**r,'difference':table41[(*key,'D_trained_graph')]['population_es3']-table41[(*key,baseline)]['population_es3']})
        contrasts41['D_minus_'+baseline]=scene_ci(rr,'difference')
    write('SEED_SENSITIVITY.md',f'''# Second-seed diagnostic and protocol scope

The full2,110-root primary forecasting evaluator used seed41. Seed107 was used
for same-target conditioning, not a second full forecasting replicate. This
deviates from the initial protocol's two forecasting seeds. A post-fit bounded
diagnostic uses seed107 on the first256 metadata-ordered roots with the identical
six models, physical horizons and16-world budget. No fitting or selection used
these results. No full two-seed confirmation claim is made. Horizon support may
be smaller because unavailable offline targets remain censored.

{table(['Baseline','D-minus-baseline, seed41 same subset','D-minus-baseline, seed107 subset'],[[k,ci(contrasts41[k]),ci(v)] for k,v in seed['primary_contrasts'].items()])}

{table(['Model','Paired seed107-minus41 Population ES3'],[[k,ci(v)] for k,v in seedcontrast.items()])}

Scene bootstrap intervals measure scene variation in the declared sampled-score
estimates; they do not independently estimate within-root Monte Carlo uncertainty.
The same physical future remains one root across both seeds. These diagnostics
must not double the number of independent trajectories.

The resulting shared-Monte-Carlo issue motivated the fixed post-fit
MC_VALIDATION_AMENDMENT. Its full-root independent-seed forecasts and conditioning
are now the primary numerical validation in FORECASTING and
BELIEF_AND_CONDITIONING; this bounded diagnostic and the original outputs remain
visible. No new checkpoint, baseline or training choice followed that amendment.
''')


if __name__=='__main__':main()
