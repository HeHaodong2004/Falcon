"""Render measured structural-release diagnostics without fitting any model.

Inputs are sealed benchmark/validation JSON, not collection or truth payloads.
Uses one CPU thread. Writes new artifacts only; refuses to overwrite reports.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import collections
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
METHODS = ('old_reference', 'structural_bound_unoptimized', 'structural_bound')
LABELS = {'old_reference': 'Preserved repair',
          'structural_bound_unoptimized': 'Structural reference kernels',
          'structural_bound': 'Structural optimized'}
COLORS = {'old_reference': '#737373', 'structural_bound_unoptimized': '#e69f00',
          'structural_bound': '#0072b2'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n': 0}
    return dict(zip(('p50', 'p95', 'p99'), map(float, np.percentile(values, [50, 95, 99]))),
                maximum=float(np.max(values)), n=len(values))


def summary(records):
    return {'calls': len(records), 'roots': len({r['root_id'] for r in records if r['root_id']}),
            'statuses': dict(collections.Counter(r['status'] for r in records)),
            'timings_ms': {k: quantiles([r['timings_ms'][k] for r in records if k in r['timings_ms']])
                           for k in sorted({k for r in records for k in r['timings_ms']})}}


def choose(records, **conditions):
    return [r for r in records if all(r.get(k) == v for k, v in conditions.items())]


def group(records, fields):
    groups = collections.defaultdict(list)
    for row in records:
        groups['/'.join(str(row.get(k, 'NA')) for k in fields)].append(row)
    return {key: summary(rows) for key, rows in sorted(groups.items())}


def table_line(label, records, component='outer_wall'):
    q = summary(records)['timings_ms'][component]
    return '| {} | {} | {:.2f} | {:.2f} | {:.2f} |'.format(label, q['n'], q['p50'], q['p95'], q['p99'])


def create_report(real, batches, validation, provenance, destination):
    records = real['records']; rr = choose(records, kind='real_dev')
    optimized = choose(rr, method='structural_bound')
    warm = choose(optimized, cache_mode='warm'); cold = choose(optimized, cache_mode='fully_cold')
    old_warm = choose(rr, method='old_reference', cache_mode='warm')
    old_cold = choose(rr, method='old_reference', cache_mode='fully_cold')
    wp = summary(warm)['timings_ms']['outer_wall']['p95']
    cp = summary(cold)['timings_ms']['outer_wall']['p95']
    speed_warm = 100 * (1-wp/summary(old_warm)['timings_ms']['outer_wall']['p95'])
    speed_cold = 100 * (1-cp/summary(old_cold)['timings_ms']['outer_wall']['p95'])
    dense = choose(records, kind='synthetic', method='structural_bound', explicit_certificate=True)
    parity = real['same_law_parity'] + batches['same_law_parity']
    repeated = [x for x in real['repeated_planner'] if x['method'] == 'structural_bound']
    aux = {'schema': 'structural_measured_analysis_1', 'input_provenance': provenance,
           'runtime_revision': real['metadata_before']['git_revision'],
           'real_summary': group(rr, ('method', 'cache_mode')),
           'batch_summary': group(batches['records'], ('method', 'batch_size', 'cache_mode')),
           'dense_summary': group(choose(records, kind='synthetic'),
                                  ('method', 'synthetic_tracked', 'explicit_certificate', 'cache_mode')),
           'hypothesis_range': [min(r['hypotheses'] for r in optimized), max(r['hypotheses'] for r in optimized)],
           'retained_mass_range': [min(r['retained_mass'] for r in optimized), max(r['retained_mass'] for r in optimized)],
           'retained_mass_root_counts': {str(t): len({r['root_id'] for r in warm if r['retained_mass'] < t})
                                         for t in (.01, .1, .5, .9)},
           'same_law': {'comparisons': len(parity), 'all_passed': all(x['passed'] for x in parity),
                        'maximum_absolute_error': max(x['maximum_absolute_numeric_error'] for x in parity)},
           'gates': {'warm_65ms': wp <= 65, 'warm_100ms': wp <= 100, 'cold_100ms': cp <= 100,
                     'dense_warm_100ms': all(summary(choose(dense, synthetic_tracked=n, cache_mode='warm'))['timings_ms']['outer_wall']['p95'] <= 100 for n in (6,12)),
                     'full_continuous_successor_closure': False, 'global_projection_bound': False},
           'decision': 'REVISE — STRUCTURAL REVISE',
           'statistical_fitting': False, 'untouched_predictive_evaluation': False}
    (destination / 'analysis_release_v2.json').write_text(json.dumps(aux, indent=2, allow_nan=False)+'\n')
    lines = ['# Complete online performance', '',
        'The final runtime is frozen at `{}`. These are full public-invocation wall times, including observation update, shared forecast, all native endpoints, graph support, observation branches and successor creation. **No latency gate is inferred from a neural forward pass.**'.format(aux['runtime_revision']), '',
        '## Fixed workload and provenance', '',
        'The primary cohort has {} original development roots in {} scenes. Horizon is 3 s, shared worlds 16, candidate rows 24, refined candidates at most four. Only legal history and query payloads are read. All methods start from isolated copies of the same preserved-repair legal prefix state; this is an operator benchmark, not a new predictive evaluation. The same-law structural kernel ablation preserves probability semantics; the preserved repair also differs in residual and successor semantics.'.format(len({r['root_id'] for r in rr}),len({r['scene_id'] for r in rr})), '',
        'The candidate sweep uses the same fixed 12 difficult development roots at each of 1/4/8/16/24 rows, selected from legal graph size, track count and history length before the final run. Available distinct routes are repeated when needed to match the reference workload; this measures computational scaling, not route-diversity quality.', '',
        'CPU only; one research thread per job, nice 19, idle I/O. At most two numerical jobs ran concurrently on the shared host. JIT/checkpoint loading and legal-prefix initialization are reported separately. Cold clears dynamic graph results, pose footprints, Sobol buffers and endpoint templates in an initialized process. It is not process startup. Source and frozen checkpoint hashes are verified before/after. Repetitions are correlated; P95/P99 are empirical workload quantiles, not independent-scene confidence guarantees.', '',
        '## Primary full-call latency', '',
        '| Implementation / cache | Calls | P50 ms | P95 ms | P99 ms |',
        '|---|---:|---:|---:|---:|']
    for method in METHODS:
        for cache in ('warm','fully_cold','graph_revalidation'):
            lines.append(table_line(LABELS[method]+' / '+cache, choose(rr,method=method,cache_mode=cache)))
    lines += ['', 'Final optimized warm P95 is {:.2f} ms; cold P95 is {:.2f} ms. Against the contemporaneously measured preserved repair, the respective P95 changes are {:.1f}% and {:.1f}% lower. The 65 ms warm target is **{}**; the 100 ms cold integration gate is **{}**. Scientific readiness remains blocked independently by incomplete continuous refinement and missing global projection bounds.'.format(wp,cp,speed_warm,speed_cold,'PASS' if wp<=65 else 'FAIL','PASS' if cp<=100 else 'FAIL'), '',
        '## Component measurements', '',
        'Component quantiles do not add to the complete-call quantile. Graph support and branch processing are nested within query evaluation.', '',
        '| Component / cache | Calls | P50 ms | P95 ms | P99 ms |', '|---|---:|---:|---:|---:|']
    for cache in ('warm','fully_cold'):
        for component in ('observe','forecast','queries','graph_support','branches'):
            lines.append(table_line(component+' / '+cache,choose(optimized,cache_mode=cache),component))
    lines += ['', '## Candidate batching on matched difficult inputs', '',
        '| Rows / cache | Calls | P50 ms | P95 ms | P99 ms |','|---|---:|---:|---:|---:|']
    for q in (1,4,8,16,24):
        for cache in ('warm','fully_cold'):
            lines.append(table_line(str(q)+' / '+cache,choose(batches['records'],method='structural_bound',batch_size=q,cache_mode=cache)))
    lines += ['', 'The forecast is shared once per call; increasing candidate count never generates independent human futures per query. Successors remain included. This sweep is a separate 12-root cohort and must not be compared with the 60-root Q24 figure as if the inputs were identical.', '',
        '## Dense stress cases', '',
        'The unchanged preserved stress fixture has 3,721 legal graph cells, 6 or 12 tracked people, hidden count support 0/2/4, 24 distinct queries, 16 worlds and horizon 3 s. Positive observations and certified-empty observations are both retained. These are synthetic stress cases, not assertions about native empty-space certificates. Warm stress quantiles use 20 calls per case; cold/revalidation only two and are descriptive.', '',
        '| People / evidence / implementation / cache | Calls | P50 ms | P95 ms | P99 ms |', '|---|---:|---:|---:|---:|']
    for n in (6,12):
        for cert in (False,True):
            for method in ('old_reference','structural_bound'):
                for cache in ('warm','fully_cold'):
                    rows=choose(records,kind='synthetic',method=method,synthetic_tracked=n,explicit_certificate=cert,cache_mode=cache)
                    lines.append(table_line('{} / {} / {} / {}'.format(n,'empty certificate' if cert else 'positive',LABELS[method],cache),rows))
    lines += ['', 'No component/count support or future world was removed for speed. The dense certificate path retains all posterior components before bounded future sampling; sampled-away probability remains explicit. Cold and dense tails are separate acceptance evidence, not hidden by the primary aggregate.', '',
        '## Repeated L3-style calls and memory', '',
        '| Fixed root / optimized | Calls | P50 ms | P95 ms | P99 ms |', '|---|---:|---:|---:|---:|']
    for x in repeated:
        lines.append(table_line(x['root_id'][:12], x['records']))
    rssdelta=[x['rss_change_kib'] for x in repeated]
    lines += ['', 'The repeated loop varies shared-world seeds while holding physical time and arrived evidence fixed; all calls assert the belief fingerprint remains unchanged. Per-loop observed RSS changes for the optimized model are {} KiB. Process RSS includes all three models, histories, JIT and allocator storage, so it is not a per-belief memory requirement. Peak process `ru_maxrss` is {:.1f} MiB; final resident RSS is {:.1f} MiB. The 500-arrival physical session is a separate synthetic test and retains two hypotheses/count one, one current sensor receipt and flat sampled RSS {:.1f} MiB.'.format(rssdelta,real['metadata_after']['rss_max_kib']/1024,real['metadata_after']['rss_kib']/1024,validation['diagnostics']['persistent_session']['samples'][-1]['rss_kib']/1024), '',
        'Footprint storage is limited to 256 entries/8 MiB, graph-support results to 256 entries, endpoint templates to 64 entries/8 MiB and Sobol normals to 16 entries/8 MiB. These are cache bounds, not a global bound on an externally supplied population prior or all live caller-held successor branches.', '',
        '## Numerical and scientific cost of optimization', '',
        '{} optimized/reference structural comparisons pass; maximum absolute numeric difference is {:.3g}. Comparisons include posterior moments/counts/weights, sampled worlds, events, branch probabilities, cursors and full successors. They do not rematch sampled worlds. Symmetric PSD covariance roots repair seeded coupling under eigenbasis changes. Grouped GH/Sobol kernels preserve the finite reference rule, and compiled transition arithmetic preserves NumPy draw order and continuation RNG states.'.format(len(parity),aux['same_law']['maximum_absolute_error']), '',
        'Retained real-input posterior mass ranges from {:.3g} to {:.3g}, with hypothesis counts {}–{}. Thus some local bounds are nearly vacuous. Fast evaluation with high unresolved mass is not evidence of decision usefulness. Scalar event outputs are residual upper endpoints conditional on the surrogate, not globally certified native safety probabilities.'.format(*aux['retained_mass_range'],*aux['hypothesis_range']), '',
        '## Bottleneck investigation and bounded revision', '',
        'The preserved first release measurement (`benchmark_release_v1.json`) recorded optimized warm/cold P95 120.49/138.01 ms. It prompted one further systems revision, without model fitting. Dense cProfile showed repeated validation, whole-belief fingerprints and residual-template construction were major costs, followed by certificate quadrature. Packed dtype-preserving validation, shared immutable templates, one source check per batch and single-thread compiled canonical arithmetic address those costs. The final measurements above supersede that intermediate runtime; its failed and successful evidence remains archived.', '',
        'The validator experiment retains valid/invalid boundary parity, all 288 dense certificate components and count support. Packed validation adds hidden-covariance PSD checking that the old implementation omitted. No validity cache assumes caller-mutable live arrays are unchanged.', '',
        '## Cooperative budgets', '',
        'A deadline now starts before lock waiting and covers observe/forecast/query/successor work, including continuation. Expiry publishes a structured timeout; whole-call actual evidence remains pending and the input is unchanged. Native kernels and OS scheduling are non-preemptive. Measured overshoots are recorded in the validation artifact, so a hard real-time deadline is **not guaranteed**. No partial normalized result is presented as a completed probability computation.', '',
        '## Collector coexistence and reproduction', '',
        'Collector status, worker count, outcomes and process provenance are captured read-only before and after each run. Occasional pre-existing DEGRADED/long-running-batch status is retained rather than relabeled healthy. Co-running throughput is observational; no collector-only causal control was performed. Production configuration, workers, data and automatic-training settings were never changed.', '',
        'See [REPRODUCE.md](REPRODUCE.md), [SCIENTIFIC_VALIDATION.md](SCIENTIFIC_VALIDATION.md), [measured analysis](analysis_release_v2.json) and [figures](figures/README.md). The artifact manifest pins the source JSON, plots and unchanged trained assets.', '']
    (destination/'PERFORMANCE.md').write_text('\n'.join(lines))
    return aux


def plot_all(real,batches,validation,destination,provenance):
    figdir=destination/'figures';figdir.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':210,'pdf.fonttype':42})
    manifest={'schema':'structural_measured_figures_1','inputs':provenance,'figures':[]}
    def save(fig,name,caption):
        fig.tight_layout(rect=(0,0,1,.97))
        names=[]
        for ext in ('png','pdf'):
            path=figdir/(name+'.'+ext);fig.savefig(path,bbox_inches='tight');names.append({'path':path.name,'sha256':sha(path)})
        plt.close(fig);manifest['figures'].append({'name':name,'caption':caption,'files':names})
    rows=real['records']; rr=choose(rows,kind='real_dev')
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.2))
    for ax,cache in zip(axes,('warm','fully_cold')):
        for method in METHODS:
            v=np.sort([r['timings_ms']['outer_wall'] for r in choose(rr,method=method,cache_mode=cache)])
            ax.plot(v,np.arange(1,len(v)+1)/len(v),label=LABELS[method],color=COLORS[method])
        for t in (65,100):ax.axvline(t,color='black',ls=':' if t==65 else '--',lw=.8)
        ax.set(xlabel='Complete invocation wall time (ms)',ylabel='Empirical cumulative probability',title=cache.replace('_',' ').capitalize(),ylim=(0,1.02))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='lower right');fig.suptitle('60 development roots; 16 worlds, 24 rows, 3 s, four refined queries')
    save(fig,'01_complete_latency','Full-call empirical CDFs on the fixed 60-root cohort. Repeated calls are correlated; these are workload distributions, not scene-independent confidence intervals. Vertical gates:65 and100ms.')
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.2));qs=(1,4,8,16,24)
    for ax,cache in zip(axes,('warm','fully_cold')):
        for method in METHODS:
            v=[summary(choose(batches['records'],method=method,batch_size=q,cache_mode=cache))['timings_ms']['outer_wall']['p95'] for q in qs]
            ax.plot(qs,v,'o-',label=LABELS[method],color=COLORS[method])
        ax.axhline(100,c='black',ls='--',lw=.8);ax.axhline(65,c='black',ls=':',lw=.8)
        ax.set(xlabel='Candidate rows (one shared human forecast)',ylabel='Complete-call P95 (ms)',title=cache.replace('_',' ').capitalize(),xticks=qs);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Same 12 difficult development inputs at every candidate count')
    save(fig,'02_candidate_batches','Matched candidate scaling. Separate12-root cohort; five warm and two cold calls per root/method/count. Four refined candidates at most; all successors remain included.')
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.2))
    labels=['6 positive','12 positive','6 empty','12 empty'];fixtures=((6,False),(12,False),(6,True),(12,True));x=np.arange(4)
    for ax,cache in zip(axes,('warm','fully_cold')):
        for j,method in enumerate(('old_reference','structural_bound')):
            v=[summary(choose(rows,kind='synthetic',method=method,synthetic_tracked=n,explicit_certificate=c,cache_mode=cache))['timings_ms']['outer_wall']['p95'] for n,c in fixtures]
            ax.bar(x+(j-.5)*.36,v,.36,label=LABELS[method],color=COLORS[method])
        ax.axhline(100,c='black',ls='--',lw=.8);ax.set(xticks=x,xticklabels=labels,ylabel='Complete-call P95 (ms)',title=cache.replace('_',' ').capitalize());ax.tick_params(axis='x',labelrotation=15)
    axes[0].legend(fontsize=8);fig.suptitle('Unchanged dense fixture; 3,721 cells, hidden counts 0/2/4, 24 distinct queries')
    save(fig,'03_dense_stress','Dense positive and certified-empty cases, with hidden people retained. Twenty warm calls/case; only two cold calls/case, descriptive stress quantiles.')
    fig,axes=plt.subplots(1,3,figsize=(13.3,4.2));diag=validation['diagnostics']
    errs=[max(x['maximum_absolute_errors'].values()) for x in diag['analytic_composition']]
    axes[0].bar(np.arange(len(errs)),np.maximum(errs,1e-18),color='#0072b2');axes[0].set(yscale='log',ylabel='Maximum absolute moment discrepancy',xticks=range(6),xticklabels=['0/2','0/3','.013/2','.013/3','1.235/2','1.235/3'],title='Same absolute 3 s target');axes[0].tick_params(axis='x',labelrotation=30)
    c=diag['coarse_observation_composition'];axes[1].bar([0,1],[c['direct_no_detection'],c['segmented_joint_no_detection']],color=['#737373','#0072b2']);axes[1].set(xticks=[0,1],xticklabels=['Direct','Segmented'],ylabel='Joint no-detection probability',title='Same finite retained observation law');axes[1].text(.05,.95,'Absolute error {:.2g}\nPosterior error {:.2g}'.format(c['probability_absolute_error'],c['conditional_weights_max_error']),transform=axes[1].transAxes,va='top',fontsize=9)
    v=diag['physical_variance_law'];axes[2].plot([x['horizon_seconds'] for x in v],[x['analytic_velocity_variance_per_axis'] for x in v],'o-',c='#0072b2');axes[2].set(xlabel='Physical forecast horizon (s)',ylabel='Velocity variance per axis',title='Declared maneuver variance law');axes[2].grid(alpha=.2)
    save(fig,'04_probability_and_transition','Analytic moment composition across2/3segments at three absolute origins; retained finite-law observation composition; analytic maneuver variance. No global Gaussian-projection or native physical-support validation is claimed.')
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.2))
    for i,x in enumerate(z for z in real['repeated_planner'] if z['method']=='structural_bound'):
        samples=x['memory_samples'];base=samples[0]['rss_kib']
        axes[0].plot([s['call'] for s in samples],[(s['rss_kib']-base)/1024 for s in samples],'o-',label=x['root_id'][:10])
    axes[0].set(xlabel='Repeated planner invocation',ylabel='Change in process RSS (MiB)',title='Fixed arrived beliefs; varied future seed');axes[0].legend(fontsize=8)
    ss=diag['persistent_session']['samples'];axes[1].plot([s['step'] for s in ss],[s['rss_kib']/1024 for s in ss],'o-',color='#0072b2');axes[1].set(xlabel='Actual synthetic observation update',ylabel='Process RSS (MiB)',title='500 arrivals over24physical seconds');axes[1].ticklabel_format(useOffset=False,axis='y')
    save(fig,'05_persistence_memory','Observed RSS samples, not a global no-leak proof. Left holds physical time fixed; right advances through500synthetic updates. Process includes models,JIT,Python and allocator state.')
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.2))
    for ax,cert in zip(axes,(False,True)):
        b=[x for x in diag['cooperative_budgets'] if x['certificate']==cert]
        for status,marker,color in [('COMPLETE','o','#0072b2'),('PENDING_OBSERVATION','x','#d55e00')]:
            vals=[x for x in b if x['status']==status]
            if vals:ax.scatter([x['budget_ms'] for x in vals],[x['elapsed_ms'] for x in vals],label=status,s=35,marker=marker,color=color)
        ax.plot([0,125],[0,125],c='black',ls='--',lw=.8);ax.set(xlabel='Requested cooperative budget (ms)',ylabel='Observed return wall time (ms)',title='Empty certificate' if cert else 'Positive observation');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig,'06_cooperative_budgets','Cooperative whole-call deadlines. Timeout leaves actual evidence pending and input unchanged. Non-preemptive stages can overrun; this is explicitly not a hard real-time guarantee.')
    fig,axes=plt.subplots(1,3,figsize=(13.3,4.2));w=choose(rr,method='structural_bound',cache_mode='warm');rootgroups=collections.defaultdict(list)
    for row in w:rootgroups[row['root_id']].append(row)
    values=[(g[0],np.percentile([x['timings_ms']['outer_wall'] for x in g],95)) for g in rootgroups.values()]
    for ax,key,label in zip(axes,('graph_cells','incoming_tracked_max','retained_mass'),('Observed graph cells','Incoming tracked slots','Retained posterior mass')):
        ax.scatter([x[key] for x,y in values],[y for x,y in values],s=23,alpha=.75,c='#0072b2');ax.axhline(100,c='black',ls='--',lw=.8);ax.set(xlabel=label,ylabel='Per-root warm P95 (ms)');ax.grid(alpha=.2)
        if key=='retained_mass':ax.set_xscale('log')
    fig.suptitle('Difficult inputs and unresolved-mass visibility; one point per root')
    save(fig,'07_workload_and_mass','Per-root latency versus legal graph/track size and retained mass. High unresolved mass is shown explicitly: fast but vacuous outputs do not establish planning utility.')
    for m in manifest['figures']:
        for f in m['files']:f['bytes']=(figdir/f['path']).stat().st_size
    (figdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Measured structural diagnostics','', 'All figures are regenerated from sealed structural JSON without training or reading native future truth.','']
    for m in manifest['figures']:lines+=['## '+m['name'],'',m['caption'],'','![{}]({}.png)'.format(m['name'],m['name']),'']
    (figdir/'README.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',default='output/l2_partial_graph_wm_structural_20261021')
    parser.add_argument('--destination',default='docs/l2/partial_graph_wm_structural_20261021')
    args=parser.parse_args();root=Path(args.inputs);dest=Path(args.destination)
    files={name:root/(name+'.json') for name in ('benchmark_release_v2','benchmark_batches_v2','validation_release_v2')}
    data={k:json.loads(p.read_text()) for k,p in files.items()}
    for name,value in data.items():
        if value['failure'] is not None:raise ValueError('input experiment failed:'+name)
        if value.get('source_changed_during_measurement',False) or value.get('sources_unchanged',True) is False:
            raise ValueError('source changed:'+name)
    if not all(data[k]['all_same_law_comparisons_passed'] for k in ('benchmark_release_v2','benchmark_batches_v2')):
        raise ValueError('same-law comparison failed')
    source=data['validation_release_v2']['source_sha256']
    for k in ('benchmark_release_v2','benchmark_batches_v2'):
        b=data[k]['sources_after']
        if any(b[Path(name).name]!=digest for name,digest in source.items()):raise ValueError('runtime sources differ between final artifacts')
    dest.mkdir(parents=True,exist_ok=True)
    if any((dest/n).exists() for n in ('PERFORMANCE.md','analysis_release_v2.json','figures')):
        raise FileExistsError('refuse to overwrite measured report/figures')
    provenance={k:{'path':str(p),'sha256':sha(p)} for k,p in files.items()}
    provenance['generator']={'path':str(Path(__file__).relative_to(REPO)),'sha256':sha(__file__)}
    result=create_report(data['benchmark_release_v2'],data['benchmark_batches_v2'],data['validation_release_v2'],provenance,dest)
    plot_all(data['benchmark_release_v2'],data['benchmark_batches_v2'],data['validation_release_v2'],dest,provenance)
    print(json.dumps({'destination':str(dest),'gates':result['gates'],'decision':result['decision']}))


if __name__=='__main__':
    main()
