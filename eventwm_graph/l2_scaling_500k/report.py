"""Render the predeclared comparisons; preserve full scientific artifacts."""
import csv
import gzip
import json
from pathlib import Path
import shutil
import numpy as np
from .common import BASE,OUT,DOC,PRIOR,sha,dump

def read(path):return json.loads(Path(path).read_text())
def number(x):return '—' if x is None else '%.4f'%x
def interval(x):return '%s [%s, %s]'%(number(x['mean']),number(x['lo']),number(x['hi']))

def render():
    c=read(OUT/'comparison.json');prep=read(OUT/'preparation.json');old=read(PRIOR/'training/training.json');new=read(OUT/'training/training.json')
    control=read(OUT/'training_control.json');assert control['epoch0_all_tensors_bitwise_equal']
    primary=next(r for r in c['forecasting']['scopes'] if r['cohort']=='native' and r['scope']=='one_second')
    effect=primary['paired_new_minus_original']['path_es']
    strong=effect['roots']>=30 and effect['scenes']>=10
    decision=('Primary native-gap scores support further scaling, subject to calibration and belief limitations.' if strong and effect['hi']<0 else
              'The primary native-gap contrast worsens; further scaling is not promising under this frozen pipeline.' if strong and effect['lo']>0 else
              'Further scaling is not established by the primary native-gap contrast; the evidence is inconclusive.')
    lines=['# Controlled data-scaling experiment','',decision,'',
        'Secondary evidence is encouraging: artificial-gap forecasting, all-gap association diagnostics and several native motion metrics improve. The predeclared primary native-gap scaling criterion is not met. Further scaling alone should not take priority over the remaining belief and dense-runtime revisions.','',
        'Only fitting data changed. Architecture and parameter count (33,309), motion law, hazards, association, objective, hyperparameters, epoch schedule, seed 221, selection rule and deployed inference remain frozen. Learned weights are retrained from the exact original initialization, not fine-tuned from the selected checkpoint.','',
        '| Quantity | Original | Scaling experiment |','|---|---:|---:|',
        '| Total sealed outcomes | 128,990 | 484,615 |',
        '| Training-eligible outcomes | 72,539 | 270,521 |',
        '| Eligible independent training roots | 9,935 | 37,086 |',
        '| Independent supervised training roots | 3,212 | %s |'%format(prep['supervised_roots'],','),
        '| Completed epochs | %s | %s |'%(old['completed_epochs'],new['completed_epochs']),
        '| Selected development path ES | %.6f | %.6f |'%(old['selected_dev_path_es'],new['selected_dev_path_es']),
        '', 'All original training roots and the complete supervised-cache prefix are retained. Selection remains 666 roots/98 scenes; evaluation remains 2,110 roots/108 scenes. The user selected the original nominal total-outcome convention. Protected development, confirmation and holdout payloads do not enter gradient fitting. The original frozen development set is reused for selection and evaluation, not treated as fresh confirmation.','',
        '## Paired forecasting comparison','',
        'Lower Energy Score is better. Values below use common supported physical cases; differences are scaled minus original. Intervals are 95% paired scene bootstrap intervals, averaging streams and observations within root, then roots within scene. Native practical age is [.75,1.25] seconds, with future H=.5/1/2 seconds.','',
        '| Frozen stratum / metric | Original | Scaled | Paired difference[95% CI] | Roots/scenes |','|---|---:|---:|---:|---:|']
    for cohort,scope,label in [('native','visible','Native visible'),('native','one_second','Native~1s missing'),('masked','one_second','Artificial~1s missing')]:
        r=next(r for r in c['forecasting']['scopes'] if r['cohort']==cohort and r['scope']==scope)
        for metric in ('tracked_es','path_es','coverage50','coverage90'):
            d=r['paired_new_minus_original'][metric]
            lines.append('| %s / %s | %s | %s | %s | %d/%d |'%(label,metric,number(r['common_original'][metric]['mean']),number(r['common_scaled'][metric]['mean']),interval(d),d['roots'],d['scenes']))
    lines+=['','The primary native-gap path ES falls from 0.5955 to 0.5708 (4.2%), but its paired interval crosses zero. Artificial-gap path ES improves with a negative interval; that does not establish native occlusion superiority. Visible 50% coverage rises from .541 to .571, farther from its nominal .5 target. Native-gap 90% coverage remains only .729. Calibration has not been resolved.','',
        'All age/horizon cells, support/censor counts and calibration inputs are in [EVALUATION.md](EVALUATION.md), `artifacts/comparison.json` and `age_horizon.csv`. Artificial masks are unavailable human measurements, not native occlusion or empty detections. Two fixed root streams quantify finite sampling sensitivity; these intervals do not include training-seed variability.','',
        '| Forecast-case status, repeated case rows | Original | Scaled |','|---|---:|---:|']
    support=c['forecasting']['support']
    for name in sorted(set(support['original'])|set(support['scaled'])):
        lines.append('| %s | %d | %d |'%(name,support['original'].get(name,0),support['scaled'].get(name,0)))
    lines+=['','Case rows include repeated horizons and streams; they are not independent futures. The original target ledger and censoring are retained, including failed cases.','',
        '## Reobservation and remaining belief defects','',
        '| Native reappearance, H=.5s | Original | Scaled | Paired difference[95% CI] | Roots/scenes |','|---|---:|---:|---:|---:|']
    for scope in ('all','one_second'):
        r=next(r for r in c['reobservation']['cells'] if r['scope']==scope and r['H']==.5)
        for metric in ('future_path_es_gain','association_accuracy','false_split','false_merge','duplicate'):
            d=r['paired_new_minus_original'][metric]
            lines.append('| %s / %s | %s | %s | %s | %d/%d |'%(scope,metric,number(r['common_original'][metric]['mean']),number(r['common_scaled'][metric]['mean']),interval(d),d['roots'],d['scenes']))
    lines+=['','Positive future-path gain means posterior improvement at the same absolute target, excluding the assimilated current point. Association accuracy is the mass of the correct preferred association; split/merge/duplicate diagnostics overlap and do not form an exhaustive partition. Cells below 30 roots or 10 scenes remain inconclusive, including the native practical-gap reappearance stratum. All-gap association and split/merge diagnostics improve with paired intervals excluding zero; the change in same-target future forecast gain remains inconclusive.','',
        '| Operational belief-state rows | Original | Scaled |','|---|---:|---:|']
    for field in ('states','zero_true_count_support','duplicate_ancestry_states','missing_preferred_ancestry_states'):
        lines.append('| %s | %d | %d |'%(field,c['belief']['original'][field],c['belief']['scaled'][field]))
    lines+=['',
        'The original readiness labels remain **FORECASTING REVISE** and **BELIEF / REOBSERVATION REVISE**. **SYSTEMS REVISE** also applies to the expanded/dense 100 ms workload. Data scaling does not repair the finite association beam, recycled slot identity, initialization, miss-projection or clock-renewal defects. Zero true-count support and duplicate states decrease, but unsupported forecast ancestries and missing preferred ancestry states increase. Supported-case ES is not a certificate of a correct persistent belief.','',
        '## Native motion physics','',
        '| Metric | Original | Scaled | Observed | Paired change[95% CI] |','|---|---:|---:|---:|---:|']
    for name in ('stop_frequency','restart_frequency','speed_p95','speed_p99','turn_p95_rad','stop_duration_p95'):
        r=next(r for r in c['physical']['metrics'] if r['metric']==name)
        lines.append('| %s | %s | %s | %s | %s |'%(name,number(r['original']['mean']),number(r['scaled']['mean']),number(r['actual']['mean']),interval(r['difference'])))
    lines+=['','Stop/restart frequency, speed tails and turning move toward native values. Speed and turning remain excessive, and completed stop-duration calibration does not improve. The native matched-interval cohort, finite-difference criteria, completed-duration censoring and 32 worlds/two root streams are unchanged. Stop/Move hazards are copied byte for byte; this experiment introduces no new dynamics law.','',
        '## Inference speed','',
        '| Same complete workload, P95 ms | Original, remeasured | Scaled |','|---|---:|---:|']
    benchmarks={arm:read(OUT/('benchmark_'+arm)/'complete/benchmark.json') for arm in ('original','new')}
    for key,label in [('representative_first60_quantiles','Representative'),('dense_quantiles','Dense'),('primary_quantiles','Whole 83-root cohort')]:
        for cache in ('warm','dynamic_cold'):
            values=[]
            for arm in ('original','new'):
                entry=next(v for k,v in benchmarks[arm][key].items() if cache in k and "16, 'primary'" in k)
                values.append(entry['metrics']['combined_ms']['p95_ms'])
            lines.append('| %s/%s | %.3f | %.3f |'%(label,cache,*values))
    lines+=['','The representative 100 ms P95 gate passes (87.4/88.3 ms), while the expanded cohort (107.2/109.1 ms) and dense cases (116.9/115.9 ms) fail. The preferred 65 ms target is not met. The observation update remains the dominant cost.','',
        'Complete means real observation update, shared 3 s/16-world/64-timestamp future, fixed-query observation prediction and supported successor. Warm and dynamically cold are loaded-process measurements; imports/JIT startup and legal-prefix reconstruction are outside this established gate. Dense cases and unsupported calls remain reported. Sequential live-collector measurements do not establish a causal latency penalty from data scaling; budget and dense-case candidate counts are identical. See [PERFORMANCE.md](PERFORMANCE.md).','',
        'All 399 unchanged regression tests pass. Complete artifacts, fixed-case failures/censoring, checkpoints, optimizer/RNG states and source/data hashes are preserved. See [DATA.md](DATA.md), [PROTOCOL.md](PROTOCOL.md) and [REPRODUCE.md](REPRODUCE.md). No production merge or collector modification occurred.','']
    (DOC/'FINAL_REPORT.md').write_text('\n'.join(lines))
    detail=['# Frozen horizon comparisons','','Differences are scaled minus original on the same supported physical cases. All intervals use the unchanged paired scene bootstrap. Ages and future horizons are separate.','']
    for name,part in [('Native visible',[r for r in c['forecasting']['cells'] if r['cohort']=='native' and r['age_bin']=='visible']),
        ('Native age[.75,1.25]s',[r for r in c['forecasting']['practical_age_horizons'] if r['cohort']=='native']),
        ('Artificial age[.75,1.25]s',[r for r in c['forecasting']['practical_age_horizons'] if r['cohort']=='masked'])]:
        detail+=['## '+name,'','| Hs | Metric | Original | Scaled | Paired change[95% CI] | Roots/scenes |','|---|---|---:|---:|---:|---:|']
        for r in sorted(part,key=lambda r:r['H']):
            for metric in ('tracked_es','path_es','coverage50','coverage90'):
                d=r['paired_new_minus_original'][metric]
                detail.append('| %.1f | %s | %s | %s | %s | %d/%d |'%(r['H'],metric,number(r['common_original'][metric]['mean']),number(r['common_scaled'][metric]['mean']),interval(d),d['roots'],d['scenes']))
        detail+=['']
    detail+=['Cells below 30 roots or 10 scenes are inconclusive. Full original/scaled status denominators, including truth censoring, annotation censoring, unsupported ancestry and operational failures, remain in the JSON/CSV artifacts. Coverage calibration error is the absolute difference from .5 or .9; no post-hoc recalibration is fitted. Native and artificial cases are never pooled.','']
    (DOC/'EVALUATION.md').write_text('\n'.join(detail))
    curves={name:read(folder/'training/learning.json') for name,folder in (('original',PRIOR),('scaled',OUT))}
    selected={name:min((r for r in curve if r['epoch']>0),key=lambda r:r['dev_es']) for name,curve in curves.items()}
    training=['# Frozen fitting experiment','',
        'One revised-data run, from scratch. The full original initialization and initial development score match exactly. The loss, motion parameters, seed, batch size, optimizer, learning-rate schedule and significant-best patience rule are unchanged; source/data hashes are in the preserved configuration and control manifests.','',
        '| Quantity | Original | Scaled |','|---|---:|---:|',
        '| Supervised roots | 3,212 | %d |'%prep['supervised_roots'],
        '| Supervised scenes | 421 | %d |'%prep['scenes'],
        '| Selected epoch | %d | %d |'%(selected['original']['epoch'],selected['scaled']['epoch']),
        '| Completed epochs | %d | %d |'%(old['completed_epochs'],new['completed_epochs']),
        '| Optimizer steps | %d | %d |'%(int(np.ceil(3212/48))*old['completed_epochs'],int(np.ceil(prep['supervised_roots']/48))*new['completed_epochs']),
        '| Fitting wall seconds | %.2f | %.2f |'%(old['seconds'],new['seconds']),
        '| Stop reason | %s | %s |'%(old['stop_reason'],new['stop_reason']),
        '', 'The original 3,212 expanded cache rows are retained verbatim. Population prior, optical coefficients and Stop/Move hazards are copied, without refitting. Actual observations remain unavailable when masked; withheld positions never filter the belief. The original gap-weighting routine shifts future target times and does not manufacture native missing-observation input examples. Its selected supervised tracks are root-visible; this limitation remains unchanged.','',
        'More roots imply more minibatches and optimizer updates under the same epoch schedule. This is a data-scaling intervention with that prescribed schedule, not a comparison at equal compute. Confidence intervals describe paired evaluation uncertainty conditional on the single fixed training seed. No new scientific defect is repaired and no alternative architecture or candidate is searched.','',
        'Initial, selected and final checkpoints include optimizer/scheduler/RNG state. All per-epoch target selections, learning curves and source/configuration hashes are preserved in `artifacts/training/`. The original curve is retained separately.','']
    (DOC/'TRAINING.md').write_text('\n'.join(training))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,ax=plt.subplots(figsize=(7,4))
    for name,curve in curves.items():
        ax.plot([r['epoch'] for r in curve],[r['dev_es'] for r in curve],label=name)
        best=selected[name];ax.scatter([best['epoch']],[best['dev_es']],s=28)
    ax.set(xlabel='Epoch',ylabel='Frozen development path Energy Score');ax.grid(alpha=.2);ax.legend();figure.tight_layout()
    figure.savefig(DOC/'learning_curves.svg');plt.close(figure)
    svg=DOC/'learning_curves.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    with (DOC/'age_horizon.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['cohort','age_bin','H','metric','old_mean','new_mean','paired_delta','lo','hi','paired_roots','paired_scenes','old_status','new_status'])
        for r in c['forecasting']['cells']:
            for metric,d in r['paired_new_minus_original'].items():
                writer.writerow([r['cohort'],r['age_bin'],r['H'],metric,r['common_original'][metric]['mean'],r['common_scaled'][metric]['mean'],d['mean'],d['lo'],d['hi'],d['roots'],d['scenes'],json.dumps(r['original_status']),json.dumps(r['new_status'])])

def archive():
    dest=DOC/'artifacts';dest.mkdir(exist_ok=False)
    for name in ('comparison.json','preparation.json','data_integrity.json','supervision_audit.json','runtime_setup.json',
        'pre_fit_input_control.json','training_control.json','evaluation_reproduction.json','evaluation_reappearance_reproduction.json',
        'collector_version_compatibility.json','preparation_compatibility.json','preparation_parallelism.json','evaluation_complete.json','statistics_reference.json','initialization_control.json','practical_age_denominator_reference.json',
        'original_baseline_complete.json','baseline_historical_reconciliation.json','environment.json'):
        shutil.copyfile(OUT/name,dest/name)
    for source,target in ((OUT/'training',dest/'training'),(OUT/'evaluation_arm_new/evaluation',dest/'evaluation'),
        (OUT/'evaluation_arm_original_full/evaluation',dest/'baseline_evaluation'),
        (OUT/'benchmark_original/complete',dest/'latency_original'),(OUT/'benchmark_new/complete',dest/'latency_scaled'),
        (OUT/'scaling_regressions',dest/'regressions')):shutil.copytree(source,target)
    for name in ('source_manifest.json','evaluation_orchestration.json'):
        shutil.copyfile(OUT/'evaluation_runtime_new'/name,dest/name)
    shutil.copyfile(PRIOR/'training/learning.json',dest/'original_learning.json')
    for name in ('physical.json','physical_execution_guard.json'):shutil.copyfile(OUT/'evaluation_arm_new'/name,dest/name)
    for name in ('physical.json','physical_execution_guard.json'):
        shutil.copyfile(OUT/'evaluation_arm_original_full'/name,dest/('baseline_'+name))
    for name in ('source_manifest.json','evaluation_orchestration.json'):
        shutil.copyfile(OUT/'evaluation_runtime_original_full'/name,dest/('baseline_'+name))
    # Large payloads remain in the immutable research snapshot. Preserve the
    # complete membership/hashes and byte-exact fitting caches in compact form.
    for source,name in ((OUT/'snapshot/manifest.json','snapshot_manifest.json.gz'),(OUT/'train.npy','train.npy.gz'),(OUT/'dev.npy','dev.npy.gz')):
        with source.open('rb') as src,(dest/name).open('xb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=3,mtime=0) as target:shutil.copyfileobj(src,target)
    guard=read(OUT/'pre_fit_input_control.json');assert all(sha(Path(p))==h for p,h in guard['files'].items())
    dump(dest/'manifest.json',dict(files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()},
        original_baseline_checkpoint_sha256=sha(PRIOR/'training/checkpoint.pt'),new_selected_checkpoint_sha256=sha(OUT/'training/checkpoint.pt'),
        protected_inputs_unchanged=True,source='051aa8c6 plus committed experiment orchestration; deployed inference unchanged'))

def performance():
    arms={a:read(OUT/('benchmark_'+a)/'complete/benchmark.json') for a in ('original','new')}
    lines=['# Complete Layer 2 timing','',
        '**Representative 100 ms P95: PASS. Expanded and dense workloads: SYSTEMS REVISE.** The scaled model measures 87.4/88.3 ms representative warm/dynamic-cold, 107.2/109.1 ms across the expanded cohort, and 116.9/115.9 ms on dense cases. The 65 ms preference is not met.','',
        'The unchanged complete invocation includes actual observation updating, a shared 3-second/16-world/64-timestamp future, fixed-query human-observation prediction and a supported none-event successor. Track readout, generic field projection and recursive propagation are separately timed. No input-domain checks, people, worlds or conditional work are omitted.','',
        '| Cohort | Model | Cache | P50 ms | P95 ms | P99 ms | Complete calls / roots | Attempts |','|---|---|---|---:|---:|---:|---:|---:|']
    for group,label in [('representative_first60_quantiles','Representative'),('primary_quantiles','Expanded 83 roots'),('dense_quantiles','Dense 12 roots')]:
        for arm,x in arms.items():
            for cache in ('warm','dynamic_cold'):
                v=next(v for k,v in x[group].items() if cache in k and "16, 'primary'" in k);q=v['metrics']['combined_ms']
                lines.append('| %s | %s | %s | %.3f | %.3f | %.3f | %d/%d | %d |'%(label,arm,cache,q['p50_ms'],q['p95_ms'],q['p99_ms'],q['n'],q['roots'],v['attempts']))
    lines+=['','The original 83-root cohort, metadata-only selection, 5 warm repeats and 2 dynamically cold repeats are unchanged. Both arms retain 7 unsupported primary calls from the same unsupported query and have no initialization failures. Quantiles condition on complete supported calls; those failures are not dropped from the ledger. Repeats are runtime replicates, not independent physical futures.','',
        '## Scaled representative components','',
        '| Component | Cache | P50 ms | P95 ms | P99 ms |','|---|---|---:|---:|---:|']
    for cache in ('warm','dynamic_cold'):
        v=next(v for k,v in arms['new']['representative_first60_quantiles'].items() if cache in k and "16, 'primary'" in k)
        for name,q in v['metrics'].items():lines.append('| %s | %s | %.3f | %.3f | %.3f |'%(name,cache,q['p50_ms'],q['p95_ms'],q['p99_ms']))
    lines+=['','The actual human observation update remains the largest combined-call component. Component quantiles do not add to the combined quantile. Dense warm observation-update P95 is 74.4 ms original and 89.3 ms scaled; forecast P95 is 19.4 and 23.1 ms. Dense mean tracked-state maximum (5.75), components (32), distinct descendant seeds/clocks (32), and enumerated candidates (237.3) are identical. Sequential runs under a live collector do not identify a causal latency penalty from learned weights; both observed runs are retained without selecting favorable timings.','',
        '## World-count sensitivity','',
        '| Worlds | Model | Cache | Complete combined P95 ms | Complete calls / roots |','|---:|---|---|---:|---:|']
    for worlds in (8,16,32,64):
        for arm,x in arms.items():
            for cache in ('warm','dynamic_cold'):
                v=next(v for k,v in x['world_sensitivity_quantiles'].items() if cache in k and (", %d, 'world_sensitivity'"%worlds) in k);q=v['metrics']['combined_ms']
                lines.append('| %d | %s | %s | %.3f | %d/%d |'%(worlds,arm,cache,q['p95_ms'],q['n'],q['roots']))
    lines+=['','Both models use 32 actual components and unchanged stochastic support. Dynamic cold clears the original dynamic caches in an already initialized process. Imports, JIT startup and legal-prefix reconstruction are outside this established integration gate; this experiment makes no new startup-latency claim. All heavy fitting and scoring jobs ended before sequential original/scaled timing. The normal-priority 28-worker collector remained live.','',
        'Raw timings, component P50/P95/P99 for every cohort, sensitivity calls, state-ownership checks, cache diagnostics, environment and source/input hashes are preserved under `artifacts/latency_original/` and `artifacts/latency_scaled/`. The inference implementation is byte-identical to 051aa8c6.','']
    (DOC/'PERFORMANCE.md').write_text('\n'.join(lines))

if __name__=='__main__':render();performance();archive()
