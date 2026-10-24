"""Seal two separate research deliverables after all fits/evaluations/audits."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

BASE=Path(__file__).resolve().parents[1]
TRANSITION=BASE/'output/research_transition_20261024'
S1=BASE/'output/l2_two_stage_20261023'
S2=BASE/'output/stage2_graphdelta_20261024'
DOC=BASE/'docs/l2/final_stage1_stage2_20261024'
REVIEW=DOC/'reviews'
PYTHON='/research-user/miniforge3/envs/eventwm-falcon/bin/python'
NAME='final_stage1_research_20261024'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def link(path,label=None):return '['+(label or Path(path).name)+']('+str(path)+')'


def estimate(value):
    if value.get('mean') is None:return 'Unscored / insufficient support'
    return '%.6f [%.6f, %.6f]; %s Roots / %s scenes'%(value['mean'],value['lo'],value['hi'],value['roots'],value['scenes'])


def copy_release(name,checkpoint,config,complete,source):
    dest=TRANSITION/name;dest.mkdir(exist_ok=False)
    for src,filename in [(checkpoint,'checkpoint.pt'),(config,'config.json'),(complete,'complete.json')]:
        shutil.copyfile(src,dest/filename);assert sha(src)==sha(dest/filename);(dest/filename).chmod(0o444)
    dump(dest/'SOURCE.json',dict(path=str(source),manifest=str(source/('FINAL_SOURCE_IDENTITY.json' if (source/'FINAL_SOURCE_IDENTITY.json').exists() else 'manifest.json'))))
    (dest/'SOURCE.json').chmod(0o444)
    return dest


def main():
    required=[TRANSITION/'STAGE1_EVALUATION_COMPLETE.json',S2/'COMPLETE.json',S2/'optimized_runtime/COMPLETE.json',
        TRANSITION/'final_collection_all_splits/manifest.json',REVIEW/'REVIEW_ALL_SPLITS_SNAPSHOT.json']
    while not all(p.exists() for p in required):time.sleep(5)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (S2/'final_independent_audit.log').open('w') as stream:
        subprocess.run([PYTHON,str(REVIEW/'review_stage2_optimized_final.py')],cwd=BASE,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    stopped=json.loads(subprocess.check_output([str(Path.home()/'.local/bin/collectorctl'),'status'],text=True))
    assert stopped['health']['collection_status']=='MANUALLY_STOPPED' and stopped['service']['ActiveState']=='inactive' and not stopped['campaign']['worker_pids']
    dump(TRANSITION/'collector_final.json',stopped)
    full=read(TRANSITION/'final_collection_all_splits/manifest.json');fitting=read(S1/'final_collection_20261024/manifest.json')
    c1=read(S1/'experiments'/NAME/'config.json');done1=read(S1/'experiments'/NAME/'complete.json')
    forecast=read(S1/'evaluation'/NAME/'metrics.json');v=forecast['variants'][NAME]
    current=read(S1/'evaluation/final_stage1_current_20261024/summary.json')
    runtime=read(REVIEW/'STAGE1_RUNTIME_REVIEW.json')
    h0=read(S1/'evaluation/final_stage1_exact_h0_20261024/b_exact_reliability.json')
    reliability=read(S1/'evaluation/final_stage1_reliability_20261024/reliability.json')
    persistence=read(REVIEW/('REVIEW_A_MULTI_PERSISTENCE_'+NAME+'.json'))
    inference=Path((TRANSITION/'inference_source_path.txt').read_text().strip())
    s1_release=copy_release('stage1_research_release',S1/'experiments'/NAME/'last.pt',S1/'experiments'/NAME/'config.json',S1/'experiments'/NAME/'complete.json',inference)
    m2=read(S2/'evaluation/optimized_dev/metrics.json');b2=read(S2/'benchmarks/optimized/metrics.json');c2=read(S2/'training/final/config.json');done2=read(S2/'training/final/complete.json')
    s2source=Path((S2/'runtime_source_path.txt').read_text().strip());s2_release=copy_release('stage2_research_release',S2/'training/final/last.pt',S2/'training/final/config.json',S2/'training/final/complete.json',s2source)
    for dest,data in [(s1_release,dict(forecast=forecast,current=current,runtime=runtime,exact_h0=h0,finite40_reliability=reliability,persistence=persistence)),
                      (s2_release,dict(evaluation=m2,performance=b2))]:
        for label,value in data.items():dump(dest/(label+'.json'),value);(dest/(label+'.json')).chmod(0o444)
    a=['# Final Stage 1 research release','',
        '**Stage 1 development is closed. This is a research checkpoint, not an officially accepted release. Stage 2 is the primary development objective.**','',
        f"The collector is inactive, disabled and MANUALLY_STOPPED with zero workers. Its final complete snapshot contains {full['roots']:,} sealed independent Roots, {full['outcomes']:,} sibling Query outcomes and {full['scenes']} scenes. All splits, source identities, logs and payloads are preserved; no further collection is planned.",'',
        f"The final compatible fitting union contains **{fitting['training_roots']:,} Roots / {fitting['training_outcomes']:,} outcomes / {fitting['training_scenes']} scenes**. It retains every previous fitting Root and adds all compatible eligible new Roots. Existing exclusions and the original 2,110-Root / 108-scene development boundary remain unchanged. Query siblings and causal anchors are dependent; confidence intervals resample scenes.",'',
        'Dataset manifest: '+link(TRANSITION/'final_collection_all_splits/manifest.json')+'. Fitting manifest: '+link(S1/'final_collection_20261024/manifest.json')+'. Independent full-byte audits: '+link(REVIEW/'REVIEW_DATA_INTEGRITY.json')+' and '+link(REVIEW/'REVIEW_ALL_SPLITS_SNAPSHOT.json')+'.','',
        '## Checkpoint and exact fit','',
        'Loadable final checkpoint: '+link(s1_release/'checkpoint.pt')+'. SHA256 `'+sha(s1_release/'checkpoint.pt')+'`.','',
        f"The established history-conditioned O model was fitted once to all {c1['fit_roots']:,} Roots and {c1['fit_anchors']:,} causal anchors. Fixed 8 epochs, seed413, batch48, AdamW lr0.0002/decay0.001, clip5, four iid worlds, history fraction0.5 and guided association. Initialization is the original O pilot, not the previous full-data checkpoint or a later calibration pilot. Final selection is last.pt. Completed {done1['steps']:,} steps in {done1['seconds']:.1f}s after loading; peak CUDA allocation {done1['peak_cuda_bytes']:,} bytes.",'',
        'Exact configuration: '+link(s1_release/'config.json')+'. Training Git commit `'+c1['git_commit']+'`; immutable original-source overlay and every source hash are preserved in '+link(TRANSITION/'training_source/FINAL_SOURCE_IDENTITY.json')+'. Only its offline target-cache path changed. The model and population law remain unchanged.','',
        '## Forecasting and coverage','',
        'Values show mean [scene-bootstrap 95% interval] and independent support. Native target times and original masks are unchanged.','',
        '| Cohort | Trajectory Energy Score | Coverage90 |','|---|---|---|']
    for key,label in [('all','All tracked'),('visible','Visible'),('practical_gap','Native-target practical gap'),('reappearance','Reappearance')]:
        a.append('| '+label+' | '+estimate(v[key]['path_es'])+' | '+estimate(v[key]['coverage90'])+' |')
    a+=['','| Horizon | Energy Score | Coverage90 |','|---|---|---|']
    for horizon,values in v['horizons'].items():a.append('| '+horizon+'s | '+estimate(values['path_es'])+' | '+estimate(values['coverage90'])+' |')
    a+=['','True native-occlusion horizon results, all cohort/horizon cross-tabs, whole-population scores, status counts and all baseline parameters/comparisons are retained in '+link(s1_release/'forecast.json')+'.','',
        '| Matched baseline | Aggregate relative change, candidate minus baseline |','|---|---|']
    for baseline in ('cv','memory_cv','graph_ridge','pilot_k_bounded_unique'):
        result=forecast['comparisons'][NAME+'_minus_'+baseline]['all']
        a.append('| '+baseline+' | '+estimate(result)+' |')
    a+=['','Relative changes are fractions; negative favors the final model. Original primary improvement≥2%, positive lower confidence bound, native-gap noninferiority upper bound≤2%, support and coverage requirements remain unchanged.','',
        '## Population quality and limitations','',
        '| All-current metric | Value and independent support |','|---|---|']
    for metric,values in current['summaries']['all_current']['model'].items():a.append('| '+metric+' | '+estimate(values)+' |')
    a+=['',f"Exact H0 reliability retains {h0['anchors']:,} anchors / {h0['roots']:,} Roots / {h0['scenes']} scenes and reports **{len(h0['supported_pointwise_deviations'])} supported pointwise deviations**. Finite40 reliability reports {len(reliability['supported_pointwise_deviations'])} supported pointwise deviations. Pointwise reliability diagnostics retain their original multiplicity and Monte Carlo caveats; they are not silently converted into new acceptance thresholds.",'',
        'Exact H0, H+, occupancy reliability, raw probability banks, current-count NLL, proper-score intervals and estimator MCSE bounds remain in '+link(S1/'evaluation/final_stage1_exact_h0_20261024')+', '+link(S1/'evaluation/final_stage1_hpositive_reliability_20261024')+' and '+link(S1/'evaluation/final_stage1_reliability_20261024')+'.','',
        'Historical physical-feasibility and persistence rerun: '+link(s1_release/'persistence.json')+'. All failures, sampling caps and insufficient-support strata are retained. The historical development cohort is previously used research data; no fresh protected-confirmation claim is made. Population calibration is not established as an accepted release requirement. Remaining deficiencies do not trigger another Stage 1 research cycle.','',
        '## Complete CPU performance','',
        f"The unchanged K40/64knots/3s complete-call protocol made {runtime['calls']:,} measured calls. Result: **{runtime['status']}**; maximum required-workload P95 {runtime['maximum_required_p95_ms']:.3f}ms. The original Stage1 P95≤40ms gate and overall integrated Layer2 P95≤100ms target remain unchanged.",'',
        '| Workload | P50 ms | P95 ms | P99 ms |','|---|---:|---:|---:|']
    for label,values in runtime['metrics'].items():a.append('| '+label+' | '+str(values.get('p50',values.get('median')))+' | '+str(values['p95'])+' | '+str(values['p99'])+' |')
    a+=['','Warm/cold, representative/dense, hard workloads, memory/RSS, full raw-history initialization, empty-JIT first startup, P99/max tails and all call rows are preserved in '+link(s1_release/'runtime.json')+' and its referenced evidence. A component runtime pass does not establish integrated Layer2 latency.','',
        'Reproduction and inference: '+link(DOC/'REPRODUCE.md')+'. Independent final audit: '+link(REVIEW/'STAGE1_REVIEW.json')+'. No further Stage1 development or collection is planned.']
    (DOC/'A_FINAL_STAGE1_RESEARCH_RELEASE.md').write_text('\n'.join(a)+'\n')
    b=['# Stage 2: Trajectory-Conditioned GraphDelta Predictor','',
        '**Stage 2 is the primary research development objective.** Its loadable trained checkpoint is '+link(s2_release/'checkpoint.pt')+' (SHA256 `'+sha(s2_release/'checkpoint.pt')+'`).','',
        'Inputs are the initial legal PartialGraph and executed [absolute_time,x,y,z,yaw] rows. The network predicts newly observed FREE/BLOCKED cells, explicit UNKNOWN/no-update probabilities and height residuals. The existing deterministic PartialGraph.update constructs the hypothetical future graph. The original map stays unchanged; predicted provenance is explicit. No Stage1 prediction is an input.','',
        '## Data and fixed offline training','',
        f"Training reuses all {c2['roots']:,} final eligible independent Roots / {c2['counts']['queries']:,} Query branches / {c2['scenes']} fitting scenes. Actual legal future-observation DAG ancestry determines executed duration/termination and observation-limited cell targets. No privileged geometry, future jaw-camera transform, pedestrian state, new simulation, planner-in-the-loop training or RL is used.",'',
        f"The shared cell MLP has {done2['parameters']:,} parameters, width48/32, UNKNOWN/FREE/BLOCKED softmax and a height head. Six fixed epochs, seed2410, AdamW lr0.001/decay0.0001, clip5 and batch8192. Up to64 new-information and64 remaining-UNKNOWN cells per Query use exact inverse-inclusion weights, with equal Root and within-Root Query weights. The sampled log score estimates full-domain risk; it does not count cells or siblings as independent trajectories. No development threshold or hyperparameter search was performed.",'',
        f"Completed {done2['steps']:,} optimizer steps in {done2['seconds']:.1f}s; peak CUDA allocation {done2['peak_cuda_bytes']:,} bytes. Exact config: "+link(s2_release/'config.json')+'. Immutable extraction/model/API/evaluation source: '+link(s2source/'manifest.json')+'.','',
        'Data extraction: '+link(BASE/'eventwm_graph/stage2/data.py')+'. Independent exact replay audits: '+link(REVIEW/'REVIEW_STAGE2_LEGAL_TARGETS.json')+' and '+link(REVIEW/'STAGE2_CORRECTNESS_REVIEW.json')+'.','',
        '## GraphDelta quality','',
        f"Every original development Query is evaluated: {m2['roots']:,} Roots / {m2['scenes']} scenes / {m2['counts']['queries']:,} outcomes. Raw-probability argmax and all scene/bootstrap rules were frozen before scoring. Candidate-domain misses: {m2['counts']['out_of_domain_positive']:,}. Deterministic graph-update correctness: {m2['graph_update_correctness']:.6f}.",'',
        '| New information | Precision | Recall | F1 | 95% scene-bootstrap F1 interval |','|---|---:|---:|---:|---|']
    for key in ('new_free','new_blocked'):
        value=m2['scene_balanced'][key];b.append('| '+key+' | %.6f | %.6f | %.6f | %s |'%(value['precision'],value['recall'],value['f1'],value['f1_ci95']))
    b+=['',f"Scene-balanced label-aware GraphDelta IoU: {m2['scene_balanced']['delta_cell_iou']:.6f}; reveal-set IoU: {m2['scene_balanced']['reveal_set_iou']:.6f}. Root-balanced multiclass Brier: {m2['multiclass_brier']:.8f}, versus persist-UNKNOWN {m2['persist_unknown_brier']:.8f}. Multiclass log score: {m2['multiclass_logloss']:.6f} (saved numerical floor1e-9). Height MAE on observed cells: {m2['height_mae_on_observed_cells']:.6f}m.",'',
        'Substantial-information trajectories (≥100 true new cells): `'+json.dumps(m2['substantial_information'])+'`.','',
        'Per-class reliability bins/ECE: `'+json.dumps(m2['calibration'])+'`. Complete pooled and scene-balanced metrics, support, confidence intervals, every Query confusion matrix and graph-update checks are retained in '+link(s2_release/'evaluation.json')+' and '+link(S2/'evaluation/optimized_dev')+'.','',
        '## Complete API performance','',
        '| Workload | Predict P50/P95/P99 ms | Predict + deterministic update + sparse JSON P50/P95/P99 ms |','|---|---|---|']
    for label,values in b2['workloads'].items():b.append('| '+label+' | '+json.dumps(values['predict'])+' | '+json.dumps(values['predict_apply_wire'])+' |')
    b+=['','Batch throughput and latency: `'+json.dumps(b2['batch'])+'`.','',
        f"Peak benchmark-process RSS: {b2['peak_rss_bytes']:,} bytes; checkpoint initialization {b2['checkpoint_load_seconds']:.6f}s. All representative/dense cold process observations, workload sizes, raw calls and memory data remain in "+link(s2_release/'performance.json')+'. These timings include domain/feature construction, neural work and delta construction; update/wire rows include deterministic graph copying/application and sparse serialization. Batch tails use eight observations and are descriptive. The overall integrated Layer2 P95≤100ms target is preserved; integration has not been benchmarked.','',
        'The original complete-call P95 failures (111.7ms representative /143.8ms dense) and original repository-namespace cold results are preserved in '+link(S2/'benchmarks/final')+'. Execution-only optimization reuses exact neighbor/pose features, tiles independent neural rows and transfers updater-owned data without redundant copies. Model architecture, trained checkpoint, candidate support and scoring protocol are unchanged. The optimized cold subprocess uses the frozen namespace from /tmp and records its actual import paths/hashes. Independent equivalence proof: '+link(REVIEW/'STAGE2_RUNTIME_EQUIVALENCE_REVIEW.json')+'. Complete original/optimized scientific-summary comparison: '+link(S2/'optimized_runtime/scientific_summary_comparison.json')+'.','',
        '## Limitations and API','',
        f"This first predictor models new information at initially UNKNOWN cells. The training ledger retains {c2['counts']['changed_known_cells']:,} per-branch known-cell change events as an unsupported scope; final known-cell correction is not predicted. Four nearest actual pose features can lose turn headings in tied-position rotations. Requested inputs omit articulated jaw-camera state and dynamic-human occluders, which leave irreducible reveal uncertainty. Observation-limited C-space certificates are not hidden material geometry or a safety certificate.",'',
        'Executable API: '+link(BASE/'eventwm_graph/stage2/api.py')+'. Loading, batch inference, delta application and exact reproduction: '+link(DOC/'REPRODUCE.md')+'. Independent final model/performance audit: '+link(REVIEW/'STAGE2_OPTIMIZED_FINAL_INDEPENDENT_REVIEW.json')+'. Stage1, L2.5, L3 and L4 implementations remain intact; no L3 RL training was started.']
    (DOC/'B_STAGE2_RESEARCH.md').write_text('\n'.join(b)+'\n')
    (DOC/'README.md').write_text('# Final research deliverables\n\n'+link(DOC/'A_FINAL_STAGE1_RESEARCH_RELEASE.md','A. Final Stage1 research release')+'\n\n'+link(DOC/'B_STAGE2_RESEARCH.md','B. Stage2 GraphDelta development')+'\n\n'+link(DOC/'PRIORITY_CHANGE.md')+' · '+link(DOC/'REPRODUCE.md')+'\n')
    paths=set()
    for directory in (s1_release,s2_release,inference,TRANSITION/'training_source',s2source,Path((S2/'source_path.txt').read_text().strip()),
                      S1/'experiments'/NAME,S1/'evaluation'/NAME,S1/'evaluation/final_stage1_current_20261024',
                      S1/'evaluation/final_stage1_reliability_20261024',S1/'evaluation/final_stage1_hpositive_reliability_20261024',
                      S1/'evaluation/final_stage1_exact_h0_20261024',S2/'training/final',S2/'evaluation/final_dev',S2/'benchmarks/final',
                      S2/'evaluation/optimized_dev',S2/'benchmarks/optimized',S2/'optimized_runtime',REVIEW):
        paths.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.nbc','.nbi'))
    for path in [S1/'final_collection_20261024/manifest.json',TRANSITION/'final_collection_all_splits/manifest.json',TRANSITION/'final_collection_all_splits/rows.jsonl.gz',
                 S2/'prepared/train/manifest.json',S2/'protocol.json',TRANSITION/'collector_final.json',BASE/'scripts/final_stage1_research.py',
                 BASE/'scripts/final_stage1_evaluate.py',BASE/'scripts/stage2_research.py',BASE/'scripts/freeze_complete_stage01.py',
                 BASE/'scripts/stage2_runtime_research.py',BASE/'scripts/extend_final_stage1_sidecars.py',BASE/'scripts/verify_final_research.py',
                 TRANSITION/'environment.json',TRANSITION/'package_versions.json',TRANSITION/'stage1_inference_config.json',
                 DOC/'A_FINAL_STAGE1_RESEARCH_RELEASE.md',DOC/'B_STAGE2_RESEARCH.md',DOC/'REPRODUCE.md',DOC/'PRIORITY_CHANGE.md',DOC/'README.md']:
        paths.add(path)
    artifacts=dict(schema='final_stage1_stage2_research_artifacts_1',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
        stage1_development='CLOSED',primary_development='Stage2 Trajectory-Conditioned GraphDelta Predictor',scientific_release_acceptance=False,
        collection='MANUALLY_STOPPED',files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)})
    dump(DOC/'ARTIFACTS.json',artifacts)
    for dest in (s1_release,s2_release):dest.chmod(0o555)
    dump(TRANSITION/'COMPLETE.json',dict(stage1=str(s1_release),stage2=str(s2_release),artifacts=str(DOC/'ARTIFACTS.json'),artifacts_sha256=sha(DOC/'ARTIFACTS.json')))
    print(json.dumps(dict(status='TWO_RESEARCH_DELIVERABLES_COMPLETE',stage1=str(s1_release),stage2=str(s2_release))),flush=True)


if __name__=='__main__':main()
