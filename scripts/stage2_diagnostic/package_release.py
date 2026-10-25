"""Package local research evidence; never publishes or modifies model inference."""
import gzip
import json
import os
from pathlib import Path
import re
import shutil
from .common import ROOT,OUT,CHECKPOINT,CHECKPOINT_SHA,digest,save

def main():
    dest=ROOT/'docs/l2/stage2_diagnostic_20261025';art=dest/'artifacts';art.mkdir(parents=True,exist_ok=True)
    mapping={
        'protocol.json':'protocol.json','precision_contract.json':'precision_contract.json',
        'source_manifest.json':'source_manifest.json','environment.json':'environment.json',
        'reports/package_link_check.json':'package_link_check.json','reports/regression_tests.json':'regression_tests.json',
        'probability/summary.json':'probability_summary.json','probability/conditional_head.json':'conditional_head.json',
        'probability/complete.json':'probability_complete.json',
        'trajectory/protocol_v1.json':'trajectory_protocol.json','trajectory/availability_v1/statistics.json':'trajectory_availability.json',
        'trajectory/implementation_addendum_v1.json':'trajectory_implementation_addendum.json',
        'trajectory/cache_artifact_addendum_v1.json':'trajectory_cache_addendum.json',
        'trajectory/primary_v1/summary.json':'trajectory_summary.json','trajectory/primary_v1/identity.json':'trajectory_identity.json',
        'trajectory/primary_v1/neutral_scene_weighted.json':'neutral_scene_weighted.json',
        'trajectory/primary_v1/sensitivity_scene_weighted.json':'sensitivity_scene_weighted.json',
        'trajectory/mixture_v1/summary.json':'mixture_summary.json',
        'visibility/full_v1/summary.json':'visibility_full_summary.json','visibility/full_v1/complete.json':'visibility_full_complete.json',
        'visibility/calibration_protocol_v2.json':'visibility_protocol.json',
        'visibility/soft_paired_v1/summary.json':'soft_geometry_summary.json','visibility/soft_paired_v1/protocol.json':'soft_geometry_protocol.json',
        'decision_relevance/counts.json':'decision_counts.json','decision_relevance/pairwise_counts.json':'pairwise_counts.json',
        'decision_relevance/protocol_addendum_v1.json':'decision_protocol.json',
        'decision_relevance/spatial_precision_addendum_v1.json':'spatial_addendum.json',
        'decision_relevance/soft_baseline_addendum_v1.json':'decision_soft_addendum.json',
        'visibility/decision_overhead_v3/metrics.json':'conversion_runtime.json',
        'visibility/full_adapter_latency_v3/metrics.json':'full_adapter_runtime.json',
        'visibility/OVERHEAD_VERIFICATION_V3.json':'runtime_verification.json',
        'visibility/decision_protocol_v3.json':'conversion_runtime_protocol.json',
        'visibility/full_adapter_protocol_v3.json':'full_runtime_protocol.json',
        'numerical/backend_threshold_v1/results.json':'backend_threshold_check.json',
        'trajectory/independent_probability_audit_v1/proof.json':'independent_threshold_proof.json',
        'trajectory/soft_geometry_independent_review.json':'independent_soft_geometry.json',
        'trajectory/decision_independent_review_v1/proof.json':'independent_decision_proof.json',
        'trajectory/overhead_independent_review.json':'independent_runtime_proof.json',
        'trajectory/SCIENTIFIC_REVIEW_RECEIPT.json':'scientific_review_receipt.json'}
    if (OUT/'final_confirmation/complete.json').exists():
        for folder,names in {
            'final_confirmation':('protocol.json','summary.json','complete.json','exposure_receipt.json',
                'FINAL_CONFIRMATION_REVIEW_RECEIPT.json','independent_review/proof.json',
                'independent_review/root_receipts.json'),
            'final_confirmation_metadata':('metadata_protocol.json','pre_manifest_selection.json',
                'selected_manifest.json','availability.json','receipt.json','availability_report_receipt.json')
        }.items():
            for name in names:mapping[folder+'/'+name]=folder+'/'+name
        mapping['reports/final_confirmation_tests.json']='final_confirmation_tests.json'
        mapping['reports/release_regression_tests.json']='release_regression_tests.json'
    for source,target in mapping.items():
        path=OUT/source
        if not path.exists():raise FileNotFoundError(path)
        (art/target).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,art/target)
    reports=[('reports/A_probability_thresholds.md','THRESHOLD_ANALYSIS.md'),
                          ('trajectory/B_TRAJECTORY_AUDIT.md','TRAJECTORY_AUDIT.md'),
                          ('trajectory/SCIENTIFIC_REVIEW.md','SCIENTIFIC_REVIEW.md'),
                          ('reports/RUNTIME.md','RUNTIME.md'),('reports/D_decision_relevance.md','DECISION_RELEVANCE.md')]
    if (OUT/'final_confirmation/complete.json').exists():
        reports.append(('final_confirmation/FINAL_CONFIRMATION_REVIEW.md','FINAL_CONFIRMATION_REVIEW.md'))
        shutil.copy2(ROOT/'reviews/stage2_segment/reviewer_b/FINAL_CONFIRMATION_PREFLIGHT.json',art/'final_confirmation/preflight.json')
        with open(art/'final_confirmation/roots.jsonl.gz','wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
                for p in sorted((OUT/'final_confirmation/roots').glob('*.json')):
                    gz.write(json.dumps(json.loads(p.read_text()),separators=(',',':')).encode()+b'\n')
    for source,target in reports:
        old=OUT/source;new=dest/target;shutil.copy2(old,new)
        original=art/'original_reports'/target;original.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,original)
        # Keep copied reports navigable: their original links were relative to
        # the output report directory, rather than this committed package.
        def rewrite(match):
            value=match.group(2)
            if value.startswith(('http:','https:','#')):return match.group(0)
            source_path=(old.parent/value).resolve()
            if not source_path.is_file():return match.group(0)
            relative=None
            for src,dst in mapping.items():
                if source_path==(OUT/src).resolve():relative=art/dst;break
            if relative is None:
                if source_path.is_relative_to(OUT/'reports/figures'):
                    relative=dest/'figures'/source_path.relative_to(OUT/'reports/figures')
                elif source_path.is_relative_to(OUT):
                    relative=art/'details'/source_path.relative_to(OUT)
                    relative.parent.mkdir(parents=True,exist_ok=True)
                    if relative.exists():
                        if digest(relative)!=digest(source_path):raise ValueError('conflicting archived evidence: '+str(relative))
                    else:shutil.copy2(source_path,relative)
                else:return match.group(0)
            return match.group(1)+os.path.relpath(relative,new.parent)+match.group(3)
        new.write_text(re.sub(r'(\[[^\]]*\]\()([^\)\n]+)(\))',rewrite,new.read_text()))
        if target in ('THRESHOLD_ANALYSIS.md','TRAJECTORY_AUDIT.md','SCIENTIFIC_REVIEW.md','DECISION_RELEVANCE.md'):
            new.write_text('Development-phase evidence. The later, separately frozen protected-scene check is documented in [FINAL_CONFIRMATION.md](FINAL_CONFIRMATION.md). Historical sealed-status statements describe the development phase.\n\n'+new.read_text())
    shutil.copy2(OUT/'reports/thresholds.csv',art/'thresholds.csv')
    table=art/'thresholds.csv';table.write_bytes(table.read_bytes().replace(b'\r\n',b'\n'))
    shutil.copytree(OUT/'reports/figures',dest/'figures',dirs_exist_ok=True)
    shutil.copytree(OUT/'trajectory/primary_v1/figures',dest/'figures/trajectory',dirs_exist_ok=True)
    for folder in ('decision_overhead_v3','full_adapter_latency_v3'):
        for name in ('calls.json','proofs.json'):
            shutil.copy2(OUT/'visibility'/folder/name,art/(folder+'_'+name))
    a=json.loads((art/'probability_summary.json').read_text());c=json.loads((art/'visibility_full_summary.json').read_text())
    soft=json.loads((art/'soft_geometry_summary.json').read_text())
    lines=['# C — Legal baseline comparison','',
        'All baselines use the starting legal graph, supplied saved poses/timestamps and recorded sensor parameters. No hidden occupancy, navmesh, future camera state, future observations or simulator execution enters predictions. The sensor uses articulated-camera geometry and C-space observation rules; the controls below are explicitly two-dimensional base-yaw opportunity surrogates. They are not exact optical visibility or physical upper bounds.','',
        'The no-revelation predictor returns an empty increment: labeled IoU0 and FREE/BLOCKED recall/F1=0. Revelation precision is mathematically undefined with no positives (the legacy rate implementation reports0). Its deliberately degenerate probabilities [1,0,0] and clipped proper scores are retained in probability_summary.json. A stronger constant probability prior is fitted on70scenes and evaluated on38.','',
        'The strict frontier requires a known-FREE grid-supercover prefix, stopping at the first UNKNOWN/BLOCKED cell. It never marks UNKNOWN FREE. The less restrictive frustum control estimates only horizontal FOV/range opportunity, allowing unknown/blocked intervening C-space cells without claiming that they are transparent or free. Both return potential-revelation masks only.','',
        '| Full47,128-window revelation control | Precision | Recall | F1 | Revelation IoU |','|---|---:|---:|---:|---:|']
    for name,r in c['raw_binary'].items():lines.append(f"|{name}|{r['precision']:.6f}|{r['recall']:.6f}|{r['f1']:.6f}|{r['revelation_iou']:.6f}|")
    r=a['rules']['threshold_0.20'];lines.append(f"|Stage2 selected0.20|{r['revelation_precision']:.6f}|{r['revelation_recall']:.6f}|{r['revelation_f1']:.6f}|{r['revelation_iou']:.6f}|")
    lines+=['', 'No labeled-map IoU is assigned to geometry controls: they cannot legitimately specify FREE/BLOCKED occupancy from unseen geometry. This prevents inventing a full-map baseline score.','',
        'A three-category probability calibration (outside frustum / frustum only / strict frontier) fits revelation frequencies only on70selectionscenes. On the other38scenes, Stage2 binary log loss/Brier/countMAE are0.0139307/0.00368273/73.511cells, versus0.0253104/0.00501942/176.061 for calibrated geometry. Paired geometry-minus-model CIs exclude zero: LL[0.010446,0.012359], countMAE[96.007,109.130]. ModelAP0.425933 exceeds three-categoryAP0.020106 at prevalence0.005121. All2110Roots and47128windows passed matching/source checks.','',
        'A stronger supplementary control uses the fixed1705same-start/horizon pairs. Its target-free continuous opportunity score sums saved-pose time increments times range and angular weights. It assigns no occupancy labels and never assumes UNKNOWN FREE. A33-category monotone PAV calibration is fitted only on1136pairs/70scenes;569pairs/38scenes are evaluated. Raw continuous-score rank metrics are reported separately from coarse calibrated ranks.','',
        '| Paired validation, identical fixed union | AUROC | AP | Binary log loss | Binary Brier | Count MAE |','|---|---:|---:|---:|---:|---:|']
    m=soft['metrics']['stage2_direct'];rank=soft['rank']['stage2_direct']
    lines.append(f"|Stage2 direct-U|{rank['auroc']:.6f}|{rank['auprc_ap']:.6f}|{m['binary_logloss']:.8f}|{m['binary_brier']:.8f}|{m['count_mae']:.3f}|")
    rank=soft['rank']['raw_continuous_geometry'];m=soft['metrics']['calibrated_geometry']
    lines.append(f"|Raw continuous geometry (ranking only)|{rank['auroc']:.6f}|{rank['auprc_ap']:.6f}|—|—|—|")
    rank=soft['calibrated_geometry_rank']
    lines.append(f"|PAV-calibrated geometry|{rank['auroc']:.6f}|{rank['auprc_ap']:.6f}|{m['binary_logloss']:.8f}|{m['binary_brier']:.8f}|{m['count_mae']:.3f}|")
    lines+=['', 'The paired geometry-minus-model log-loss CI is[0.008294,0.010253] and countMAE CI[51.356,78.293]. Direct-U includes values outside each individual natural candidate support; the primary trajectory-conditioning proof additionally uses the common in-support intersection. Default production-U comparisons are retained separately in the artifact and closely agree. No protected confirmation or cross-planner transfer is claimed.','',
        'The continuous scorer is label-free. Its PAV calibration uses revelation labels from the selection partition; calling the whole calibrated control label-free would be incorrect. Independent isotonic reconstruction reproduced the calibration, validation metrics and CIs to1.07e-14.','',
        'Stage2 outperforms the tested legal geometry controls for revelation probability/count prediction. Neural-network necessity is not established: an independently trained model without future path information, and stronger learned non-neural alternatives, remain unmeasured. See MISSING_BASELINES.md.','']
    (dest/'BASELINES.md').write_text('\n'.join(lines).replace(
        'No protected confirmation or cross-planner transfer is claimed.',
        'These are development comparisons; the later frozen calibration check is in FINAL_CONFIRMATION.md. Cross-planner transfer remains unmeasured.'))
    b=json.loads((art/'trajectory_summary.json').read_text());gallery=['# Trajectory audit examples','',
        'Six cases were selected by fixed hash before inspecting losses. Six others are explicitly outcome-selected extremes: three highest and three lowest conditioning gains. They are not a representative random sample. Both paths have recorded legal targets. The stationary neutral input is an OOD ablation with no recorded counterfactual ground truth. Threshold0.20 panels are diagnostics, not production.','']
    for group,ids in b['visual_selection'].items():
        if not isinstance(ids,list):continue
        for identifier in ids:
            gallery.extend([f'## {group}: {identifier[:12]}','',f'[PDF](figures/trajectory/{identifier}.pdf)','',f'![{group} {identifier[:12]}](figures/trajectory/{identifier}.png)',''])
    (dest/'GALLERY.md').write_text('\n'.join(gallery))
    if digest(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('checkpoint changed')
    save(art/'artifact_manifest.json',dict(checkpoint_sha256=CHECKPOINT_SHA,neural_training=False,production_changed=False,
        files={str(p.relative_to(dest)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'},
        large_caches_location=str(OUT),large_cache_note='Full float32 cell probabilities/labels and legal paired map caches retained locally; not committed as dense Git data'))
    print(dest)

if __name__=='__main__':main()
