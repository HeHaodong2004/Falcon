#!/usr/bin/env python3
"""Export compact, data-derived results for the revised large-data study."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import eventwm_v4_research as r
from eventwm_v4_scaling import DEFAULT, OLD
from eventquerywm.sr_metrics import ranking_metrics

WORKTREE = Path(__file__).resolve().parents[1]
SIZES = ('30k', '100k', '150k', 'all')
SEEDS = (20261012, 20261013)
STAGES = ('fixed', 'exposure', 'best')
SPLITS = ('train', 'validation', 'natural_reference', 'unseen_evaluation')
FIELDS = {'bce': 'bce', 'ap': 'collision_pr_auc_average_precision',
          'auc': 'collision_roc_auc', 'brier': 'brier_score', 'ece': 'collision_ece',
          'recall': 'collision_recall', 'false_safe': 'false_safe_rate_given_collision'}
FEATURES = ('Human features contain decision-frame position, estimated velocity, visibility/velocity validity, observation validity/age, physical clocks, time increments, presence, track age and observation count. Robot features contain world position, heading sine/cosine, decision-frame relative position, clocks, previous linear/angular command and tracker availability. Query features contain local position, physical time, velocity, heading, arclength, angular velocity and time fraction. Full arrived histories contain 11–221 observations; no image tensor, route-goal tensor or separate acquisition/arrival-latency schema enters this baseline.')


def csv_write(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def flatten(metrics):
    result = {name: metrics.get(field) for name, field in FIELDS.items()}
    result['within_path'] = metrics['within_group_ranking']['pair_concordance']
    result['within_root'] = metrics['within_root_ranking']['pair_concordance']
    return result


def fmt(value):
    return '—' if value is None else f'{value:.4f}'


def mean_metrics(runs, size, stage, split):
    values = [flatten(runs[f'{size}_full_seed{s}']['results'][stage]['splits'][split]) for s in SEEDS]
    return {k: None if any(v[k] is None for v in values) else float(np.mean([v[k] for v in values])) for k in values[0]}


def uncertainty(out, cache, split='validation'):
    """Paired uncertainty over evaluation scenes, not correlated candidate rows."""
    result = {}
    for stage in STAGES:
        predictions = {}
        indices = None
        for size in ('30k', 'all'):
            for seed in SEEDS:
                p = np.load(out/'runs'/f'{size}_full_seed{seed}'/f'{stage}_{split}.npz')
                if indices is None:
                    indices = p['indices']
                assert np.array_equal(indices, p['indices'])
                predictions[size, seed] = p['logits'].astype(float)
        rows = [cache.rows[i] for i in indices]
        y = np.array([v['sr_label'] for v in rows])
        scenes = sorted({v['scene'] for v in rows})
        blocks = [np.flatnonzero([v['scene'] == s for v in rows]) for s in scenes]

        def measures(z, ix):
            labels, z = y[ix], z[ix]
            pc = 1/(1+np.exp(np.clip(z, -700, 700)))
            return dict(bce=float(np.mean(np.logaddexp(0, z)-labels*z)),
                        ap=ranking_metrics(1-labels, pc)[0],
                        brier=float(np.mean((pc-(1-labels))**2)))

        rng = np.random.default_rng(20261012)
        differences = defaultdict(list)
        for _ in range(512):
            ix = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
            m = {k: measures(z, ix) for k, z in predictions.items()}
            for field in ('bce', 'ap', 'brier'):
                if any(v[field] is None for v in m.values()):
                    continue
                differences[field].append(float(np.mean([m['all', s][field]-m['30k', s][field] for s in SEEDS])))
        point = {k: measures(z, np.arange(len(y))) for k, z in predictions.items()}
        result[stage] = {field: dict(delta=float(np.mean([point['all', s][field]-point['30k', s][field] for s in SEEDS])),
                                    ci95=np.quantile(values, [.025, .975]).tolist()) for field, values in differences.items()}
    return dict(policy=f'512 paired physical-scene bootstrap replicates; average paired seed differences; {len(scenes)} scenes in {split}; exploratory intervals without multiplicity correction',
                results=result)


def figures(out, export, runs, cache):
    colors = dict(zip(SIZES, ('#d16c2d', '#379b74', '#9456ad', '#2470b0')))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for size in SIZES:
        curves = [r.read(out/'runs'/f'{size}_full_seed{seed}'/'curve.json') for seed in SEEDS]
        assert [v['step'] for v in curves[0]] == [v['step'] for v in curves[1]]
        for row, clock in enumerate(('step', 'epochs')):
            x = [v[clock] for v in curves[0]]
            for col, field, metric, style in ((0, 'validation', 'bce', '-'),
                                              (0, 'train_probe', 'bce', '--'),
                                              (1, 'validation', 'collision_pr_auc_average_precision', '-')):
                values = np.array([[v[field][metric] for v in c] for c in curves])
                axes[row, col].plot(x, values.mean(axis=0), color=colors[size], ls=style,
                                   label=('188k' if size == 'all' else size) if style == '-' else None)
                axes[row, col].fill_between(x, values.min(axis=0), values.max(axis=0), color=colors[size], alpha=.1)
    for row, label in enumerate(('Optimizer steps', 'Dataset passes / epochs')):
        axes[row, 0].set(xlabel=label, ylabel='BCE')
        axes[row, 1].set(xlabel=label, ylabel='Validation collision AP')
        for ax in axes[row]:
            ax.grid(alpha=.2)
    axes[0, 0].set_title('Two-seed means; shading: seed range; dashed: train probe')
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(export/'learning_curves.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, split in zip(axes, SPLITS[1:]):
        for name, path in [('30k', out/'runs/30k_full_seed20261012'/f'best_{split}.npz'),
                           ('188k', out/'runs/all_full_seed20261012'/f'best_{split}.npz'),
                           ('Gaussian CV', out/'cv_fast'/f'{split}.npz')]:
            p = np.load(path)
            risk = 1/(1+np.exp(np.clip(p['logits'].astype(float), -700, 700)))
            labels = np.array([1-cache.rows[i]['sr_label'] for i in p['indices']])
            bucket = np.minimum((risk*10).astype(int), 9)
            x, y = [], []
            for k in range(10):
                mask = bucket == k
                if mask.any():
                    x.append(risk[mask].mean()); y.append(labels[mask].mean())
            ax.plot(x, y, marker='o', ms=3, label=name)
        ax.plot([0, 1], [0, 1], ls='--', c='gray', lw=.8)
        ax.set(title=split.replace('_', ' '), xlabel='Predicted collision probability', xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Observed collision frequency')
    axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(export/'calibration.png', dpi=160); plt.close(fig)


def main(cli_args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args(cli_args)
    torch.set_num_threads(1)
    out = args.output.resolve()
    export = WORKTREE/'research_results/eventwm_v4_full_scaling_20261012'
    export.mkdir(parents=True, exist_ok=True)
    audit = r.read(out/'audit.json')
    cache = r.Cache(out)
    snapshot = r.read(out/'snapshot.json')
    runs = {p.parent.name: r.read(p) for p in (out/'runs').glob('*/results.json')}
    assert set(runs) == {f'{n}_full_seed{s}' for n in SIZES for s in SEEDS}
    cv = r.read(out/'cv_fast/results.json')
    planning = {}
    metrics_rows, category_rows, plan_rows, curve_rows, scaling_rows, cv_rows = [], [], [], [], [], []
    for name, run in runs.items():
        # Correct a descriptive string in runs launched before the revised label
        # was fixed. Actual membership and training were root-disjoint throughout.
        protocol = dict(run['protocol'], checkpoint_selection='minimum full held-out-root validation BCE', early_stopping_enabled=False)
        run['protocol'] = protocol
        curve = r.read(out/'runs'/name/'curve.json')
        exposure_by_step = {v['step']: v for v in curve}
        allowed = set(snapshot['subsets'][protocol['subset']])
        pool_indices = [i for i in cache.splits['train'] if cache.rows[i]['root_id'] in allowed]
        first_order = np.random.default_rng(protocol['seed']+999).permutation(pool_indices)
        for stage in STAGES:
            reports = run['results'][stage]
            consumed = exposure_by_step[reports['step']]
            presentations = consumed['sample_presentations']
            # Every first shuffled epoch is without replacement; after it, all
            # pool queries have necessarily appeared at least once.
            unique_seen = min(protocol['unique_train_queries'], presentations)
            seen_rows = [cache.rows[i] for i in first_order[:unique_seen]]
            seen_roots = len({v['root_id'] for v in seen_rows})
            seen_scenes = len({v['scene'] for v in seen_rows})
            tr, val = reports['splits']['train'], reports['splits']['validation']
            scaling_rows.append(dict(run=name, checkpoint=stage, step=reports['step'],
                unique_queries=protocol['unique_train_queries'], roots=protocol['train_roots'], scenes=protocol['train_scenes'],
                unique_seen_by_checkpoint=unique_seen, sample_presentations=presentations, epochs=consumed['epochs'],
                roots_seen_by_checkpoint=seen_roots, scenes_seen_by_checkpoint=seen_scenes,
                train_bce=tr['bce'], validation_bce=val['bce'], bce_gap=val['bce']-tr['bce'], **flatten(val)))
            for split in SPLITS:
                m = reports['splits'][split]
                metrics_rows.append(dict(run=name, checkpoint=stage, step=reports['step'],
                    unique_queries=protocol['unique_train_queries'], roots=protocol['train_roots'], scenes=protocol['train_scenes'],
                    split=split, count=m['count'], **flatten(m)))
                if split != 'train':
                    assert m['count'] == cv['reports'][split]['count']
                    wm, baseline = flatten(m), flatten(cv['reports'][split])
                    cv_rows.append(dict(run=name, checkpoint=stage, split=split,
                        **{k+'_wm_minus_cv': None if wm[k] is None or baseline[k] is None else wm[k]-baseline[k] for k in wm}))
                for category, cm in m['categories'].items():
                    category_rows.append(dict(run=name, checkpoint=stage, split=split, category=category,
                        count=cm['count'], **{k: cm.get(v) if k != 'false_safe' else (1-cm['collision_recall'] if cm['collision_recall'] is not None else None) for k, v in FIELDS.items()}))
            plan = r.read(out/'planning'/f'{name}_{stage}'/'results.json')
            planning[name, stage] = plan
            for policy, values in plan['comparisons']['same_path']['methods'].items():
                m = values['root_metrics']
                pc, clean = m['collision']['mean'], m['safe']['mean']
                plan_rows.append(dict(run=name, checkpoint=stage, policy=policy, collision=pc,
                    social_violation=m['social']['mean'], clean_safe=clean, social_success_given_noncollision=clean/(1-pc),
                    nominal_time=m['duration']['mean'], movement=m['movement']['mean'],
                    realized_nominal_cost=m['realized_nominal_cost']['mean']))
        for row in curve:
            curve_rows.append(dict(run=name, step=row['step'], epochs=row['epochs'], sample_presentations=row['sample_presentations'],
                seconds=row['wall_seconds'], train_probe_bce=row['train_probe']['bce'], validation_bce=row['validation']['bce'],
                validation_ap=row['validation']['collision_pr_auc_average_precision']))
    csv_write(export/'evaluation.csv', metrics_rows)
    csv_write(export/'scaling.csv', scaling_rows)
    csv_write(export/'gaussian_comparison.csv', cv_rows)
    for split in SPLITS:
        for category, cm in cv['reports'][split]['categories'].items():
            category_rows.append(dict(run='gaussian_cv', checkpoint='fixed_dynamics_train_only_calibration', split=split,
                category=category, count=cm['count'],
                **{k: cm.get(v) if k != 'false_safe' else (1-cm['collision_recall'] if cm['collision_recall'] is not None else None) for k, v in FIELDS.items()}))
    csv_write(export/'visibility.csv', category_rows)
    csv_write(export/'planning.csv', plan_rows)
    csv_write(export/'learning_curves.csv', curve_rows)
    csv_write(export/'run_costs.csv', [dict(run=name, wall_seconds=v['wall_seconds'],
        optimization_steps=v['stopped_at'], best_step=v['best_step'],
        unique_training_queries=v['protocol']['train_queries'],
        cuda_peak_allocated_bytes=v['peak_gpu_bytes'] or None) for name, v in runs.items()])
    census = {}
    subsets = {}
    for split in SPLITS:
        rows = [row for b in snapshot['blocks'] if b['split'] == split for row in b['rows']]
        census[split] = {field: dict(Counter(str(row.get(field)) for row in rows))
                         for field in ('sr_label', 'outcome_label', 'completed', 'category', 'cohort', 'source_commit', 'label_protocol')}
        census[split]['human_collision_queries'] = sum(v['outcome_label'] == 2 for v in rows)
        census[split]['static_collision_queries'] = 0
        census[split]['successor_duration_coverage'] = {str(h): sum(v['observed_duration_s'] >= h for v in rows) for h in (1., 3., 5.)}
    for size in SIZES:
        ids = set(snapshot['subsets'][size])
        rows = [row for b in snapshot['blocks'] if b['root']['root_id'] in ids for row in b['rows']]
        subsets[size] = dict(unique_queries=len({v['query_id'] for v in rows}), roots=len(ids), scenes=len({v['scene'] for v in rows}),
                             cohorts=dict(Counter(v['cohort'] for v in rows)))
    ci = uncertainty(out, cache)
    generalization_ci = {s: uncertainty(out, cache, s) for s in SPLITS[2:]}
    r.save(export/'scaling_uncertainty.json', ci)
    r.save(export/'generalization_uncertainty.json', generalization_ci)
    v4_scenes = len({b['root']['physical_scene_id'] for b in snapshot['blocks'] if b['cohort'] == 'V4'})
    v4_labels = Counter(v['sr_label'] for b in snapshot['blocks'] if b['cohort'] == 'V4' for v in b['rows'])
    r.save(export/'audit.json', dict(audit, census=census, subsets=subsets, validated_v4_scenes=v4_scenes,
            validated_v4_safe_queries=v4_labels[1], validated_v4_human_collision_queries=v4_labels[0],
            history_observations_per_root=dict(Counter(len(b['base'].robot) for b in cache.blocks))))
    r.save(export/'snapshot_identity.json', r.read(out/'snapshot_identity.json'))
    r.save(export/'configurations.json', {n: dict(v['protocol'], checkpoint_selection='minimum full held-out-root validation BCE',
                                               early_stopping_enabled=False) for n, v in runs.items()})
    r.save(export/'gaussian_cv.json', cv)
    r.save(export/'planning_deltas.json', {f'{n}_{s}': v['comparisons']['same_path']['new_minus_gaussian'] for (n, s), v in planning.items()})
    r.save(export/'checkpoint_manifest.json', {f'{n}_{s}': dict(path=v['checkpoint'], sha256=v['checkpoint_sha256'],
        snapshot_sha256=runs[n]['protocol']['snapshot_sha256'], step=runs[n]['results'][s]['step']) for (n, s), v in planning.items()})
    successors = {n: r.read(OLD/'successor_bounded'/n/'results.json') for n in ('30k', 'all')}
    r.save(export/'successor_diagnostics.json', dict(scope='Earlier strict-scene bounded decoder panel; not a 188k successor experiment',
                                                  outputs=str(OLD/'successor_bounded'), results=successors))
    figures(out, export, runs, cache)
    summary = dict(audit=audit, subsets=subsets, runs=runs, gaussian_cv=cv,
                   planning={f'{n}_{s}': dict(metrics={policy: value['root_metrics'] for policy, value in v['comparisons']['same_path']['methods'].items()},
                       new_minus_gaussian=v['comparisons']['same_path']['new_minus_gaussian'],
                       time_risk_new_minus_gaussian=v['comparisons']['same_path']['time_risk_new_minus_gaussian']) for (n, s), v in planning.items()},
                   uncertainty=ci, generalization_uncertainty=generalization_ci, snapshot_identity=r.read(out/'snapshot_identity.json'))
    r.save(export/'summary.json', summary)
    report = ['# EventQueryWM V4 scaling research — 12 October 2026', '']
    small = mean_metrics(runs, '30k', 'best', 'unseen_evaluation')
    full = mean_metrics(runs, 'all', 'best', 'unseen_evaluation')
    report += [f"Scaling improves unseen-scene generalization: validation-selected collision AP rises **{small['ap']:.4f} → {full['ap']:.4f}**, and within-path ranking **{small['within_path']:.4f} → {full['within_path']:.4f}**. Much of the gain is already present at 100k. Validation BCE improves, while validation AP does not improve consistently and two-pass training still overfits. The retrained selector approaches Gaussian CV on the frozen planning bank, with **no demonstrated collision-rate advantage**. Larger capacity is not justified by these results.", '']
    report += ['The revised study uses **188,265 unique training queries**, with nested 30k/100k/150k/full pools and two seeds at every size. Scene overlap is permitted for validation and natural-reference evaluation; roots, queries, episodes and physical-waypoint trajectory fixtures remain disjoint. Unseen evaluation scenes remain outside training. All eight models and all three checkpoint protocols are evaluated on the same frozen held-out queries and the same planning bank.', '']
    report += ['| Pool | Unique queries | Roots | Scenes | V4 queries | Compatible V3 queries |', '|---|---:|---:|---:|---:|---:|']
    for size, v in subsets.items():
        report.append(f"| {size} | {v['unique_queries']:,} | {v['roots']:,} | {v['scenes']} | {v['cohorts'].get('V4',0):,} | {v['cohorts'].get('V3',0):,} |")
    report += ['', f'The sealed V4 cutoff contains 198,957 retained queries / 6,029 completed roots / {v4_scenes} distinct physical scenes: {v4_labels[1]:,} safe completions and {v4_labels[0]:,} declared human collisions. After reserving 3,498 validation queries and preserving V4 natural-reference/unseen assignments, all 171,864 eligible V4 training queries are used. Matching schema 4, query schema 3, social-outcome-v3 and binary-execution-safety-v1 labels, deep seals, legal/query/label hashes and per-query digests permit adding 16,401 immutable V3 training queries. The 413 legacy reference/planning roots remain excluded. No valid training query is discarded solely for scene overlap. Invalid completed-record exclusions: **0**. Incomplete/unsealed roots are outside the frozen membership.', '']
    report += ['| Split | Queries | Roots | Scenes | SR safe | Human collision |', '|---|---:|---:|---:|---:|---:|']
    for split, v in audit['splits'].items():
        report.append(f"| {split} | {v['queries']:,} | {v['roots']} | {v['scenes']} | {census[split]['sr_label'].get('1',0):,} | {census[split]['sr_label'].get('0',0):,} |")
    report += ['', f"All pairwise root/query/episode/fixture/trajectory intersections are zero. Training shares {audit['scene_overlap_train_validation']} validation scenes and {audit['scene_overlap_train_natural']} natural-reference scenes, and zero unseen scenes. The protected planning bank retains 124 roots, 4,092 queries and 992 movement-matched same-path decisions. Waypoint grouping normalizes physical scene IDs and active human count; it checks shared physical trajectory fixtures, without claiming equality of every stochastic continuous realization.", '',
               'Raw archives are read through finalized symlinks under `/tmp/eventwm_v4_300k`, resolving to `/home/haodong/workspace/datasets/eventwm-falcon/raw/eventwm_v4_300k_archive/roots`. Read-only SQLite transactions freeze membership from the deep-verified index; training-facing files, completion/seals and query content digests are checked independently. The snapshot pins archived paths and hashes; raw data is not copied. The numerical cache is 418,436,222 bytes, with 18 canonical-loader parity checks covering six source epochs. Older static-contact epochs retain compatible input/label semantics and pass sealed validation; the audit does not rerun native simulation.', '',
               'SR=1 is safe completion and SR=0 is declared collision. Noncollision early termination is excluded from retained supervision. Audited collision labels are human contact; static contact count is zero, so human-collision metrics equal the collision metrics and static-specific AP is undefined. Exact split label/visibility/cohort distributions are in `research_results/eventwm_v4_full_scaling_20261012/audit.json`.', '']
    report += ['**Unchanged baseline and budgets.** EventQueryWM-M has 1,035,649 parameters, width 128 and four attention heads. Legal human histories have 13 features/observation and a two-layer GRU; robot odometry has 12 features/observation and mean/last pooling. Two scene-attention layers fuse track/robot tokens; 16 timed query tokens with ten features pass through two query-attention and two cross-attention layers. Mean/max query pooling and an MLP output the safe-execution logit. There is no static-map input tensor: geometry governs candidate feasibility. No future truth, privileged trajectory identity or outcome labels enter the model input.', '',
               '```mermaid', 'flowchart LR', ' H[Legal human history N×T×13] --> G[GRU 2×128]', ' R[Robot history T×12] --> P[Mean + last pooling]', ' G --> S[Scene attention ×2]', ' P --> S', ' Q[Timed candidate query 16×10] --> E[Query attention ×2]', ' S --> F[Cross attention ×2]', ' E --> F', ' F --> O[Mean + max; MLP; safe logit]', '```', '',
               'Every run uses uniform shuffled sampling without replacement, unweighted safe-positive BCE, AdamW LR 1e-4, weight decay .01, dropout .2, batch 64 and gradient clipping 1. Seeds are 20261012/20261013. Fixed-step comparison uses 1,024 steps; matched exposure uses exactly two passes, including smaller epoch-final batches. Two-pass steps are 938 / 3,126 / 4,688 / 5,884. Best validation BCE is also reported from evaluations every 256 steps and epoch boundaries. Early stopping is disabled to complete the exposure comparison. Best-checkpoint results involve more selection opportunities for larger runs and are supplementary to the two controlled budgets.', '']
    report += [FEATURES, '']
    report += ['**Scaling results.** Values below average the two seeds. AP is collision-positive average precision; ECE uses 15 equal-width bins. Recall uses collision probability ≥.5. Ranking is pair concordance on contrasting safe/collision timing variants sharing a root and spatial path; ties score .5 and WAIT is excluded. Full per-seed, per-checkpoint and per-split metrics, including recall and false-safe rates, are in `evaluation.csv`.', '',
               'Counts label the available optimization pool. At 1,024 steps, the 30k model has seen all 29,997 unique queries, while each larger-pool model has seen 65,536 unique queries. At two passes every model has seen its entire pool. Full-pool best checkpoints occur at steps 2,048/2,560 and have seen 131,072/163,840 unique queries, respectively. `scaling.csv` records exact unique queries, roots and scenes consumed, sample presentations and epochs for every checkpoint, reconstructed from the saved deterministic sampling configuration. Best-selection results must not be interpreted as full exposure to all 188k queries. Train BCE evaluates the full declared training pool; the two-pass train/validation gaps provide the fully exposed comparison.', '',
               '| Budget | Pool queries | Train roots | Train BCE | Val BCE | Gap | Collision AP | ROC-AUC | Brier | ECE | Recall | Within-path |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for stage in STAGES:
        for size in SIZES:
            v = mean_metrics(runs, size, stage, 'validation'); t = mean_metrics(runs, size, stage, 'train'); sub = subsets[size]
            report.append(f"| {stage} | {sub['unique_queries']:,} | {sub['roots']} | {fmt(t['bce'])} | {fmt(v['bce'])} | {fmt(v['bce']-t['bce'])} | {fmt(v['ap'])} | {fmt(v['auc'])} | {fmt(v['brier'])} | {fmt(v['ece'])} | {fmt(v['recall'])} | {fmt(v['within_path'])} |")
    report += ['', '![Learning curves](research_results/eventwm_v4_full_scaling_20261012/learning_curves.png)', '']
    report.append(f"Total recorded run time, including final held-out inference, is {sum(v['wall_seconds'] for v in runs.values())/60:.1f} minutes. Per-run timings and selected steps are in `run_costs.csv`. Training GPU peak memory is unavailable: the original recorder queried default GPU 0 while these runs used GPU 1. The recorder is corrected for future runs; zero readings are not treated as measured GPU 1 memory.")
    for stage in STAGES:
        a, b = mean_metrics(runs, '30k', stage, 'validation'), mean_metrics(runs, 'all', stage, 'validation')
        ap_ci = ci['results'][stage]['ap']['ci95']
        report.append(f"{stage.capitalize()}, full minus 30k: BCE {b['bce']-a['bce']:+.4f}, AP {b['ap']-a['ap']:+.4f} (exploratory paired scene-bootstrap 95% interval [{ap_ci[0]:+.4f}, {ap_ci[1]:+.4f}]), within-path ranking {b['within_path']-a['within_path']:+.4f}.")
    report += ['', '**Generalization and Gaussian CV.** The next table averages validation-selected models across seeds. CV retains its frozen Gaussian dynamics and 64-clock exposure; only a monotone probability mapping is fitted on the full optimization pool, with no held-out calibration. Its calibration sees more training labels than the small WM models; exposure ranking itself has no learned calibration. The planning bank separately preserves its old CV calibration byte for byte.', '',
               '| Model | Val BCE / AP | Natural BCE / AP | Unseen BCE / AP | Val ranking | Natural ranking | Unseen ranking |', '|---|---:|---:|---:|---:|---:|---:|']
    for size in SIZES:
        m = [mean_metrics(runs, size, 'best', s) for s in SPLITS[1:]]
        report.append(f"| {size} | {fmt(m[0]['bce'])} / {fmt(m[0]['ap'])} | {fmt(m[1]['bce'])} / {fmt(m[1]['ap'])} | {fmt(m[2]['bce'])} / {fmt(m[2]['ap'])} | {fmt(m[0]['within_path'])} | {fmt(m[1]['within_path'])} | {fmt(m[2]['within_path'])} |")
    m = [flatten(cv['reports'][s]) for s in SPLITS[1:]]
    report.append(f"| Gaussian CV | {fmt(m[0]['bce'])} / {fmt(m[0]['ap'])} | {fmt(m[1]['bce'])} / {fmt(m[1]['ap'])} | {fmt(m[2]['bce'])} / {fmt(m[2]['ap'])} | {fmt(m[0]['within_path'])} | {fmt(m[1]['within_path'])} | {fmt(m[2]['within_path'])} |")
    report += ['', '![Calibration: seed 12, validation-selected checkpoints](research_results/eventwm_v4_full_scaling_20261012/calibration.png)', '',
               'Partial visibility remains a separate diagnostic: `visibility.csv` reports visible, lost, visible_lost, unseen-human, observed_expired and zero-human categories where present. “Unseen-human” visibility is different from unseen-scene evaluation. These categories are not matched visibility interventions, so differences alone do not establish causation.', '']
    for split, estimates in generalization_ci.items():
        d = estimates['results']['best']['ap']
        report.append(f"{split.replace('_', ' ').capitalize()}, best checkpoints: full-minus-30k AP {d['delta']:+.4f}, paired scene-bootstrap 95% interval [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]. See `generalization_uncertainty.json` for both controlled budgets and BCE/Brier intervals; these exploratory intervals do not adjust for multiple comparisons.")
    report += ['**Frozen planning bank.** Same-path decisions preserve candidate sets, movement, timing, decision rules and score/duration/query-ID tie breaks. Collision and clean-safe rates are root-weighted. Social success below is clean-safe among noncollisions, rather than closed-loop task success. The old model was selected on this development bank, which limits comparisons involving that model.', '',
               '| Model / checkpoint | Collision | Clean-safe | Social success | Nominal time (s) |', '|---|---:|---:|---:|---:|']
    reference = planning['all_full_seed20261012', 'best']['comparisons']['same_path']['methods']
    for policy in ('fastest', 'gaussian_cv', 'old_eventwm'):
        m = reference[policy]['root_metrics']; pc = m['collision']['mean']; clean = m['safe']['mean']
        report.append(f"| {policy} | {pc:.4f} | {clean:.4f} | {clean/(1-pc):.4f} | {m['duration']['mean']:.4f} |")
    for seed in SEEDS:
        for stage in STAGES:
            m = planning[f'all_full_seed{seed}', stage]['comparisons']['same_path']['methods']['new_eventwm']['root_metrics']
            pc, clean = m['collision']['mean'], m['safe']['mean']
            report.append(f"| 188k seed {seed%100}, {stage} | {pc:.4f} | {clean:.4f} | {clean/(1-pc):.4f} | {m['duration']['mean']:.4f} |")
    report += ['', 'All sizes, seeds and checkpoints, including the pre-existing nominal-time + 10×collision-probability policy, appear in `planning.csv`; paired root-bootstrap differences appear in `planning_deltas.json`. No selector penalty or heuristic is tuned for the new model.', '']
    for seed in SEEDS:
        delta = planning[f'all_full_seed{seed}', 'best']['comparisons']['same_path']['new_minus_gaussian']
        d = delta['collision']; t = delta['duration']
        report.append(f"188k best seed {seed%100}: collision difference versus CV {d['mean']:+.4f}, paired root-bootstrap 95% interval [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]; nominal-time difference {t['mean']:+.4f}s, interval [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}].")
    assert all(v['comparisons']['same_path']['methods']['new_eventwm_time_risk']['selected_query_ids'] ==
               v['comparisons']['same_path']['methods']['fastest']['selected_query_ids'] for (name, stage), v in planning.items() if name.startswith('all_'))
    report += ['The collision intervals include zero for both selected full-pool seeds; a planning advantage over CV is not established. At the unchanged time-risk coefficient, every full-pool checkpoint selects exactly the fastest bank candidates. This fixed rule therefore provides no additional risk/time tradeoff on this bank. No penalty is retuned after inspecting outcomes.', '']
    report += ['', '**History and successor diagnostics.** Earlier retrained ablations used the smaller, strict-scene 46,629-query pool: full validation AP .596 versus query-only .443; history-only AP .602 but within-path ranking .500. These results support use of human conditioning for aggregate classification and expose weak candidate conditioning; they are not full-pool retrained ablations. The full-pool best checkpoints additionally save inference-time masks in `summary.json`; masking is a distribution shift rather than an independently trained diagnostic.', '',
               'The preserved successor panel uses 332 versus 512 training roots, 106 validation roots and 64 roots each for natural/unseen testing, with one preselected b000 branch/root. Existing physical/observed occupancy head topologies use a frozen strict-study backbone, with future truth only in target tensors. At the best decoder checkpoint, physical 1s AP rises from .0128 to .0155 while train AP is .3250 / .2017; Gaussian CV is about .0758. At 3s, full-panel neural AP is .0110, below CV about .0145. Late optimization drives train 1s AP above .64 while validation falls below .011. This bounded panel does not establish full-pool successor scaling. There are no valid 5s targets in this b000 panel; no 5s prediction claim is made. Occupancy grids lack identity association for ADE/FDE and do not represent a complete physical state. Observed-state occupancy results and target masks are preserved in `successor_diagnostics.json`.', '']
    report += ['| Full-pool seed | Inference inputs | Val BCE | Val AP | Within-path |', '|---|---|---:|---:|---:|']
    for seed in SEEDS:
        values = runs[f'all_full_seed{seed}']['results']['best']['splits']
        for label, key in [('Full', 'validation'), ('Human tokens masked', 'validation_inference_query_only'),
                           ('Candidate query zeroed', 'validation_inference_history_only')]:
            m = flatten(values[key])
            report.append(f"| {seed%100} | {label} | {fmt(m['bce'])} | {fmt(m['ap'])} | {fmt(m['within_path'])} |")
    report += ['', 'Removing dynamic human tokens tests all human conditioning, including current position/velocity, rather than the incremental value of earlier observations. A latest-observation-only retrained control would be needed to establish benefit from multi-frame temporal history. Near-chance query-zeroed ranking can contain tiny floating-point differences between batches.', '']
    report += ['**Research answers and next experiment.**', '']
    report += [f"Q1: **Yes for generalization, with diminishing returns beyond 100k; no consistent validation-AP or planning win.** Unseen AP increases {small['ap']:.4f} → {full['ap']:.4f}, but the 100k/150k means are higher than the full-pool mean. The larger dataset helps without guaranteeing monotonic improvement."]
    for stage in STAGES:
        a, b = mean_metrics(runs, '30k', stage, 'validation'), mean_metrics(runs, 'all', stage, 'validation')
        report.append(f"{stage.capitalize()} budget: validation BCE {a['bce']:.4f} → {b['bce']:.4f}, AP {a['ap']:.4f} → {b['ap']:.4f}, ranking {a['within_path']:.4f} → {b['within_path']:.4f}.")
    a = mean_metrics(runs, 'all', 'exposure', 'train'); b = mean_metrics(runs, 'all', 'exposure', 'validation')
    sg = mean_metrics(runs, '30k', 'exposure', 'validation')['bce']-mean_metrics(runs, '30k', 'exposure', 'train')['bce']
    report += [f"Q2: **The gap decreases but overfitting persists.** Full-pool two-pass train/validation BCE is {a['bce']:.4f}/{b['bce']:.4f}; its gap is {b['bce']-a['bce']:.4f}, versus {sg:.4f} at 30k. Best-checkpoint selection limits late-training degradation.",
               'Q3: the existing retrained ablations demonstrate aggregate benefit from human conditioning, but candidate-specific use remains weak and the incremental value of multi-frame history is not isolated. Full-pool inference masks provide additional evidence, not a causal replacement for full-pool retraining.',
               'Q4: no controlled larger-capacity run was justified before completing this study. The current backbone fits training data; these experiments do not establish that parameter count limits held-out decision quality.',
               'Q5: the bounded occupancy decoders generalize poorly and lose to CV at 1s. Meaningful structured future-state prediction is not demonstrated.',
               'Q6: **No demonstrated win over Gaussian CV.** Selected full-pool models have slightly lower collision point estimates with intervals spanning zero; nominal times are similar. The fixed time-risk rule collapses to fastest.',
               'Q7: the highest-value next experiment is a controlled same-path timing-pair training objective added to the unchanged full-pool backbone, compared with BCE-only under matched exposure and the frozen selector. Hold data/features/selector fixed and select the auxiliary weight on training/validation only. This directly tests the gap between aggregate classification and counterfactual decision quality before increasing model capacity.', '']
    production = r.production_sample()
    r.save(out/'production_final.json', production)
    before = r.read(out/'production_before.json')
    fingerprints = r.read(OLD/'production_source_fingerprints.json')
    final_hashes = {name: r.sha(Path(fingerprints['science'])/name) for name in fingerprints['hashes']}
    assert final_hashes == fingerprints['hashes'], 'Production scientific source changed'
    for field in ('runner_pid', 'source_commit', 'native_limit', 'pipeline_settings'):
        assert before[field] == production[field], (field, 'Production settings changed')
    assert production['runner_alive'] and production['state'] == 'running'
    r.save(export/'production_isolation.json', dict(before=before, after=production,
        source_hashes_unchanged=True, hashes=final_hashes, collector_controls_invoked=False,
        raw_dataset_writes=False, manifests_modified=False))
    report += ['**Isolation and reproduction.** Training/evaluation run sequentially on CUDA GPU 1 with one PyTorch CPU thread, zero loader workers, nice 15 and idle I/O priority. CUDA was tested through PyTorch outside the sandbox; no driver, environment or collector changes were made. Earlier optional CPU jobs were stopped conservatively when publication slowed. Publication/resource logs remain under the experiment output. Production runner PID and source were checked throughout; a publication guard only stops research.', '',
               f"Frozen snapshot: `{out/'snapshot.json'}`\n\nSHA256: `{r.read(out/'snapshot_identity.json')['sha256']}`\n\nCheckpoints/predictions: `{out/'runs'}`\n\nPlanning: `{out/'planning'}`\n\nSuccessor diagnostics: `{OLD/'successor_bounded'}`\n\nWorktree: `{WORKTREE}`\n\nBranch: `research/eventwm-v4-200k-scaling-20261012`; base `d4a718275f96f1875b8afd1272a9e9a2fb729570`. The delivery commit hash is reported in the final response and Git log; datasets and checkpoints are excluded from Git.", '',
               'Run from the research worktree using the preserved frozen output. The suite skips completed runs, and evaluation skips completed bank results. To rerun from scratch, use a new output directory populated with the unchanged snapshot/cache manifests and shards; do not run `snapshot` again to reproduce this cutoff.', '', '```bash',
               'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1',
               f"PY=/home/haodong/miniforge3/envs/eventwm-falcon/bin/python",
               f"OUT={out}",
               'nice -n 15 ionice -c 3 "$PY" -u scripts/eventwm_v4_scaling.py suite --output "$OUT" --device cuda:1',
               'nice -n 15 ionice -c 3 "$PY" -u scripts/eventwm_v4_scaling.py evaluate --output "$OUT" --device cuda:1',
               'nice -n 15 ionice -c 3 "$PY" -u scripts/eventwm_v4_scaling.py report --output "$OUT"',
               '"$PY" -B -m unittest discover -s tests -p test_v4_research.py -q', '```', '',
               'The original membership/cache commands were `scripts/eventwm_v4_scaling.py snapshot` and `prepare` with this output. Re-running membership capture while collection advances creates a different study, so the exact archived frozen manifest is the reproduction authority.']
    (WORKTREE/'EVENTWM_V4_SCALING_RESEARCH_20261012.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(report=str(WORKTREE/'EVENTWM_V4_SCALING_RESEARCH_20261012.md'), exports=str(export), models=len(runs))))


if __name__ == '__main__':
    main()
