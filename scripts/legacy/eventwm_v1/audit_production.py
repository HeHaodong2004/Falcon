"""Read-only offline production audit. Never starts a simulator or collector."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
from collector.core import build_labels, digest, load_wm_input, read, write
from collector.observation import HumanDetection, LegalHumanTracker, ObservationConfig, TRACK_FIELDS
from eventwm_scale import verify_root

CATEGORIES = ('missing_assets', 'root_prefix_construction_failure',
              'static_contact_environment_termination', 'scene_config_incompatibility',
              'collection_process_failure', 'integrity_replay_failure')


def classify_failures(dataset, manifest):
    failures = []
    for root in manifest['roots']:
        if root['state'] != 'failed':
            continue
        attempt = root['attempts'][-1]
        path = dataset / attempt['path']
        logs = sorted(path.glob('audits/workers/*/run.log'))
        texts = [(p, p.read_text(errors='replace')) for p in logs]
        combined = '\n'.join(t for _, t in texts) + attempt.get('error', '')
        tags = []
        if any(s in combined for s in ('Root replay mismatch', 'Collector integrity audit failed', 'Canonical/worker mismatch')):
            category, reason = 'integrity_replay_failure', 'Collector integrity/replay assertion'
        elif 'Prefix terminated before root' in combined:
            category, reason = 'root_prefix_construction_failure', 'Environment episode ended during prefix; contact cause not persisted'
            tags = ['static_contact_environment_termination']
        elif 'No Stage Attributes exists' in combined:
            category, reason = 'missing_assets', 'Requested scene GLB unavailable; stage registration failed'
        elif 'Stage Template managed object handle `` not found' in combined:
            category, reason = 'scene_config_incompatibility', 'ReplicaCAD scene references unresolved stage/object/URDF assets; empty stage handle abort'
            tags = ['missing_assets']
        else:
            category, reason = 'collection_process_failure', 'Unclassified process failure; inspect evidence'
        evidence = []
        for p, content in texts:
            lines = content.splitlines()
            selected = [(i + 1, s) for i, s in enumerate(lines) if any(k in s for k in
                        ('Prefix terminated before root', 'No Stage Attributes exists',
                         'Stage Template managed object handle', 'cannot be found', 'Root replay mismatch',
                         'Collector integrity audit failed', 'Canonical/worker mismatch'))]
            evidence.extend(dict(path=str(p.relative_to(dataset)), line=i, text=s) for i, s in selected[-4:])
        failures.append(dict(root_id=root['root_id'], scene=root['scene'], human_count=root['human_count'],
                             category=category, secondary_tags=tags, reason=reason, evidence=evidence,
                             attempt=attempt))
    groups = Counter((r['scene'], r['human_count'], r['category'], r['reason']) for r in failures)
    return dict(roots=failures, category_counts={k: sum(r['category'] == k for r in failures) for k in CATEGORIES},
                grouped=[dict(scene=s, human_count=h, category=c, reason=r, count=n)
                         for (s, h, c, r), n in sorted(groups.items())])


def audit_root(dataset, root):
    path = dataset / root['path']
    checks = {}
    def check(name, ok):
        checks[name] = checks.get(name, True) and bool(ok)
    verify_root(dataset, root, deep=True)
    check('completion_marker_and_all_file_seals', True)
    saved = read(path / 'audits/audits.json')
    expected = {'query_immutability', 'same_root', 'same_query_reproducibility',
                'branch_order_independence', 'generator_privileged_boundary',
                'hidden_gt_not_in_legal_input', 'offline_label_recomputation',
                'truncation_censoring', 'timestamps', 'static_planner_human_blind'}
    check('sealed_collector_integrity_audits', expected <= saved.keys() and
          all(saved[k]['status'] == 'PASS' for k in expected))
    history = read(path / 'legal_observations/history.json')
    qs = read(path / 'queries/queries.json')
    byq = {q['query']['query_id']: q for q in qs}
    labels = read(path / 'labels/labels.json')
    bs = [json.loads(line) for line in (path / 'branch_truth/branches.jsonl').read_text().splitlines()]
    byb = {b['branch_id']: b for b in bs}
    primary = {f'b{i:03}' for i in range(len(qs))}
    check('branch_inventory', len(byb) == len(bs) == len(qs) + 3 and
          set(byb) == primary | {'reverse_0', 'reverse_1', 'truncated_wait'} and
          len(byq) == len(qs) and [byb[f'b{i:03}']['query_id'] for i in range(len(qs))] == list(byq))
    check('offline_label_recomputation', [build_labels(b) for b in bs] == labels)
    for i in (0, 1):
        a, b = byb[f'b{i:03}'], byb[f'reverse_{i}']
        check('same_query_reproducibility_and_order_independence',
              all(digest(a[k]) == digest(b[k]) for k in ('truth', 'controls', 'legal_observations')))
    trunc = next(l for l in labels if l['branch_id'] == 'truncated_wait')
    check('truncation_censoring', trunc['censored'] and trunc['full_query_safe'] is None)
    priv = read(path / 'roots/privileged_root.json')
    check('canonical_root_fingerprint', digest(priv['state']) == priv['sha256'])
    check('history_fingerprint', digest(history) == priv['state']['history_hash'])
    root_time = history[-1]['timestamp']
    check('causal_history_timestamps', np.all(np.diff([h['timestamp'] for h in history]) > 0) and
          root_time == priv['state']['timestamp'])
    with np.load(path / 'legal_observations/history_images.npz', allow_pickle=False) as images:
        rgb, depth = images['rgb'], images['depth']
        check('image_timestamps', np.array_equal(images['timestamps'], [h['timestamp'] for h in history]) and
              len(rgb) == len(depth) == len(history))
        image_hash = hashlib.sha256()
        for d, r in zip(depth, rgb):  # sorted sensor keys: depth then rgb
            image_hash.update(d.tobytes()); image_hash.update(r.tobytes())
        check('history_image_fingerprint', image_hash.hexdigest() == priv['state']['image_hash'])
    for b in bs:
        qrow = byq[b['query_id']]; q = qrow['query']
        check('query_immutability', b['query_hash_before'] == b['query_hash_after'] == qrow['sha256'] == digest(q))
        check('same_root', b['root_id'] == q['root_id'] == root['root_id'] and b['root_fingerprint'] == priv['sha256'])
        times = np.array([r['timestamp'] for r in b['truth']]); t0 = times[0]
        tol = q['clock_quantization_tolerance']
        check('timestamps_and_controls', t0 == root_time and np.all(np.diff(times) > 0) and
              np.max(np.diff(times)) <= .02400001 and
              all(c['index'] == i and abs(c['t_start'] - (t0 + i*q['control_dt'])) < tol and
                  abs(c['t_end'] - c['t_start'] - q['control_dt']) < tol and
                  c['issued_command'] == q['executable_controls'][i]
                  for i, c in enumerate(b['controls'])) and
              (not b['completed'] or abs(b['observed_duration'] - q['planned_duration']) < tol))
        check('full_horizon_coverage', not b['completed'] or
              (len(b['controls']) == len(q['executable_controls']) and b['termination_reason'] == 'query_completed'))
        check('branch_root_legal_state', all(b['legal_observations'][0][k] == history[-1][k]
              for k in ('timestamp', 'robot_position', 'robot_yaw', 'human_tracks', 'tracking_available')))
    # Diagnostic GT positions are used only to reconstruct already-gated legal detections for audit.
    # Nothing from this function is passed to a model.
    cfg = ObservationConfig(**read(path / 'legal_observations/observation_config.json'))
    for bid in ['root'] + list(byb):
        worker = path / 'audits/workers' / bid
        fp = read(worker / 'root_fingerprint.json')
        check('worker_root_fingerprints', fp['sha256'] == priv['sha256'] == digest(fp['state']))
        check('worker_execution_success', read(worker / 'execution.json')['returncode'] == 0)
        tracker = LegalHumanTracker(cfg)
        diagnostics = read(worker / 'observation_diagnostics.json')
        rebuilt = []
        for row in diagnostics:
            t = row['timestamp']
            ds = [HumanDetection(t, tuple(h['gt_position'])) for h in row['humans'] if h['detected']]
            tracks = tracker.update(t, ds)
            check('legal_tracker_replay_velocity_and_lost_memory', tracks == row['legal_tracks'])
            check('track_fields_and_causality', all(set(tr) == set(TRACK_FIELDS) and
                  tr['first_seen_time'] <= tr['last_seen_time'] <= t and
                  abs(tr['observation_age'] - (t - tr['last_seen_time'])) < 1e-9 for tr in tracks))
            rebuilt.append((t, tracks))
        check('prefix_future_causal_boundary', [(t, tr) for t, tr in rebuilt if t <= root_time] ==
              [(h['timestamp'], h['human_tracks']) for h in history])
        if bid != 'root':
            check('future_tracks_match_diagnostics', [(t, tr) for t, tr in rebuilt if t >= root_time] ==
                  [(h['timestamp'], h['human_tracks']) for h in byb[bid]['legal_observations']])
    for qid in byq:
        legal = load_wm_input(path, qid)
        check('legal_loader_history', legal['history'] == history)
    return dict(root_id=root['root_id'], scene=root['scene'], status='PASS' if all(checks.values()) else 'FAIL',
                checks=checks, primary_branches=len(qs), branches=len(bs), sealed_files=len(root['seals']),
                human_reactivity=saved.get('human_reactivity'),
                static_planner_note='Original PASS is SHA256-sealed; no simulator or static planner rerun')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=REPO / 'datasets/raw/eventwm_prod_v1')
    parser.add_argument('--output', type=Path, default=REPO / 'output/legacy/eventwm_tiny_overfit/audit.json')
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if dataset == args.output.resolve() or dataset in args.output.resolve().parents:
        parser.error('Audit output must be outside the raw dataset')
    manifest = read(dataset / 'manifest.json')
    report = dict(schema_version=1, dataset=str(dataset), manifest_sha256=hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest(),
                  failures=classify_failures(dataset, manifest), completed=[], status='RUNNING')
    write(args.output, report)
    for root in manifest['roots']:
        if root['state'] != 'completed':
            continue
        try:
            result = audit_root(dataset, root)
        except Exception as exc:
            result = dict(root_id=root['root_id'], status='FAIL', error=f'{type(exc).__name__}: {exc}')
        report['completed'].append(result)
        print(root['root_id'], result['status'], [k for k, v in result.get('checks', {}).items() if not v], flush=True)
        if result['status'] != 'PASS':
            report['status'] = 'FAIL'; write(args.output, report)
            raise SystemExit('STOP: completed root failed integrity; no training permitted')
        write(args.output, report)
    report['status'] = 'PASS'; write(args.output, report)
    print('AUDIT PASS', len(report['completed']), 'completed roots;', len(report['failures']['roots']), 'failed roots classified')


if __name__ == '__main__':
    main()
