"""Reference-only index and legal loader for horizon-complete primary branches."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
from collector.core import digest, load_wm_input, read, write


def exclusion_reason(label, query):
    if label['termination_reason'] == 'collection_error':
        return 'collection_error'
    if label['censored'] or not label['completed']:
        return 'censored_incomplete_future'
    if label['termination_reason'] != 'query_completed':
        return 'ambiguous_termination'
    if abs(label['planned_duration'] - query['planned_duration']) > 1e-9 or abs(
            label['observed_duration'] - query['planned_duration']) >= query['clock_quantization_tolerance']:
        return 'invalid_full_horizon'
    if label['event_observed'] != label['human_collision_event']:
        return 'ambiguous_event_coverage'
    if label['human_collision_event']:
        t = label['first_event_time']
        if t is None or not 0 <= t <= min(label['observed_duration'], query['planned_duration']):
            return 'event_outside_requested_horizon'
    elif label['full_query_safe'] is not True or label['first_event_time'] is not None:
        return 'not_clean_safe'
    return None


def composition(rows):
    return dict(samples=len(rows), roots=len({r['root_id'] for r in rows}),
                positive=sum(r['event_label'] for r in rows), negative=sum(not r['event_label'] for r in rows),
                visible_history_samples=sum(r['visible_human_history'] for r in rows),
                visible_at_root_samples=sum(r['visible_tracks_at_root'] > 0 for r in rows),
                lost_at_root_samples=sum(r['lost_tracks_at_root'] > 0 for r in rows),
                no_current_track_samples=sum(r['visible_tracks_at_root'] + r['lost_tracks_at_root'] == 0 for r in rows),
                scenes=dict(Counter(r['scene'] for r in rows)),
                human_counts=dict(Counter(str(r['human_count']) for r in rows)),
                visibility_categories=dict(Counter(r['visibility_history_category'] for r in rows)),
                query_variants=dict(Counter(r['query_variant'] for r in rows)),
                directions=dict(Counter(str(r['direction']) for r in rows)),
                per_root={rid: dict(samples=len(rs), positive=sum(r['event_label'] for r in rs),
                                   visible=rs[0]['visible_tracks_at_root'], lost=rs[0]['lost_tracks_at_root'])
                          for rid in sorted({r['root_id'] for r in rows})
                          for rs in [[r for r in rows if r['root_id'] == rid]]})


def select_micro(rows, seed=2301):
    """Deterministic balanced selection, greedily covering legal history/query categories."""
    selected, covered = [], set()
    def features(r):
        return {('root', r['root_id']), ('visibility', r['visibility_history_category']),
                ('variant', r['query_variant']), ('direction', str(r['direction'])),
                ('scene', r['scene']), ('count', r['human_count']),
                ('visible', r['visible_tracks_at_root'] > 0), ('lost', r['lost_tracks_at_root'] > 0)}
    for i in range(16):
        target = i % 2
        pool = [r for r in rows if r['event_label'] == target and r not in selected]
        if not pool:
            pool = [r for r in rows if r not in selected]
        if not pool:
            raise ValueError('Fewer than 16 eligible samples')
        chosen = min(pool, key=lambda r: (-len(features(r) - covered),
                     digest([seed, r['root_id'], r['query_id']])))
        selected.append(chosen); covered.update(features(chosen))
    return selected


class LegalProductionView:
    """Metadata/targets remain separate; features always come from load_wm_input.

    This is deliberately not a model tensorizer: the existing model contract is required.
    No scene, human_count, IDs, labels, event times or privileged files are forwarded as features.
    """
    def __init__(self, index_path):
        self.index = read(index_path)
        self.dataset = Path(self.index['dataset'])
        actual = hashlib.sha256((self.dataset / 'manifest.json').read_bytes()).hexdigest()
        if actual != self.index['manifest_sha256']:
            raise ValueError('Manifest changed after index audit')
        self.samples = self.index['samples']

    def __len__(self):
        return len(self.samples)

    def load(self, i):
        row = self.samples[i]
        path = (self.dataset / row['root_path']).resolve()
        if self.dataset.resolve() not in path.parents:
            raise ValueError('Root path escapes dataset')
        wm = load_wm_input(path, row['query_id'])
        if digest(wm['query']) != row['query_sha256'] or digest(wm['history']) != row['history_sha256']:
            raise ValueError('Legal input changed after index creation')
        return wm, row['event_label']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, default=REPO / 'output/legacy/eventwm_tiny_overfit/audit.json')
    p.add_argument('--output', type=Path, default=REPO / 'output/legacy/eventwm_tiny_overfit')
    args = p.parse_args()
    audit = read(args.audit); dataset = Path(audit['dataset'])
    if dataset == args.output.resolve() or dataset in args.output.resolve().parents:
        p.error('Index output must be outside raw dataset')
    manifest = read(dataset / 'manifest.json')
    manifest_sha = hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest()
    done = {r['root_id'] for r in manifest['roots'] if r['state'] == 'completed'}
    if audit['status'] != 'PASS' or audit['manifest_sha256'] != manifest_sha or {
            r['root_id'] for r in audit['completed'] if r['status'] == 'PASS'} != done:
        raise RuntimeError('Completed-data integrity gate has not passed')
    rows, excluded = [], []
    for root in manifest['roots']:
        if root['state'] != 'completed':
            continue
        path = dataset / root['path']
        queries = read(path / 'queries/queries.json')
        labels = {l['branch_id']: l for l in read(path / 'labels/labels.json')}
        for i, qrow in enumerate(queries):
            q = qrow['query']; bid = f'b{i:03}'; label = labels[bid]
            if label['query_id'] != q['query_id']:
                raise ValueError('Primary branch/query mismatch')
            reason = exclusion_reason(label, q)
            if reason:
                excluded.append(dict(root_id=root['root_id'], query_id=q['query_id'], branch_id=bid, reason=reason))
                continue
            legal = load_wm_input(path, q['query_id'])
            history = legal['history']; last = history[-1]['human_tracks']
            visible = sum(t['valid'] and t['visible_now'] for t in last)
            lost = sum(t['valid'] and not t['visible_now'] for t in last)
            seen = any(t['visible_now'] for h in history for t in h['human_tracks'])
            category = ('visible_and_lost' if visible and lost else 'visible' if visible else 'lost' if lost
                        else 'previously_seen_no_current_track' if seen else 'never_seen_no_track')
            variant = 'wait_only' if q['direction'] == 'WAIT' else 'slow' if q['speed_scale'] < 1 else (
                       'wait_advance' if q['wait_duration'] > 0 else 'immediate')
            rows.append(dict(root_id=root['root_id'], query_id=q['query_id'], branch_id=bid,
                root_path=root['path'], scene=root['scene'], scene_family=root['scene'], root_family=root['root_family'],
                human_count=root['human_count'], visibility_history_category=category, visible_human_history=seen,
                visible_tracks_at_root=visible, lost_tracks_at_root=lost, event_label=int(label['human_collision_event']),
                event_time=label['first_event_time'], query_variant=variant, direction=q['direction'],
                local_target=q['local_target'], wait_duration=q['wait_duration'], effective_wait_duration=q['effective_wait_duration'],
                speed_scale=q['speed_scale'], planned_duration=q['planned_duration'], observed_duration=label['observed_duration'],
                control_dt=q['control_dt'], integration_dt=q['integration_dt'],
                query_sha256=qrow['sha256'], history_sha256=digest(history), full_horizon_valid=True))
    index = dict(schema_version=1, dataset=str(dataset), manifest_sha256=manifest_sha,
                 audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
                 policy='PRIMARY only; exclude ALL censored; positive event within requested horizon; negatives clean full-query safe',
                 samples=rows, composition=composition(rows), exclusions=excluded,
                 exclusion_counts=dict(Counter(r['reason'] for r in excluded)),
                 audit_repeat_branches_excluded=sum(r['branches'] - r['primary_branches'] for r in audit['completed']))
    write(args.output / 'training_index.json', index)
    micro = select_micro(rows)
    write(args.output / 'micro16.json', dict(schema_version=1, seed=2301, selection='balanced greedy coverage; digest tie-break',
          index_sha256=hashlib.sha256((args.output / 'training_index.json').read_bytes()).hexdigest(),
          samples=[dict(root_id=r['root_id'], query_id=r['query_id']) for r in micro], composition=composition(micro)))
    view = LegalProductionView(args.output / 'training_index.json')
    for i in range(len(view)):
        view.load(i)
    print('VALID PRIMARY', len(rows), 'EXCLUDED', index['exclusion_counts'])
    print('MICRO16', composition(micro))


if __name__ == '__main__':
    main()
