"""Read-only mass-collection SR index and canonical legal-input Dataset."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from collector.core import read, digest
from .inputs import load_sample, collate

ROLES = ('train', 'natural_reference', 'unseen_evaluation')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_sr_target(label):
    """Validate the frozen native-contact SR contract without using social labels."""
    if not label['sr_valid']:
        return None
    target = label['sr_label']
    if target not in (0, 1):
        raise ValueError('Invalid SR label value')
    collision = bool(label['native_human_collision'] or label['static_collision'])
    if target == 0 and not collision:
        raise ValueError('SR=0 without declared physical collision')
    if target == 1 and (collision or not label['completed'] or not label['full_horizon_observed']
                        or label['termination_reason'] != 'query_completed'):
        raise ValueError('Incomplete non-collision branch cannot be safe')
    return target


def summarize(rows):
    roots, groups = defaultdict(set), defaultdict(list)
    for r in rows:
        roots[r['root_id']].add(r['sr_label'])
        if r['variant'] != 'WAIT':
            groups[r['group_id']].append(r)
    zeros = sum(r['sr_label'] == 0 for r in rows)
    return dict(roots=len(roots), queries=len(rows), sr_0=zeros, sr_1=len(rows)-zeros,
                collision_rate=zeros/len(rows) if rows else None,
                mixed_label_roots=sum(len(x) == 2 for x in roots.values()),
                within_path_groups=len(groups),
                mixed_within_path_groups=sum(len({r['sr_label'] for r in members}) == 2 and
                    len({(r['wait_duration'], r['speed_scale']) for r in members}) > 1
                    for members in groups.values()))


def inspect_root(dataset, root):
    path = dataset/root['path']
    seals = read(dataset/root['seal_file'])
    marker = read(path/'COMPLETE.json')
    if marker != dict(root_id=root['root_id'], seal_digest=digest(seals)):
        raise ValueError('Completion seal mismatch: '+root['root_id'])
    # Read and verify only training-facing data and validation receipt. Do not
    # rehash 507 GB of branch truth or launch simulation for a training index.
    for name in ('VALIDATED.json', 'legal_observations/history.json',
                 'queries/queries.json', 'labels/labels.json'):
        if sha(path/name) != seals[name]:
            raise ValueError('Sealed training record changed: '+str(path/name))
    if read(path/'VALIDATED.json')['root_id'] != root['root_id']:
        raise ValueError('Validation receipt root mismatch')
    qs = read(path/'queries/queries.json')
    queries = {q['query']['query_id']: q['query'] for q in qs}
    labels = [l for l in read(path/'labels/labels.json') if l['branch_id'].startswith('b')]
    primary = read(dataset/'primary_indices'/f"{root['root_id']}.json")
    for key, expected in [('root_id', root['root_id']), ('role', root['role']),
                          ('path', root['path']), ('split_group', root['root_id'])]:
        if primary[key] != expected:
            raise ValueError('Primary metadata mismatch: '+key)
    if primary['samples'] != labels:
        raise ValueError('Primary index labels differ from sealed labels')
    if len(qs) != 33 or len(queries) != 33 or len(labels) != 33:
        raise ValueError('Expected 33 unique primary candidates per completed root')
    ids = [l['query_id'] for l in labels]
    if len(set(ids)) != 33 or set(ids) != set(queries) or sorted(ids) != root['query_ids']:
        raise ValueError('Primary query coverage mismatch')
    legal = root['stats']['legal_state']
    rows = []
    for label in labels:
        if label['static_feasibility_violation']:
            raise ValueError('Static contract violation')
        target = valid_sr_target(label)
        if target is None:
            continue
        # Collision is authoritative even for early termination. No social
        # outcome is used to derive the binary target.
        q = queries[label['query_id']]
        if q['root_id'] != root['root_id'] or not q['static_precheck']['static_feasible']:
            raise ValueError('Invalid primary query')
        variant = ('WAIT' if q['direction'] == 'WAIT' else 'half_speed' if q['speed_scale'] < 1
                   else 'immediate_normal' if q['wait_duration'] == 0 else f"wait_{q['wait_duration']:.1f}")
        rows.append(dict(dataset=str(dataset), root_path=root['path'], root_id=root['root_id'],
            query_id=label['query_id'], branch_id=label['branch_id'], role=root['role'],
            split=root['role'], sr_valid=True, sr_label=target,
            group_id=root['root_id']+'/'+q['spatial_path_id'], spatial_path_id=q['spatial_path_id'],
            wait_duration=q['wait_duration'], speed_scale=q['speed_scale'], variant=variant,
            scene=root['physical_scene_id'], human_count=root['human_count'],
            visible_count=legal['visible_count'], lost_count=legal['lost_count'],
            category=legal['category'], sampling_profile=root['sampling_profile']))
    if dict(Counter(str(r['sr_label']) for r in rows)) != root['stats']['sr_targets']:
        raise ValueError('SR census differs from completed root stats')
    return rows


def build_index(dataset, output, seed=20261008, validation_fraction=.2):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError('Training outputs must be outside production dataset')
    manifest = read(dataset/'manifest.json')
    aggregate = read(dataset/'aggregate.json')
    if manifest['state'] != 'target_reached' or aggregate['primary_queries'] < 30000:
        raise ValueError('Collection has not reached its target')
    if aggregate['running_roots'] or aggregate['static_violations'] or aggregate['invalid']:
        raise ValueError('Collection is incomplete or contains invalid records')
    if manifest['sessions'][-1]['state'] != 'target_reached':
        raise ValueError('Final collection session did not finish normally')
    runner_pid = manifest['runner']['pid']
    proc = Path(f'/proc/{runner_pid}/cmdline')
    if proc.exists() and b'eventwm_combined_runner.py' in proc.read_bytes():
        raise ValueError('Collection coordinator is still running')
    roots = [r for r in manifest['roots'] if r['state'] == 'completed']
    if len({r['root_id'] for r in roots}) != len(roots):
        raise ValueError('Duplicate completed root')
    with ThreadPoolExecutor(max_workers=8) as pool:
        blocks = list(pool.map(lambda r: inspect_root(dataset, r), roots))
    rows = [r for block in blocks for r in block]
    if len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate primary query IDs')
    if len(rows) != aggregate['primary_queries'] or len(roots) != aggregate['completed_roots']:
        raise ValueError('Final aggregate census mismatch')
    if dict(Counter(str(r['sr_label']) for r in rows)) != aggregate['sr_targets']:
        raise ValueError('Final aggregate SR counts mismatch')
    if dict(Counter(r['role'] for r in roots)) != aggregate['roles']:
        raise ValueError('Final role census mismatch')
    if any(r['role'] not in ROLES for r in rows):
        raise ValueError('Unknown dataset role')
    train_ids = sorted({r['root_id'] for r in rows if r['role'] == 'train'},
                       key=lambda rid: hashlib.sha256(f'{seed}/{rid}'.encode()).hexdigest())
    val_ids = set(train_ids[:round(len(train_ids)*validation_fraction)])
    for r in rows:
        if r['role'] == 'train' and r['root_id'] in val_ids:
            r['split'] = 'validation'
    split_ids = {s: {r['root_id'] for r in rows if r['split'] == s}
                 for s in ('train', 'validation', *ROLES[1:])}
    for a, ids in split_ids.items():
        for b, other in split_ids.items():
            if a != b and ids & other:
                raise ValueError('Root leakage between splits')
    report = dict(total=summarize(rows), roles={s: summarize([r for r in rows if r['role'] == s]) for s in ROLES},
                  splits={s: summarize([r for r in rows if r['split'] == s]) for s in split_ids},
                  split_seed=seed, validation_fraction=validation_fraction,
                  split_policy='SHA256(seed/root_id), first 20% TRAIN roots -> validation; all 33 queries stay together',
                  completion=dict(state=manifest['state'], final_session=manifest['sessions'][-1],
                     coordinator_terminated=True, valid_completed_receipts=len(roots),
                     sealed_training_files_verified=4*len(roots), static_violations=0,
                     verification_scope='completion seal digests, validation receipts, legal history/query/label hashes and primary-index equality'),
                  production_hashes={n:sha(dataset/n) for n in ('manifest.json', 'aggregate.json')},
                  collector_commit=manifest['frozen_commit'], invalid_sr_excluded=aggregate['invalid'],
                  loader='collector.core.load_wm_input; full arrived history, existing 16-token nominal query encoding',
                  target='P(SR=1); BCEWithLogitsLoss on sr_valid=true authoritative sr_label')
    save(output/'index.json', dict(version='eventwm-v3-sr-index-v1', report=report, rows=rows))
    save(output/'dataset_statistics.json', report)
    return rows, report


class SRDataset(Dataset):
    """Metadata chooses the sample; only the canonical legal envelope is tensorized."""
    def __init__(self, rows, optimization=False):
        self.rows = rows
        if optimization and any(r['role'] != 'train' or r['split'] != 'train' for r in rows):
            raise ValueError('Only internal optimization TRAIN roots may be optimized')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return load_sample(self.rows[i])


def collate_sr(items):
    return collate([x for x, y in items]), torch.tensor([y for x, y in items], dtype=torch.float32)
