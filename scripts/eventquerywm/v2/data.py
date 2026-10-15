"""Frozen development caches and bounded, sharded preparation of completed data."""
import hashlib
import inspect
import json
from collections import defaultdict, Counter
from pathlib import Path
import torch
from torch.utils.data import Dataset
from ..inputs import collate, tensorize, input_hash
from ..v3_data import valid_sr_target, inspect_root
from collector.core import load_wm_input
from .history import arrived_tracks
from .labels import outcome_record_target, inspect_social_root, sealed_labels, supervision, fixture_metadata, PROTOCOL, METRIC


def torch_load(path):
    # The validated workstation has torch 1.12; newer torch defaults differ.
    kwargs = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    return torch.load(path, map_location='cpu', **kwargs)


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sr_record_target(target):
    declared = target.get('sr_valid')
    if declared is not None and not bool(declared):
        return -1,False
    raw = target.get('sr_label')
    try:
        label = int(raw)
    except (TypeError,ValueError,OverflowError):
        if declared is not None and bool(declared):raise ValueError('Malformed valid SR label')
        return -1,False
    valid = label in (0,1)
    if declared is not None and bool(declared) and not valid:raise ValueError('Invalid unmasked SR label')
    return label if valid else -1,valid


def safe_output(path, dataset):
    path, dataset = Path(path).resolve(), Path(dataset).resolve()
    if path == dataset or dataset in path.parents:
        raise ValueError('Outputs must be outside input data')
    # Never place development artifacts inside the protected checkout/launcher.
    if any(p in str(path) for p in ('eventwm_72h_v4_checkout', 'production/eventwm-v4-300k', '/tmp/eventwm_v4_300k')):
        raise ValueError('Protected production output path')
    return path


def audit_splits(entries):
    groups = {s:{r['root_id'] for r in entries if r['split'] == s} for s in ('train', 'validation')}
    if not groups['train'] or not groups['validation'] or groups['train'] & groups['validation']:
        raise ValueError('Need nonempty root-disjoint development splits')
    if len({r['query_id'] for r in entries}) != len(entries):
        raise ValueError('Duplicate query sample')
    if any(r.get('role', 'train') != 'train' for r in entries):
        raise ValueError('Held-out roles may not enter development')
    overlap = {}
    for key in ('scene', 'trajectory_id', 'episode_id', 'fixture_id'):
        a = {r[key] for r in entries if r['split'] == 'train' and r.get(key) is not None}
        b = {r[key] for r in entries if r['split'] == 'validation' and r.get(key) is not None}
        overlap[key] = dict(available=bool(a and b), shared=sorted(a & b), train_count=len(a), validation_count=len(b))
    return dict(roots={k:len(v) for k,v in groups.items()},
                samples={s:sum(r['split'] == s for r in entries) for s in groups}, overlap=overlap)


class DevelopmentDataset(Dataset):
    """One legacy cache or a prepared manifest; at most two shards in memory."""
    def __init__(self, path, split, validation_fraction=.2, seed=20261009, reference_only=False):
        self.path, self.split = Path(path).resolve(), split
        self.shards, self.records = {}, None
        if self.path.suffix == '.pt':
            payload = torch_load(self.path)
            self.records = payload['records']
            for record in self.records:record.pop('_v2_motion',None)
            # Reject held-out entries before preprocessing; never silently read them for development.
            entries = [dict(r['metadata'], offset=i, shard=None) for i,r in enumerate(self.records)]
            self.schema = payload['schema']
        else:
            payload = json.loads(self.path.read_text())
            if payload.get('version') != 'eventwm-v2-sharded-v1':
                raise ValueError('Run prepare on raw data/index before training')
            entries, self.schema = payload['entries'], payload['schema']
            self.shard_hashes = payload['shards']
        if reference_only:
            if any(r.get('role','train') != 'natural_reference' for r in entries):
                raise ValueError('Reference inference requires a separate natural-reference manifest')
            self.audit = dict(role='natural_reference',roots=len({r['root_id'] for r in entries}),samples=len(entries),training_forbidden=True)
        else:
            self.audit = audit_splits(entries)
        self.entries = [r for r in entries if r['split'] == split]

    def __len__(self):
        return len(self.entries)

    def _record(self, i):
        entry = self.entries[i]
        if self.records is not None:
            record = self.records[entry['offset']]
        else:
            name = entry['shard']
            if name not in self.shards:
                if len(self.shards) >= 2:
                    self.shards.pop(next(iter(self.shards)))
                path = self.path.parent/name
                if fingerprint(path) != self.shard_hashes[name]:
                    raise ValueError('Prepared shard changed: '+name)
                self.shards[name] = torch_load(path)['records']
                for record in self.shards[name]:record.pop('_v2_motion',None)
            record = self.shards[name][entry['offset']]
        if record['metadata']['query_id'] != entry['query_id']:
            raise ValueError('Shard index misalignment')
        return record

    def target_at(self, i):
        return self._record(i)['targets']

    def __getitem__(self, i):
        record = self._record(i)
        target = record['targets']
        y,valid = sr_record_target(target)
        outcome_record_target(target)
        if 'physical_mask' in target:
            duration = float(target['observed_duration_s'])
            horizons = torch.tensor(self.schema['horizons_s'])
            if (target['physical_mask'] & (horizons > duration+1e-5)).any():
                raise ValueError('Future target after branch termination')
        if '_v2_motion' not in record:
            record['_v2_motion'] = arrived_tracks(collate([record['input']]))
        return record


def collate_records(records):
    value = collate([r['input'] for r in records])
    # Legality preprocessing sees only ModelInput, never metadata/target.
    if all('_v2_motion' in r for r in records):
        b,n = value.humans.shape[:2]
        parts = [torch.zeros(b,n,8),torch.zeros(b,n,dtype=torch.bool),
                 torch.zeros(b,n,16,10),torch.zeros(b,n,16,dtype=torch.bool)]
        for i,r in enumerate(records):
            for part,source in zip(parts,r['_v2_motion']):
                part[i,:source.shape[1]] = source[0]
        motion = tuple(parts)
    else:
        motion = arrived_tracks(value)
    sr = [sr_record_target(r['targets']) for r in records]
    target = dict(sr_label=torch.tensor([label for label,valid in sr]),
                  sr_valid=torch.tensor([valid for label,valid in sr]))
    categorical = [outcome_record_target(r['targets']) for r in records]
    target.update(outcome_label=torch.tensor([y for y,valid in categorical]),
                  outcome_valid=torch.tensor([valid for y,valid in categorical]))
    for key in ('observed_duration_s','planned_duration_s','completed','censored',
                'human_collision_time_s','human_collision_time_valid','human_collision_evidence_valid',
                'social_violation_time_s','social_violation_time_valid','social_violation_evidence_valid'):
        if all(key in r['targets'] for r in records):
            target[key] = torch.stack([torch.as_tensor(r['targets'][key]) for r in records])
    if all('physical_occupancy' in r['targets'] for r in records):
        target['physical_occupancy'] = torch.stack([r['targets']['physical_occupancy'] for r in records])
        target['physical_mask'] = torch.stack([r['targets']['physical_mask'] for r in records])
    return value, motion, target, [r['metadata'] for r in records]


def prepare(dataset, output, seed=20261009, validation_fraction=.2, shard_roots=16, with_successor=False, role='train'):
    """SR-ready shards from a sealed completed collection or existing index.

    This preparation does not read branch truth. Dynamics stages require the
    separate existing successor cache or prepared physical target shards.
    No natural-reference/unseen inputs are loaded here.
    """
    dataset = Path(dataset).resolve()
    output = safe_output(output, dataset)
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must lie in (0,1)')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty preparation directory')
    collector_exclusions = {}
    def census(collection, roots, manifest):
        reasons = Counter()
        for root in roots:reasons.update(root.get('stats',{}).get('invalid_reasons',{}))
        collector_exclusions[str(collection)] = dict(roots=len(roots),offered_primary_queries=33*len(roots),
            retained_primary_queries=sum(r['stats']['primary_queries'] for r in roots),
            invalid_excluded=sum(r['stats'].get('invalid',0) for r in roots),invalid_reasons=dict(reasons),
            protocol_version=manifest.get('protocol_version'),sr_contract_version=manifest.get('sr_contract_version'),
            production_geometry_sampler=manifest.get('production_geometry_sampler',False),
            scope='Referenced completed source roots; an input index may contain only a subset of their retained queries')
    if dataset.is_dir():
        manifest = json.loads((dataset/'manifest.json').read_text())
        if manifest['state'] != 'target_reached' or any(r['state'] in ('running', 'preparing', 'validating') for r in manifest['roots']):
            raise ValueError('Collection must be completed before preparation')
        if manifest.get('protocol_version') != PROTOCOL:raise ValueError('Incompatible raw collection label protocol')
        census(dataset,[r for r in manifest['roots'] if r['state'] == 'completed' and r['role'] == role],manifest)
        rows = []
        for root in manifest['roots']:
            if root['state'] == 'completed' and root['role'] == role:
                rows.extend(inspect_social_root(dataset, root))
        source_hash = fingerprint(dataset/'manifest.json')
    else:
        index = json.loads(dataset.read_text())
        rows = [dict(r) for r in index['rows'] if r['role'] == role]
        for collection in {Path(r['dataset']).resolve() for r in rows}:
            manifest_path = collection/'manifest.json'
            if manifest_path.exists() and json.loads(manifest_path.read_text()).get('state') != 'target_reached':
                raise ValueError('An index cannot authorize reading an actively collected dataset')
            if manifest_path.exists():
                m = json.loads(manifest_path.read_text())
                roots = {r['root_id']:r for r in m['roots'] if r['state'] == 'completed' and r['role'] == role}
                scope = {r['root_id'] for r in rows if Path(r['dataset']).resolve() == collection}
                census(collection,[roots[r] for r in scope],m)
                labels = {}
                for row in (r for r in rows if Path(r['dataset']).resolve() == collection):
                    rid = row['root_id']
                    if rid not in labels:labels[rid] = sealed_labels(collection,roots[rid])
                    label = labels[rid][row['query_id']]
                    target = supervision(label)
                    if (target['sr_label'],target['sr_valid']) != (row['sr_label'],row['sr_valid']):
                        raise ValueError('Index SR differs from sealed original label')
                    row.update(target,**fixture_metadata(collection,roots[rid]),label_protocol=PROTOCOL,label_metric=METRIC)
        source_hash = fingerprint(dataset)
    if not rows:
        raise ValueError('No training roots')
    # Existing split assignments are authoritative. New splits are root-grouped.
    if role == 'natural_reference':
        for r in rows:r['split']='natural_reference'
    elif not any(r['split'] == 'validation' for r in rows):
        roots = sorted({r['root_id'] for r in rows}, key=lambda r: hashlib.sha256(f'{seed}/{r}'.encode()).hexdigest())
        validation = set(roots[:max(1, round(len(roots)*validation_fraction))])
        for r in rows:
            r['split'] = 'validation' if r['root_id'] in validation else 'train'
    audit = audit_splits(rows) if role == 'train' else dict(role=role,training_forbidden=True,roots=len({r['root_id'] for r in rows}),samples=len(rows))
    byroot = defaultdict(list)
    for row in rows:
        byroot[row['root_id']].append(row)
    output.mkdir(parents=True, exist_ok=True)
    entries, shards, records, root_count = [], {}, [], 0
    def flush():
        if not records:
            return
        name = f'shard_{len(shards):05d}.pt'
        torch.save(dict(records=records), output/name)
        shards[name] = fingerprint(output/name)
        for offset,r in enumerate(records):
            entries.append(dict(r['metadata'], shard=name, offset=offset))
        records.clear()
    for root_id, members in sorted(byroot.items()):
        root_count += 1
        physical = {}
        if with_successor:
            from .targets import root_physical_targets
            physical = root_physical_targets(Path(members[0]['dataset'])/members[0]['root_path'],
                                              {r['query_id'] for r in members})
        for row in members:
            value = tensorize(load_wm_input(Path(row['dataset'])/row['root_path'], row['query_id']))
            # If supplied by an audited index, invalid outcomes are explicitly masked.
            valid = bool(row['sr_valid'])
            label = int(row['sr_label']) if valid else -1
            if valid and label not in (0,1):
                raise ValueError('Invalid authoritative SR label')
            metadata = {k:row[k] for k in ('root_id','query_id','split','role','scene','group_id','variant','category',
                'episode_id','fixture_id','trajectory_id','spatial_path_id','wait_duration','speed_scale','sampling_profile',
                'label_protocol','label_metric') if k in row}
            targets = dict(sr_label=torch.tensor(label), sr_valid=torch.tensor(valid))
            targets.update(outcome_label=torch.tensor(row.get('outcome_label',-1)),
                           outcome_valid=torch.tensor(row.get('outcome_valid',False)))
            for key in ('observed_duration_s','planned_duration_s','completed','censored',
                        'human_collision_time_s','human_collision_time_valid','human_collision_evidence_valid',
                        'social_violation_time_s','social_violation_time_valid','social_violation_evidence_valid'):
                if key in row:targets[key] = torch.tensor(row[key])
            targets.update(physical.get(row['query_id'], {}))
            records.append(dict(input=value, input_hash=input_hash(value), metadata=metadata, targets=targets))
        if root_count % shard_roots == 0:
            flush()
    flush()
    result = dict(version='eventwm-v2-sharded-v1', entries=entries, shards=shards,
        source=str(dataset), source_sha256=source_hash, split_seed=seed, validation_fraction=validation_fraction, role=role,
        collector_exclusions=collector_exclusions,
        schema=dict(horizons_s=[.25,.5,1,2,3], grid_shape=[24,24], physical_supervision=with_successor,
                    outcome_protocol=PROTOCOL,outcome_metric=METRIC,
                    outcome_names=['SAFE_SUCCESS','SOCIAL_VIOLATION_SUCCESS','HUMAN_COLLISION']), audit=audit)
    write_json(output/'manifest.json', result)
    return output/'manifest.json'
