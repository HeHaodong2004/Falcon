"""Root-balanced, immutable V4 transition supervision.

Only ``base``, ``query``, ``sensor_config``, ``robot_geometry`` and explicit
invalid geometry placeholders are model inputs. Everything in ``targets`` is
privileged supervision and must never enter encode_history/imagine/observe.
Physical futures are deduplicated per root after a measured exogeneity check.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from collector.core import load_wm_input
from eventquerywm.inputs import tensorize, collate

VERSION = 'factorized-belief-root-transitions-v1'
EXTRACTOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def extraction_identity(block, dt, horizon):
    return dict(version=VERSION, root_id=block['root']['root_id'], dt=dt, horizon=horizon,
                extractor_sha256=EXTRACTOR_SHA256,
                seal_sha256=block['seal_sha256'], complete_sha256=block['complete_sha256'],
                branch_truth_sha256=block['evidence']['branch_truth/branches.jsonl.gz'])


def joined_labels(labels):
    result = {}
    for label in labels:
        key = label['branch_id'], label['query_id']
        if key in result:
            raise ValueError('Duplicate (branch_id, query_id) label')
        result[key] = label
    return result


def first_contact(label):
    values = [label.get('first_human_collision_time'), label.get('first_static_collision_time')]
    values = [float(v) for v in values if v is not None]
    if label.get('static_collision') and not values:
        return 0.  # No trustworthy pre-contact exposure without a contact clock.
    return min(values, default=float('inf'))


def temporal_event_targets(label, times):
    """At-risk discrete hazards: right-censored partial safe intervals are masked."""
    end = float(label['observed_duration'])
    event = label.get('first_human_collision_time')
    contact = first_contact(label)
    left = np.r_[0., times[:-1]]
    positive = np.zeros(len(times), dtype=bool)
    if event is not None:
        event = float(event)
        if not 0 <= event <= end + 1e-5:
            raise ValueError('Contact outside observed exposure')
        positive = (event > left + 1e-8) & (event <= times + 1e-8)
        if event <= 1e-8:
            positive[0] = True
    valid = ((times <= end + 1e-8) & (times < contact - 1e-8)) | positive
    return positive, valid


def planar_frame(history):
    origin = np.asarray(history[-1]['robot_position'], dtype=np.float64)
    yaw = float(history[-1]['robot_yaw'])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return origin, yaw, rotation


def compact_truth(frames, origin, yaw, rotation):
    """Strip joints immediately; velocity targets use timestamped positions only."""
    bytime = {float(f['timestamp']): f for f in frames}
    clocks = np.asarray(sorted(bytime), dtype=np.float64)
    ids = sorted({int(h['object_id']) for f in frames for h in f['humans']})
    index = {identity: i for i, identity in enumerate(ids)}
    xy = np.zeros((len(clocks), len(ids), 2), dtype=np.float64)
    valid = np.zeros((len(clocks), len(ids)), dtype=bool)
    robot = np.zeros((len(clocks), 3), dtype=np.float64)
    for j, clock in enumerate(clocks):
        f = bytime[clock]
        for h in f['humans']:
            i = index[int(h['object_id'])]
            xy[j, i] = rotation @ (np.asarray(h['position']) - origin)[[0, 2]]
            valid[j, i] = True
        robot[j, :2] = rotation @ (np.asarray(f['robot']['position']) - origin)[[0, 2]]
        robot[j, 2] = float(f['robot']['yaw']) - yaw
    robot[:, 2] = np.unwrap(robot[:, 2])
    return dict(times=clocks, ids=ids, xy=xy, valid=valid, robot=robot)


def interpolate(truth, clocks):
    """Interpolate only between actual native frames; never extrapolate exposure."""
    clocks = np.asarray(clocks)
    t = truth['times']
    hi = np.clip(np.searchsorted(t, clocks, side='left'), 0, len(t) - 1)
    lo = np.maximum(hi - 1, 0)
    exact = np.abs(t[hi] - clocks) < 1e-8
    lo[exact] = hi[exact]
    weight = np.divide(clocks-t[lo], t[hi]-t[lo], out=np.zeros(len(clocks)), where=hi != lo)
    support = (clocks >= t[0] - 1e-8) & (clocks <= t[-1] + 1e-8)
    xy = truth['xy'][lo] + weight[:, None, None] * (truth['xy'][hi]-truth['xy'][lo])
    mask = truth['valid'][lo] & truth['valid'][hi] & support[:, None]
    robot = truth['robot'][lo] + weight[:, None] * (truth['robot'][hi]-truth['robot'][lo])
    return xy.astype(np.float32), mask, robot.astype(np.float32), support


def associate_tracks(history, prefix, origin, rotation, tolerance=1e-4):
    """Target-only identity association from positive observations at their clocks.

    Lost tracks keep the identity established by earlier arrived detections.
    Ambiguous associations or observed identity switches are masked completely.
    """
    ids = sorted({t['track_id'] for h in history for t in h['human_tracks']})
    matches = {identity: set() for identity in ids}
    ambiguous = set()
    for h in history:
        xy, valid, _, support = interpolate(prefix, [float(h['timestamp'])])
        if not support[0]:
            continue
        for tr in h['human_tracks']:
            if not (tr['visible_now'] and tr['valid']):
                continue
            p = rotation @ (np.asarray(tr['position'])-origin)[[0, 2]]
            close = np.flatnonzero((np.linalg.norm(xy[0]-p, axis=-1) <= tolerance) & valid[0])
            if len(close) == 1:
                matches[tr['track_id']].add(prefix['ids'][close[0]])
            else:
                ambiguous.add(tr['track_id'])
    mapping = {identity: next(iter(matches[identity])) for identity in ids
               if len(matches[identity]) == 1 and identity not in ambiguous}
    # Two simultaneous track slots mapping to one physical human are not independent targets.
    duplicate = {x for x in mapping.values() if list(mapping.values()).count(x) > 1}
    mapping = {k: v for k, v in mapping.items() if v not in duplicate}
    return ids, mapping, dict(tracks=len(ids), associated=len(mapping),
                              ambiguous=len(ambiguous), switched=sum(len(v)>1 for v in matches.values()))


def observation_targets(observations, absolute_times, track_ids, origin, rotation, end, contact,
                        truth=None, associations=None):
    """Actual arrived measurements for correction; no synthetic future detection."""
    n, l = len(track_ids), len(absolute_times)
    index = {identity: i for i, identity in enumerate(track_ids)}
    available = np.zeros((l, n), dtype=bool)
    valid = np.zeros((l, n), dtype=bool)
    positions = np.zeros((l, n, 2), dtype=np.float32)
    acquisition = np.zeros((l, n), dtype=np.float64)
    clocks = np.zeros(l, dtype=np.float64)
    ot = np.asarray([float(h['timestamp']) for h in observations])
    if not len(ot):
        return available, valid, positions, acquisition, clocks
    start = absolute_times[0] - (absolute_times[1]-absolute_times[0] if l > 1 else .25)
    identity_ok = np.ones((len(observations), n), dtype=bool)
    if truth is not None and associations is not None:
        physical_index = {identity:i for i,identity in enumerate(truth['ids'])}
        positions_at_observation, physical_valid, _, _ = interpolate(truth, ot)
        current = np.array([identity in associations for identity in track_ids])
        for o, h in enumerate(observations):
            for tr in h['human_tracks']:
                identity = tr['track_id']
                if identity not in index or not (tr['visible_now'] and tr['valid']):
                    continue
                i = index[identity]
                physical = physical_index.get(associations.get(identity))
                if physical is None:
                    current[i] = False
                    continue
                p = rotation @ (np.asarray(tr['position'])-origin)[[0,2]]
                if (not physical_valid[o,physical] or
                        np.linalg.norm(positions_at_observation[o,physical]-p)>1e-4):
                    current[i] = False  # A reassigned tracker identity cannot supervise this physical track.
            identity_ok[o] = current
    for j, clock in enumerate(absolute_times):
        ix = np.searchsorted(ot, clock+1e-8, side='right')-1
        if ix < 0 or clock-ot[ix] > .060001 or clock-start > end+1e-8 or clock-start >= contact-1e-8:
            continue
        h = observations[ix]
        clocks[j] = ot[ix]
        valid[j] = bool(h.get('tracking_available', False)) & identity_ok[ix]
        for tr in h['human_tracks']:
            if tr['track_id'] not in index:
                continue
            i = index[tr['track_id']]
            available[j, i] = bool(tr['visible_now'] and tr['valid']) and bool(identity_ok[ix,i])
            if available[j, i]:
                positions[j, i] = rotation @ (np.asarray(tr['position'])-origin)[[0, 2]]
                acquisition[j, i] = float(tr['last_seen_time'])
    return available, valid, positions, acquisition, clocks


def extract_root(block, dt=.25, horizon=5., verify_hashes=True):
    path = Path(block['archive_path'])
    rows = block['rows']
    metadata = {(r['branch_id'], r['query_id']): r for r in rows}
    if len(metadata) != len(rows):
        raise ValueError('Duplicate authoritative branch/query row')
    seals = read(Path(rows[0]['dataset']) / block['root']['seal_file'])
    if verify_hashes:
        if sha256(Path(rows[0]['dataset']) / block['root']['seal_file']) != block['seal_sha256']:
            raise ValueError('Frozen root seal provenance changed')
        for name in ('legal_observations/history.json', 'queries/queries.json', 'labels/labels.json',
                     'branch_truth/branches.jsonl.gz', 'roots/prefix_truth.json.gz',
                     'legal_observations/observation_config.json'):
            if name not in seals or sha256(path/name) != seals[name]:
                raise ValueError('Sealed input mismatch: '+name)
        if sha256(path/'COMPLETE.json') != block['complete_sha256']:
            raise ValueError('Completion provenance changed')
    labels = joined_labels(read(path/'labels/labels.json'))
    history = read(path/'legal_observations/history.json')
    origin, yaw, rotation = planar_frame(history)
    t0 = float(history[-1]['timestamp'])
    times = np.arange(1, int(round(horizon/dt))+1, dtype=np.float64)*dt
    if abs(times[-1]-horizon) > 1e-8:
        raise ValueError('Horizon must be an integer number of canonical transitions')
    query_records = {q['query']['query_id']: q['query'] for q in read(path/'queries/queries.json')}
    base = tensorize(load_wm_input(path, rows[0]['query_id']))
    query = torch.stack([tensorize(dict(history=[history[-1]], history_images=None,
                                      query=query_records[r['query_id']])).query for r in rows])
    with gzip.open(path/'roots/prefix_truth.json.gz', 'rt') as f:
        prefix = compact_truth(json.load(f), origin, yaw, rotation)
    track_ids, associations, association_audit = associate_tracks(history, prefix, origin, rotation)
    track_index = {identity:i for i,identity in enumerate(track_ids)}
    acquisition_history = np.zeros((len(track_ids),len(history)),dtype=np.float64)
    for j,h in enumerate(history):
        for tr in h['human_tracks']:
            acquisition_history[track_index[tr['track_id']],j] = float(tr['last_seen_time'])
    branches = {}
    with gzip.open(path/'branch_truth/branches.jsonl.gz', 'rt') as f:
        for line in f:
            branch = json.loads(line)
            key = branch['branch_id'], branch['query_id']
            if key not in metadata:
                continue
            if key in branches:
                raise ValueError('Duplicate native branch')
            if branch['query_hash_before'] != branch['query_hash_after']:
                raise ValueError('Query changed during execution')
            truth = compact_truth(branch['truth'], origin, yaw, rotation)
            if abs(truth['times'][0]-t0) > 1e-7:
                raise ValueError('Branch and legal history timebase mismatch')
            label = labels[key]
            if label['sr_valid'] != metadata[key]['sr_valid'] or label['sr_label'] != metadata[key]['sr_label']:
                raise ValueError('Frozen label differs from authoritative branch/query join')
            end = float(label['observed_duration'])
            if abs(truth['times'][-1]-t0-end) > 1e-6:
                raise ValueError('Native duration mismatch')
            contact = first_contact(label)
            xy, mask, robot, support = interpolate(truth, np.r_[t0, t0+times])
            mask &= (np.r_[0., times] <= end+1e-8)[:, None]
            mask &= (np.r_[0., times] < contact-1e-8)[:, None]
            observations = observation_targets(branch['legal_observations'], t0+times, track_ids,
                                                origin, rotation, end, contact,truth,associations)
            event, event_valid = temporal_event_targets(label, times)
            branches[key] = dict(ids=truth['ids'], xy=xy, mask=mask, robot=robot[1:],
                                 exposure=support[1:] & (times <= end+1e-8) & (times < contact-1e-8),
                                 observations=observations, event=event, event_valid=event_valid,
                                 end=end, contact=contact)
            if len(branches) == len(metadata):
                break
    if set(branches) != set(metadata):
        raise ValueError('Missing native primary branches')
    ordered = [branches[r['branch_id'], r['query_id']] for r in rows]
    reference = max(ordered, key=lambda b: min(b['end'], b['contact']))
    max_difference, comparisons = 0., 0
    for b in ordered:
        if b['ids'] != reference['ids']:
            raise ValueError('Exogenous physical identity set changed across candidates')
        both = b['mask'] & reference['mask']
        delta = np.linalg.norm(b['xy']-reference['xy'], axis=-1)[both]
        if len(delta):
            max_difference = max(max_difference, float(delta.max()))
            comparisons += len(delta)
    if max_difference > 1e-4:
        raise ValueError(f'Exogeneity rejected: {max_difference:.6g}m across candidates')
    m, n = len(reference['ids']), len(track_ids)
    positions = np.zeros((len(times)+1, n, 2), dtype=np.float32)
    valid = np.zeros((len(times)+1, n), dtype=bool)
    pindex = {identity: i for i, identity in enumerate(reference['ids'])}
    for i, identity in enumerate(track_ids):
        physical = associations.get(identity)
        if physical in pindex:
            j = pindex[physical]
            positions[:, i] = reference['xy'][:, j]
            valid[:, i] = reference['mask'][:, j]
    targets = dict(times=times, track_positions=positions[1:], track_valid=valid[1:],
                   initial_track_positions=positions[0], initial_track_valid=valid[0],
                   all_human_positions=reference['xy'][1:], all_human_valid=reference['mask'][1:],
                   physical_exposure=reference['exposure'],
                   actual_robot=np.stack([b['robot'] for b in ordered]),
                   query_exposure=np.stack([b['exposure'] for b in ordered]),
                   collision_interval=np.stack([b['event'] for b in ordered]),
                   collision_valid=np.stack([b['event_valid'] for b in ordered]),
                   observed_tracks=np.stack([b['observations'][0] for b in ordered]),
                   observed_valid=np.stack([b['observations'][1] for b in ordered]),
                   actual_observed_positions=np.stack([b['observations'][2] for b in ordered]),
                   actual_acquisition_times=np.stack([b['observations'][3] for b in ordered]),
                   actual_observation_times=np.stack([b['observations'][4] for b in ordered]))
    targets = {k: torch.from_numpy(v) for k, v in targets.items()}
    first_query = query_records[rows[0]['query_id']]
    execution_errors, representation_errors = [], []
    for k, row in enumerate(rows):
        trajectory = np.asarray(query_records[row['query_id']]['nominal_timed_trajectory'])
        xy = (rotation @ (trajectory[:, 1:4]-origin)[:, [0,2]].T).T
        full = np.column_stack([np.interp(times, trajectory[:,0], xy[:,d]) for d in (0,1)])
        encoded = np.column_stack([np.interp(times, query[k,:,2].numpy(), query[k,:,d].numpy()) for d in (0,1)])
        support = ordered[k]['exposure'] & (times <= trajectory[-1,0]+1e-8)
        execution_errors.extend(np.linalg.norm(ordered[k]['robot'][:,:2]-full,axis=-1)[support].tolist())
        representation_errors.extend(np.linalg.norm(encoded-full,axis=-1)[support].tolist())
    error_summary = lambda values: dict(count=len(values), mean=float(np.mean(values)) if values else None,
                                        p95=float(np.quantile(values,.95)) if values else None,
                                        max=float(max(values)) if values else None)
    return dict(version=VERSION, extraction=extraction_identity(block,dt,horizon), base=base, query=query, targets=targets,
                history_timestamps=torch.tensor([float(h['timestamp']) for h in history],dtype=torch.float64),
                track_acquisition_times=torch.from_numpy(acquisition_history),
                geometry=torch.zeros(2,32,32), geometry_support=torch.zeros(32,32,dtype=torch.bool),
                sensor_config=read(path/'legal_observations/observation_config.json'),
                robot_geometry=first_query['static_precheck']['geometry'],
                metadata=dict(root_id=block['root']['root_id'], split=block['split'], rows=rows,
                              track_ids=track_ids, frame_origin=origin.tolist(), frame_yaw=yaw,
                              timestamp=t0, association=association_audit,
                              exogeneity_max_difference_m=max_difference, exogeneity_comparisons=comparisons,
                              physical_source_branch=next(k for k,v in branches.items() if v is reference),
                              physical_humans_target_only=m, geometry_valid=False,
                              nominal_execution_error_m=error_summary(execution_errors),
                              query16_interpolation_error_m=error_summary(representation_errors),
                              physical_targets='actual precontact positions; no postterminal extrapolation'))


def collate_roots(samples, device='cpu'):
    """Pad legal tracks and target humans independently; counts never enter input."""
    base = collate([s['base'] for s in samples]).to(device)
    n = base.humans.shape[1]
    m = max(1, max(s['targets']['all_human_positions'].shape[1] for s in samples))
    track_axes = dict(track_positions=1, track_valid=1, initial_track_positions=0,
                      initial_track_valid=0, observed_tracks=2, observed_valid=2,
                      actual_observed_positions=2, actual_acquisition_times=2)
    human_axes = dict(all_human_positions=1, all_human_valid=1)
    targets = {}
    history_timestamps = torch.zeros(len(samples),base.robot.shape[1],dtype=torch.float64,device=device)
    track_acquisition_times = torch.zeros(len(samples),n,base.robot.shape[1],dtype=torch.float64,device=device)
    for i,sample in enumerate(samples):
        ti=sample['history_timestamps'].shape[0]
        ni=sample['track_acquisition_times'].shape[0]
        history_timestamps[i,:ti]=sample['history_timestamps'].to(device)
        track_acquisition_times[i,:ni,:ti]=sample['track_acquisition_times'].to(device)
    for name in samples[0]['targets']:
        values = []
        for sample in samples:
            value = sample['targets'][name]
            axis = track_axes.get(name, human_axes.get(name))
            if axis is not None:
                shape = list(value.shape)
                shape[axis] = n if name in track_axes else m
                padded = value.new_zeros(shape)
                slices = [slice(None)]*len(shape)
                slices[axis] = slice(0, value.shape[axis])
                padded[tuple(slices)] = value
                value = padded
            values.append(value)
        targets[name] = torch.stack(values).to(device)
    return dict(base=base, query=torch.stack([s['query'] for s in samples]).to(device),
                history_timestamps=history_timestamps,track_acquisition_times=track_acquisition_times,
                targets=targets, geometry=torch.stack([s['geometry'] for s in samples]).to(device),
                geometry_support=torch.stack([s['geometry_support'] for s in samples]).to(device),
                sensor_config=[s['sensor_config'] for s in samples],
                robot_geometry=[s['robot_geometry'] for s in samples],
                metadata=[s['metadata'] for s in samples])


class RootTransitionDataset(Dataset):
    """One item per independent root; all queries retained inside each root item."""
    def __init__(self, cache, split='train'):
        self.cache = Path(cache)
        self.manifest = read(self.cache/'manifest.json')
        if self.manifest['version'] != VERSION:
            raise ValueError('Transition cache version mismatch')
        self.rows = [r for r in self.manifest['roots'] if r['split'] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        value = torch.load(self.cache/row['file'], map_location='cpu')
        if value['metadata']['root_id'] != row['root_id']:
            raise ValueError('Cache root provenance mismatch')
        return value


def build_cache(snapshot, output, counts, dt=.25, horizon=5., seed=20261014, cache_from=None):
    snapshot, output = Path(snapshot), Path(output)
    snapshot_hash = sha256(snapshot)
    source = read(snapshot)
    config = dict(version=VERSION, snapshot=str(snapshot.resolve()), snapshot_sha256=snapshot_hash,
                  counts=counts, dt=dt, horizon=horizon, seed=seed,extractor_sha256=EXTRACTOR_SHA256)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output/'config.json'
    if identity_path.exists() and read(identity_path) != config:
        raise ValueError('Resume config differs; use a new output directory')
    identity_path.write_text(json.dumps(config, indent=2)+'\n')
    selected = []
    for split, count in counts.items():
        candidates = [b for b in source['blocks'] if b['split'] == split]
        candidates.sort(key=lambda b: hashlib.sha256(f"{seed}/{b['root']['root_id']}".encode()).hexdigest())
        selected.extend(candidates if count < 0 else candidates[:count])
    entries = []
    reusable = {}
    if cache_from is not None:
        cache_from = Path(cache_from)
        previous = read(cache_from/'manifest.json')
        reusable = {r['root_id']: r for r in previous['roots']}
    started = time.time()
    for i, block in enumerate(selected):
        if (output/'STOP_RESEARCH').exists():
            raise RuntimeError('Research preprocessing stop requested; collector untouched')
        rid = block['root']['root_id']
        destination = output/(rid+'.pt')
        if not destination.exists():
            reused = False
            if rid in reusable:
                old_path = cache_from/reusable[rid]['file']
                if sha256(old_path) != reusable[rid]['sha256']:
                    raise ValueError('Reusable compact cache hash changed')
                old = torch.load(old_path,map_location='cpu')
                if old.get('extraction') == extraction_identity(block,dt,horizon):
                    shutil.copy2(old_path,destination)
                    reused = True
            if not reused:
                sample = extract_root(block, dt, horizon)
                temporary = destination.with_suffix('.pt.tmp')
                torch.save(sample, temporary)
                os.replace(temporary, destination)
            else:
                sample = old
        else:
            sample = torch.load(destination, map_location='cpu')
        if (sample['version'] != VERSION or sample['metadata']['root_id'] != rid or
                sample.get('extraction') != extraction_identity(block,dt,horizon)):
            raise ValueError('Incompatible partial cache')
        coverage = {str(h): int(sample['targets']['track_valid'][round(h/dt)-1].sum())
                    for h in (1.,3.,5.) if h <= horizon}
        entries.append(dict(root_id=rid, split=block['split'], file=destination.name,
                            sha256=sha256(destination), queries=len(block['rows']),
                            track_coverage=coverage, association=sample['metadata']['association'],
                            physical_horizon_coverage={str(h): bool(sample['targets']['physical_exposure'][round(h/dt)-1])
                                                       for h in (1.,3.,5.) if h <= horizon},
                            query_horizon_coverage={str(h): int(sample['targets']['query_exposure'][:,round(h/dt)-1].sum())
                                                    for h in (1.,3.,5.) if h <= horizon},
                            strata=dict(category=block['rows'][0]['category'],scene=block['rows'][0]['scene'],
                                        visible_now=block['rows'][0]['visible_now'],ever_tracks=block['rows'][0]['ever_tracks'],
                                        human_count_target_only=sample['metadata']['physical_humans_target_only']),
                            exogeneity_max_difference_m=sample['metadata']['exogeneity_max_difference_m']))
        print(json.dumps(dict(stage='root_cached', index=i+1, total=len(selected), root_id=rid,
                              seconds=round(time.time()-started,2), coverage=coverage)), flush=True)
    manifest = dict(config, roots=entries, created_at=time.time(), raw_data_copied=False,
                    training_unit='independent root; query outcomes retained within root',
                    target_privilege_boundary='targets and metadata never model inputs')
    temporary = output/'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    os.replace(temporary, output/'manifest.json')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-roots', type=int, default=32)
    p.add_argument('--validation-roots', type=int, default=16)
    p.add_argument('--natural-roots', type=int, default=0)
    p.add_argument('--unseen-roots', type=int, default=0)
    p.add_argument('--dt', type=float, default=.25)
    p.add_argument('--horizon', type=float, default=5.)
    p.add_argument('--seed', type=int, default=20261014)
    p.add_argument('--cache-from', type=Path, help='Reuse identical sealed root targets across frozen snapshots')
    args = p.parse_args()
    counts = dict(train=args.train_roots, validation=args.validation_roots,
                  natural_reference=args.natural_roots, unseen_evaluation=args.unseen_roots)
    build_cache(args.snapshot, args.output, counts, args.dt, args.horizon, args.seed,args.cache_from)


if __name__ == '__main__':
    main()
