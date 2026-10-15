"""Authoritative social outcomes, independent SR and event censoring masks.

Never infer social labels from binary SR or successor-cache event_class.
"""
from collections import Counter
from pathlib import Path
import json
import hashlib
from collector.core import digest
from ..v3_data import valid_sr_target

PROTOCOL = 'eventwm-social-outcome-v3'
METRIC = 'falcon-psc-geodesic-fallback-lt1-native-samples-v1'
NAMES = ('SAFE_SUCCESS', 'SOCIAL_VIOLATION_SUCCESS', 'HUMAN_COLLISION')


def outcome_target(label):
    if not bool(label.get('social_outcome_valid', False)):
        if label.get('social_outcome_class') is not None:
            raise ValueError('Masked outcome must not carry a categorical label')
        return -1, False
    if label.get('protocol_version') != PROTOCOL or label.get('social_metric_version') != METRIC:
        raise ValueError('Incompatible categorical label protocol/metric')
    if label.get('social_threshold') != 1.0:
        raise ValueError('Incompatible social threshold')
    y = label.get('social_outcome_class')
    if type(y) is not int or y not in range(3) or label.get('social_outcome_name') != NAMES[y]:
        raise ValueError('Malformed authoritative categorical label')
    if label.get('static_collision') or label.get('static_feasibility_violation'):
        raise ValueError('Static collision cannot be a valid social outcome')
    native = bool(label['native_human_collision'])
    if native != (y == 2):
        raise ValueError('Human contact precedence mismatch')
    if y != 2:
        if not (label['completed'] and label['full_horizon_observed'] and label['termination_reason'] == 'query_completed'):
            raise ValueError('Non-contact outcome requires clean full completion')
        if bool(label['social_distance_violation']) != (y == 1):
            raise ValueError('Social success label mismatch')
    sr = valid_sr_target(label)
    if sr != int(y != 2):
        raise ValueError('Categorical and independent SR labels disagree')
    return y, True


def outcome_record_target(target):
    """Prepared targets retain an explicit mask; legacy binary caches stay masked."""
    valid = bool(target.get('outcome_valid', False))
    raw = target.get('outcome_label', -1)
    if not valid:
        return -1, False
    y = int(raw)
    if y not in range(3) or float(raw) != y:
        raise ValueError('Invalid unmasked prepared categorical label')
    return y, True


def supervision(label):
    y, valid = outcome_target(label)
    sr = valid_sr_target(label)
    end = float(label['observed_duration'])
    result = dict(outcome_label=y, outcome_valid=valid, sr_label=-1 if sr is None else sr,
                  sr_valid=sr is not None, observed_duration_s=end,
                  planned_duration_s=float(label['planned_duration']),
                  completed=bool(label['completed']), censored=bool(label['censored']))
    for name in ('human_collision', 'social_violation'):
        t = label.get('first_'+name+'_time')
        evidence = name == 'human_collision' or label.get('social_evidence_error') is None
        if t is not None and (not evidence or not 0 <= float(t) <= end+1e-5):
            raise ValueError('Event time outside observed branch/evidence')
        result[name+'_time_s'] = 0. if t is None else float(t)
        result[name+'_time_valid'] = bool(evidence and t is not None)
        result[name+'_evidence_valid'] = bool(evidence)
    return result


def sealed_labels(dataset, root):
    """Verify small sealed label evidence, never scan branch truth."""
    from .data import fingerprint
    dataset = Path(dataset)
    path = dataset/root['path']
    seals = json.loads((dataset/root['seal_file']).read_text())
    marker = json.loads((path/'COMPLETE.json').read_text())
    if marker != dict(root_id=root['root_id'], seal_digest=digest(seals)):
        raise ValueError('Root completion seal mismatch')
    if fingerprint(path/'labels/labels.json') != seals['labels/labels.json']:
        raise ValueError('Sealed labels changed')
    labels = json.loads((path/'labels/labels.json').read_text())
    primary = [l for l in labels if l['branch_id'].startswith('b')]
    if len({l['query_id'] for l in primary}) != len(primary):raise ValueError('Duplicate primary outcome')
    return {l['query_id']: l for l in primary}


def fixture_metadata(dataset, root):
    """Privileged episode metadata for split auditing only, never inference."""
    result = dict(episode_id=root.get('episode_sha256'),fixture_id=root.get('fixture_id'),trajectory_id=None)
    path = Path(dataset)/'catalogue/episodes'/f"{root['root_id']}.json"
    if not path.exists():return result
    episode = json.loads(path.read_text())
    count = root['human_count']
    waypoints = {k:v for k,v in episode['info'].items() if any(k.startswith(f'human_{i}_waypoint_') for i in range(count))}
    body = dict(scene=episode['scene_id'],human_count=count,waypoints=waypoints)
    result['fixture_id'] = hashlib.sha256(json.dumps(dict(body,policy_seed=episode['info'].get('policy_seed')),
                                                     sort_keys=True).encode()).hexdigest()
    if count and waypoints:
        result['trajectory_id'] = hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    return result


def inspect_social_root(dataset, root):
    """V3/V4 completed roots: 33 offered queries, possibly fewer retained labels."""
    from .data import fingerprint
    dataset = Path(dataset)
    path = dataset/root['path']
    all_labels = sealed_labels(dataset, root)
    seals = json.loads((dataset/root['seal_file']).read_text())
    for name in ('VALIDATED.json', 'legal_observations/history.json', 'queries/queries.json'):
        if fingerprint(path/name) != seals[name]:
            raise ValueError('Sealed training record changed: '+name)
    if json.loads((path/'VALIDATED.json').read_text())['root_id'] != root['root_id']:
        raise ValueError('Validation receipt mismatch')
    qs = json.loads((path/'queries/queries.json').read_text())
    queries = {v['query']['query_id']:v['query'] for v in qs}
    if any(v['sha256'] != digest(v['query']) for v in qs):raise ValueError('Query content digest mismatch')
    if len(qs) != 33 or len(all_labels) != 33 or set(queries) != set(all_labels):
        raise ValueError('Expected complete offered 33-query bank')
    if any(l['static_feasibility_violation'] for l in all_labels.values()):
        raise ValueError('Static contract quarantine')
    retained = [l for l in all_labels.values() if l['social_outcome_valid'] and l['sr_valid']]
    excluded = [l for l in all_labels.values() if l not in retained]
    if any(l['invalid_reason'] != 'incomplete_or_invalid_nonhuman_outcome' or l['sr_valid'] for l in excluded):
        raise ValueError('Unexpected collector exclusion')
    primary = json.loads((dataset/'primary_indices'/f"{root['root_id']}.json").read_text())
    if (primary['samples'] != retained or sorted(l['query_id'] for l in retained) != root['query_ids']
            or primary['root_id'] != root['root_id'] or primary['role'] != root['role']
            or primary['path'] != root['path'] or primary['split_group'] != root['root_id']):
        raise ValueError('Retained primary index mismatch')
    rows = []
    legal = root['stats']['legal_state']
    fixture = fixture_metadata(dataset,root)
    for label in retained:
        target = supervision(label)
        q = queries[label['query_id']]
        if q['root_id'] != root['root_id'] or q['protocol_version'] != PROTOCOL or not q['static_precheck']['static_feasible']:
            raise ValueError('Incompatible query contract')
        variant = ('WAIT' if q['direction'] == 'WAIT' else 'half_speed' if q['speed_scale'] < 1
                   else 'immediate_normal' if q['wait_duration'] == 0 else f"wait_{q['wait_duration']:.1f}")
        rows.append(dict(target, dataset=str(dataset), root_path=root['path'], root_id=root['root_id'],
            query_id=label['query_id'], branch_id=label['branch_id'], role=root['role'], split=root['role'],
            group_id=root['root_id']+'/'+q['spatial_path_id'], spatial_path_id=q['spatial_path_id'],
            wait_duration=q['wait_duration'], speed_scale=q['speed_scale'], variant=variant,
            scene=root['physical_scene_id'], **fixture, human_count=root['human_count'], category=legal['category'],
            sampling_profile=root['sampling_profile'], label_protocol=PROTOCOL, label_metric=METRIC))
    if dict(Counter(str(r['sr_label']) for r in rows)) != root['stats']['sr_targets']:
        raise ValueError('SR census mismatch')
    if dict(Counter(NAMES[r['outcome_label']] for r in rows)) != root['stats']['class_counts']:
        raise ValueError('Categorical census mismatch')
    return rows
