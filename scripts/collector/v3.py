"""Social-outcome v3: privileged label derivation, never query selection.

PSC's instantaneous predicate is frozen from falcon.additional_metric.PersonalSpaceCompliance.
Native contact alone is the validated physical-human detector. Research discs remain diagnostics.
"""
from enum import IntEnum
from pathlib import Path
import math
import numpy as np
from collector.core import Query, digest, read
from collector.v2 import generate_queries_v2, human_event_labels

PROTOCOL = 'eventwm-social-outcome-v3'
SOCIAL_METRIC = 'falcon-psc-geodesic-fallback-lt1-native-samples-v1'
SOCIAL_THRESHOLD = 1.0


class SocialOutcome(IntEnum):
    SAFE_SUCCESS = 0
    SOCIAL_VIOLATION_SUCCESS = 1
    HUMAN_COLLISION = 2


def guard_output(path):
    """Resolve aliases, reject legacy names/ancestors and foreign protocol markers."""
    dest = Path(path).resolve()
    repo = Path(__file__).resolve().parents[2]
    legacy = [(repo/'datasets/raw/eventwm_prod_v1').resolve()]
    legacy += [p.resolve() for p in (repo/'datasets/raw').glob('eventwm_human_event_v2*')]
    if any(dest == p or dest in p.parents or p in dest.parents for p in legacy):
        raise ValueError('V1/V2 datasets are read-only; use an isolated v3 output')
    for p in (Path(path).absolute(), dest):
        if any(x.startswith('eventwm_human_event_v2') or x == 'eventwm_prod_v1' for x in p.parts):
            raise ValueError('Reserved legacy dataset path')
    for p in (dest, *dest.parents):
        for marker in (p/'manifest.json', p/'roots/replay.json'):
            if marker.is_file() and read(marker).get('protocol_version') != PROTOCOL:
                raise ValueError('Cannot write v3 into a foreign protocol dataset')
    return dest


def generate_queries_v3(*args, **kwargs):
    """Reuse every v2 candidate/control/checker operation; version only the envelope."""
    queries, diagnostics = generate_queries_v2(*args, **kwargs)
    result, ids = [], {}
    for query in queries:
        body = query.data
        old = body.pop('query_id')
        body.update(schema_version=3, protocol_version=PROTOCOL)
        body['query_id'] = 'q_' + digest(body)[:12]
        ids[old] = body['query_id']
        result.append(Query.make(body))
    diagnostics['protocol_version'] = PROTOCOL
    for candidate in diagnostics['accepted_candidates']:
        candidate['query_ids'] = [ids[q] for q in candidate['query_ids']]
    return result, diagnostics


def social_violation(distance):
    if not math.isfinite(distance) or distance < 0:
        raise ValueError('Ambiguous/nonfinite social distance')
    return distance < SOCIAL_THRESHOLD


def social_outcome_labels(branch, query, human_count=None):
    """Single categorical label function; all raw evidence is retained separately."""
    old = human_event_labels(branch, query)  # shared collision extraction and full-horizon check
    rows = branch['truth']
    times = np.asarray([r['timestamp'] for r in rows], dtype=float)
    bad = None
    distances, flags, per_human = [], [], []
    try:
        if len(times) < 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('invalid timestamps')
        for row in rows:
            evidence = row['social_metric']
            if evidence['version'] != SOCIAL_METRIC:
                raise ValueError('metric version mismatch')
            ds = evidence['distances']
            if len(ds) != len(row['humans']) or (human_count is not None and len(ds) != human_count):
                raise ValueError('human count mismatch')
            values = [float(d['distance']) for d in ds]
            mask = [social_violation(x) for x in values]
            if any(d['mode'] not in ('geodesic', 'euclidean_no_path') for d in ds):
                raise ValueError('wrong PSC distance mode')
            distances.extend(values); per_human.append(mask); flags.append(any(mask))
        if len({len(x) for x in per_human}) != 1:
            raise ValueError('changing human count')
    except (KeyError, ValueError, TypeError):
        bad = 'missing_or_invalid_social_evidence'
    native = old['native_robot_human_contact']
    static = old['static_contact']
    social = any(flags)
    q = query.data if hasattr(query, 'data') else query
    reason = None
    if static:
        reason = 'static_feasibility_violation'
    elif q.get('protocol_version') != PROTOCOL:
        reason = 'protocol_mismatch'
    elif not q.get('static_precheck', {}).get('static_feasible'):
        reason = 'missing_static_precheck'
    elif human_count == 0 and (native or old['research_disc_contact'] or social):
        reason = 'zero_human_label_violation'
    elif branch['termination_reason'] == 'collection_error':
        reason = 'collection_error'
    elif native:
        pass  # determined physical outcome does not require a complete social future
    elif bad:
        reason = bad
    elif not (branch['completed'] and old['full_horizon_observed'] and branch['termination_reason'] == 'query_completed'):
        reason = 'incomplete_or_invalid_nonhuman_outcome'
    outcome = None if reason else (SocialOutcome.HUMAN_COLLISION if native else
              SocialOutcome.SOCIAL_VIOLATION_SUCCESS if social else SocialOutcome.SAFE_SUCCESS)
    centers = [float(np.linalg.norm(np.asarray(r['robot']['position'])-h['position']))
               for r in rows for h in r['humans']]
    episodes = 0
    if not bad and per_human:
        previous = [False] * len(per_human[0])
        for mask in per_human:
            episodes += sum(now and not before for now, before in zip(mask, previous))
            previous = mask
    # Keep old raw/general metadata, but do not publish v2 binary targets on v3 rows.
    for key in ('human_event_valid', 'human_event_label', 'protocol_version', 'invalid_reason', 'legacy_fields_note'):
        old.pop(key, None)
    # SR is independent of the three-class social outcome. A declared physical
    # collision determines SR even when it terminates execution early. Static
    # collisions still trip the independent static-feasibility quarantine gate.
    sr_valid = bool(native or static or (branch['completed'] and old['full_horizon_observed']
                    and branch['termination_reason'] == 'query_completed'))
    return dict(old, protocol_version=PROTOCOL, sr_contract_version='binary-execution-safety-v1',
        sr_valid=sr_valid, sr_label=(int(not (native or static)) if sr_valid else None),
        sr_invalid_reason=None if sr_valid else branch['termination_reason'],
        social_metric_version=SOCIAL_METRIC,
        social_threshold=SOCIAL_THRESHOLD, social_outcome_valid=reason is None,
        social_outcome_class=None if outcome is None else int(outcome),
        social_outcome_name=None if outcome is None else outcome.name, invalid_reason=reason,
        native_human_collision=native, direct_human_collision=native,
        direct_human_collision_detector='native_articulated_contact_only',
        research_disc_collision=old['research_disc_contact'],
        social_distance_violation=None if bad else social,
        first_human_collision_time=old['first_native_event_time'],
        first_social_violation_time=None if bad or not social else float(times[flags.index(True)]-times[0]),
        min_human_distance=min(centers) if centers else None,
        min_human_clearance=old['minimum_human_clearance'],
        human_clearance_definition='horizontal_research_disc_center_distance_minus_0.25_minus_0.30_not_mesh_surface',
        min_social_distance=min(distances) if distances and not bad else None,
        min_social_clearance=min(distances)-SOCIAL_THRESHOLD if distances and not bad else None,
        social_violation_duration=None if bad else float(np.sum(np.diff(times)*np.asarray(flags[1:]))),
        social_violation_count=None if bad else episodes,
        social_evidence_error=bad, static_collision=static,
        query_completed=bool(branch['completed'] and old['full_horizon_observed'] and branch['termination_reason']=='query_completed'))
