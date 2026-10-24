#!/usr/bin/env python3
"""Read-only, manifest-gated training census. No model/collector imports.

Independent unit is a manifest root. Future DAG nodes and event signatures are
descriptive dependent occurrences; sibling queries never become futures.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import time

import numpy as np

DEFAULT = '/research-user/workspace/projects/eventwm-falcon-l2-intermittent-500k-20261022/output/l2_intermittent_500k_20261022/snapshot'
EPS = 1e-7
POSITION_TOL = 2e-4
HORIZONS = (0., .5, 1., 2., 3., 5.)
# Static cell-update objects have no nested objects. Remove only these irrelevant
# arrays before decoding; if a new schema violates that property, decode fully.
# The original compressed bytes are still hashed. Legal root's graph is never
# needed: raw_decode reads just the complete history value, without decoding the
# large unrelated final graph. This changes parsing cost, not audited fields.
STATIC_ARRAY = re.compile(rb'("static_updates"\s*:\s*)\[(?:\s*\{[^{}]*\}\s*,?)*\s*\]')


def decode_payload(raw, legal=False):
    compact, removed = STATIC_ARRAY.subn(rb'\1[]', raw)
    if removed != raw.count(b'"static_updates"'):
        compact = raw
    if legal:
        match = re.search(rb'"history"\s*:\s*', compact)
        assert match is not None
        history, _ = json.JSONDecoder().raw_decode(compact[match.end():].decode())
        return {'history':history}
    return json.loads(compact)


def positive_tracks(ob):
    t = float(ob['timestamp'])
    if not ob.get('observation_valid', True) or not ob.get('human_sensor_available', True):
        return []
    return [r for r in ob.get('tracks', []) if r.get('valid', True)
            and r.get('visible_now', False)
            and abs(float(r.get('last_seen_time', -1e30)) - t) <= EPS
            and len(r.get('position', [])) == 3
            and np.isfinite(r['position']).all()]


def available(ob):
    return ob.get('observation_valid', True) and ob.get('human_sensor_available', True)


def advance_source(ob, state):
    state = dict(state)
    starts, returns = [], []
    if not available(ob):
        return state, starts, returns
    t = float(ob['timestamp'])
    ids = {r['track_id'] for r in positive_tracks(ob)}
    for identity, (last, missing) in list(state.items()):
        if identity not in ids and missing is None:
            state[identity] = (last, t)
            starts.append((identity, round(last, 8), round(t, 8)))
    for identity in ids:
        if identity in state and state[identity][1] is not None:
            last, missing = state[identity]
            returns.append((identity, round(last, 8), round(missing, 8), round(t, 8)))
        state[identity] = (t, None)
    return state, starts, returns


def match_physical(ob, times, positions):
    t = float(ob['timestamp'])
    k = int(np.searchsorted(times, t))
    candidates = [j for j in (k - 1, k) if 0 <= j < len(times) and abs(times[j] - t) <= EPS]
    if len(candidates) != 1:
        return None, False, 'timestamp_not_unique_native'
    y = positions[candidates[0]]
    matched = {}
    clean = True
    for track in positive_tracks(ob):
        d = np.linalg.norm(y - np.asarray(track['position']), axis=-1)
        close = np.flatnonzero(d < POSITION_TOL)
        if len(close) != 1:
            clean = False
            continue
        p = int(close[0])
        if p in matched:
            clean = False
        matched[p] = track['track_id']
    return matched, clean, None


def advance_physical(ob, state, times, positions):
    state = dict(state)
    starts, returns, aged = [], [], []
    ids, clean, failure = match_physical(ob, times, positions)
    if not available(ob) or ids is None or not clean:
        return state, starts, returns, aged, ids, clean, failure
    t = float(ob['timestamp'])
    # An unmatched positive might be the supposedly missing actor. Such frames
    # cannot start/advance a certified physical nondetection.
    if clean:
        for p, (last, missing, source_id) in list(state.items()):
            if p not in ids:
                if missing is None:
                    state[p] = (last, t, source_id)
                    starts.append((p, round(last, 8), round(t, 8)))
                aged.append((p, round(t, 8), round(last, 8)))
    for p, source_id in ids.items():
        if p in state and state[p][1] is not None:
            last, missing, previous_source = state[p]
            returns.append((p, round(last, 8), round(missing, 8), round(t, 8), previous_source == source_id))
        state[p] = (t, None, source_id)
    return state, starts, returns, aged, ids, clean, failure


def one_root(job):
    snapshot, row = job
    folder = Path(snapshot) / 'roots' / row['root_id']
    c = Counter()
    failures = []
    def read(name, zipped=True):
        b = (folder / name).read_bytes()
        c['payload_files_hashed'] += 1
        if hashlib.sha256(b).hexdigest() != row['files'][name]:
            failures.append('hash_mismatch:' + name)
        return decode_payload(gzip.decompress(b), name=='legal_root.json.gz') if zipped else b
    legal = read('legal_root.json.gz')
    future = read('future_observations.json.gz')
    with np.load(io.BytesIO(read('privileged_future.npz', False)), allow_pickle=False) as z:
        times, positions = z['timestamps'].copy(), z['positions'].copy()
    history = legal['history']
    assert history
    decision = float(history[-1]['timestamp'])
    assert np.isfinite(times).all() and np.isfinite(positions).all()
    assert positions.ndim == 3 and positions.shape[2] == 3 and len(times) == len(positions)
    assert np.all(np.diff(times) > 0) and abs(times[0] - decision) <= EPS
    assert all(float(b['timestamp']) >= float(a['timestamp']) for a,b in zip(history, history[1:]))
    c.update(roots=1, history_frames=len(history), future_dag_nodes=len(future), physical_actors=len(positions[0]))
    c['roots_with_multiple_history_frames'] = int(len(history) > 1)
    c['roots_with_future_observations'] = int(bool(future))
    c['roots_with_physical_actors'] = int(positions.shape[1] > 0)
    source_state = {}
    prefix_starts, prefix_returns = set(), set()
    for ob in history:
        source_state, starts, returns = advance_source(ob, source_state)
        prefix_starts.update(starts); prefix_returns.update(returns)
        pos = positive_tracks(ob)
        c['history_positive_detection_records'] += len(pos)
        c['history_available_frames'] += int(available(ob))
        c['history_available_empty_frames'] += int(available(ob) and not pos)
        c['history_invalid_frames'] += int(not ob.get('observation_valid', True))
        c['history_unavailable_sensor_frames'] += int(not ob.get('human_sensor_available', True))
    root_tracks = history[-1].get('tracks', [])
    retained_ages = [max(0., decision-float(r['last_seen_time'])) for r in root_tracks]
    all_seen_ages = [max(0., decision-v[0]) for v in source_state.values()]
    retained_ids = {r['track_id'] for r in root_tracks}
    c['root_retained_track_records'] = len(root_tracks)
    c['root_retained_aged_records'] = sum(a > EPS for a in retained_ages)
    c['roots_with_retained_aged_records'] = int(any(a > EPS for a in retained_ages))
    c['root_ever_positive_anonymous_ids'] = len(source_state)
    c['root_retired_anonymous_ids'] = len(set(source_state) - retained_ids)
    c['roots_with_retired_anonymous_ids'] = int(bool(set(source_state) - retained_ids))
    c['prefix_source_disappearance_events'] = len(prefix_starts)
    c['prefix_source_return_events'] = len(prefix_returns)
    c['roots_with_prefix_source_disappearance'] = int(bool(prefix_starts))
    c['roots_with_prefix_source_return'] = int(bool(prefix_returns))
    for h in HORIZONS:
        c[f'roots_with_native_future_at_least_{h:g}s'] = int(times[-1]-decision+EPS >= h)
    physical_state, _, _, _, root_ids, clean, failure = advance_physical(history[-1], {}, times, positions)
    c['root_physical_match_unclean'] = int(not clean)
    if failure: failures.append(failure)
    c['root_uniquely_matched_positive_actors'] = len(root_ids or {})
    # Source-track and physical states are copied from the ACTUAL parent node.
    states = {None: (source_state, physical_state, decision)}
    post_source_starts, post_source_returns = set(), set()
    physical_starts, physical_returns, aged_decisions = set(), set(), set()
    post_source_relinks = set()
    future_ages = []
    examples = []
    for node in future:
        nid, parent, ob = node['node_id'], node.get('parent'), node['observation']
        assert nid not in states and parent in states
        ss, ps, parent_t = states[parent]
        assert float(ob['timestamp']) > parent_t + EPS
        ss, starts, returns = advance_source(ob, ss)
        post_source_starts.update(starts); post_source_returns.update(returns)
        ps, pstarts, preturns, aged, ids, clean, failure = advance_physical(ob, ps, times, positions)
        physical_starts.update(pstarts)
        for event in preturns:
            physical_returns.add(event[:4])
            if not event[4]: post_source_relinks.add(event[:4])
            if len(examples) < 2:
                examples.append(dict(root_id=row['root_id'],node_id=nid,parent=parent,physical_actor=event[0],
                                     last_positive_time=event[1],first_missing_time=event[2],return_time=event[3],
                                     same_anonymous_source_id=event[4]))
        aged_decisions.update(aged)
        c['future_physical_match_unclean_frames'] += int(not clean)
        c['future_native_timestamp_unmatched_frames'] += int(failure is not None)
        c['future_matched_positive_actor_node_occurrences'] += len(ids or {})
        pos = positive_tracks(ob)
        c['future_positive_detection_node_records'] += len(pos)
        c['future_available_nodes'] += int(available(ob))
        c['future_available_empty_nodes'] += int(available(ob) and not pos)
        c['future_invalid_nodes'] += int(not ob.get('observation_valid', True))
        c['future_unavailable_sensor_nodes'] += int(not ob.get('human_sensor_available', True))
        future_ages.extend(max(0., float(ob['timestamp'])-float(r['last_seen_time'])) for r in ob.get('tracks', []))
        states[nid] = (ss, ps, float(ob['timestamp']))
    for label, events in [('postroot_source_disappearance',post_source_starts),('postroot_source_return',post_source_returns),
                          ('postroot_verified_physical_disappearance',physical_starts),('postroot_verified_physical_reappearance',physical_returns),
                          ('postroot_verified_physical_reappearance_new_source_id',post_source_relinks)]:
        c[label+'_signatures'] = len(events)
        c['roots_with_'+label] = int(bool(events))
    c['postroot_physically_anchored_aged_decision_signatures'] = len(aged_decisions)
    c['roots_with_postroot_physically_anchored_aged_decisions'] = int(bool(aged_decisions))
    for h in HORIZONS:
        support = [x for x in aged_decisions if times[-1]-x[1]+EPS >= h]
        c[f'aged_decision_signatures_with_{h:g}s_future'] = len(support)
        c[f'roots_with_aged_decisions_and_{h:g}s_future'] = int(bool(support))
        returns_support = [x for x in physical_returns if times[-1]-x[3]+EPS >= h]
        c[f'verified_reappearance_signatures_with_{h:g}s_future'] = len(returns_support)
        c[f'roots_with_verified_reappearance_and_{h:g}s_future'] = int(bool(returns_support))
    result = dict(counts=dict(c), retained_ages=retained_ages, all_seen_ages=all_seen_ages,
                  future_ages=future_ages, aged_physical_ages=[x[1]-x[2] for x in aged_decisions],
                  history_span=decision-float(history[0]['timestamp']), history_length=len(history),
                  native_future_span=float(times[-1]-times[0]),native_future_samples=len(times),
                  source_return_gaps=[x[3]-x[1] for x in prefix_returns],
                  verified_return_gaps=[x[3]-x[1] for x in physical_returns],
                  examples=examples,failures=failures)
    return result


def describe(values, age=False):
    x=np.asarray(values,dtype=float)
    out=dict(count=len(x),min=float(x.min()) if len(x) else None,max=float(x.max()) if len(x) else None,
             mean=float(x.mean()) if len(x) else None,
             percentiles={str(p):float(np.percentile(x,p)) for p in (0,25,50,75,90,95,99,100)} if len(x) else {})
    if age:
        bins=[('zero',x<=EPS),('0_to_0.25s',(x>EPS)&(x<=.25+EPS)),('0.25_to_0.5s',(x>.25+EPS)&(x<=.5+EPS)),
              ('0.5_to_1s',(x>.5+EPS)&(x<=1+EPS)),('1_to_2s',(x>1+EPS)&(x<=2+EPS)),('over_2s',x>2+EPS)]
        out['bins']={k:int(v.sum()) for k,v in bins}
    return out


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--snapshot',default=DEFAULT)
    parser.add_argument('--output',required=True);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args(); started=time.time()
    manifest_path=Path(args.snapshot)/'manifest.json'; mb=manifest_path.read_bytes();manifest=json.loads(mb)
    rows=manifest['rows']; assert all(r['split']=='train' and r['source_split']=='train' for r in rows)
    assert all(r['independent_futures']==1 for r in rows)
    assert len({r['root_id'] for r in rows})==len(rows)
    if args.limit: rows=rows[:args.limit]
    counts=Counter(); vectors={k:[] for k in ['retained_ages','all_seen_ages','future_ages','aged_physical_ages','history_span',
                                             'history_length','native_future_span','native_future_samples','source_return_gaps','verified_return_gaps']}
    examples=[]; failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,r in enumerate(pool.map(one_root,((args.snapshot,row) for row in rows),chunksize=16),1):
            counts.update(r['counts'])
            for k in vectors:
                v=r[k]; vectors[k].extend(v if isinstance(v,list) else [v])
            if len(examples)<12: examples.extend(r['examples'][:12-len(examples)])
            failures.extend(r['failures'])
            if i%1000==0: print(json.dumps(dict(processed_roots=i,total_roots=len(rows),seconds=round(time.time()-started,1))),flush=True)
    out=dict(schema='l2_native_history_census_v1',snapshot=args.snapshot,manifest_sha256=hashlib.sha256(mb).hexdigest(),
             audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             scope='ONLY manifest rows with split=train AND source_split=train; three payloads per root; no dev/confirmation/protected payloads',
             complete_snapshot=not args.limit,independent_roots=len(rows),scenes=len({r['scene_id'] for r in rows}),
             counts=dict(sorted(counts.items())),distributions={k:describe(v,'age' in k or 'gaps' in k) for k,v in vectors.items()},
             event_examples=examples,integrity_failures=failures,elapsed_seconds=time.time()-started,
             semantics=dict(positive='valid available frame; valid visible track; last_seen equals frame time within 1e-7; finite XYZ',
                            physical_match='unique stored timestamp within 1e-7 s AND unique physical XYZ center within 2e-4 m; no temporal interpolation; duplicate actor matches make absence uncertified',
                            physical_reappearance='uniquely matched same physical actor before and after >=1 actual valid available nondetection on same parent path; first anchor is root or later (no physical prefix truth)',
                            disappearance='first available actual nondetection after a positive; geometric/FOV occlusion included, not human sensor outage',
                            future_support='native privileged tape extends at least horizon beyond actual aged decision/return; not an assertion every desired endpoint is a native sample',
                            signatures='root-union signatures deduplicated over sibling DAG nodes by actor/source id and anchor/miss/return times; descriptive dependent opportunities, NOT independent futures',
                            age='retained track ages are source last-seen summaries; root deletion censors at 2 s. all_seen_ages includes retired anonymous IDs, not proven missing physical humans',
                            root_aged_correspondence='pre-root physical tape absent; retained root-aged source summaries lack direct unique physical correspondence from these files. Post-root anchored aged decisions are separately certified',
                            heldout='No protected, dev, confirmation or excluded payload opened; excluded metadata not traversed'))
    Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(output=args.output,roots=len(rows),seconds=out['elapsed_seconds'],integrity_failures=len(failures))),flush=True)


if __name__=='__main__': main()
