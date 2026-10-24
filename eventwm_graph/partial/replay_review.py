"""Independent fresh-Env replay audit for sealed development roots.

This validation tool reads privileged labels and replays native actions. It is
never imported by the collector or proposal API, and refuses held-out scenes.
Run with ``python -m eventwm_graph.partial.replay_review ROOT --output REPORT``.
"""
import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from ..config import REPO, file_digest, read_json, write_json
from ..stage1.storage import read_gzip, scene_split
from .map import PartialGraph, KNOWN_OCCUPIED
from .metrics import step_labels, summarize
from .observer import GeometryObserver
from .query import FORWARD_STEP, TURN_STEP, static_body_clearance, turn_clearance, wrap, generate_queries
from .storage import implementation, validate_global_graph, validate_root


REVIEW_VERSION = 'fresh_native_partial_graph_replay_review_3'
PRODUCER_QUERY_CHECKPOINTS = (11,12,13,14,15,16,17,18,24,34,39,58,81)


@dataclass(frozen=True)
class Tolerances:
    """Native-to-native bounds; compiled floor approximation is not used here."""
    position_m: float = 1e-4
    yaw_rad: float = 1e-5
    camera_rotation: float = 1e-5
    velocity_m_s: float = 1e-3
    distance_m: float = 1e-4
    timestamp_s: float = 1e-8
    scalar: float = 1e-7
    max_reported_differences: int = 40

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Replay tolerances must be positive and finite')


class Comparison:
    """Bound report size while retaining total mismatch counts and maxima."""
    def __init__(self, tolerances=None, resolution=None):
        self.t = tolerances or Tolerances()
        self.count = 0
        self.differences = []
        self.maximum_numeric_error = {}
        self.raw_static_multiplicity_differences = []
        self.known_occupied_refresh_differences = []
        self.static_resolution = resolution
        self.static_expected = self.static_actual = None

    def fail(self, path, reason, expected=None, actual=None):
        self.count += 1
        if len(self.differences) < self.t.max_reported_differences:
            self.differences.append(dict(path=path, reason=reason, expected=expected, actual=actual))

    def numeric(self, expected, actual, path, tolerance=None, angular=False):
        a, b = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            self.fail(path, 'shape_or_nonfinite', list(a.shape), list(b.shape))
            return
        d = a-b
        if angular:
            d = np.arctan2(np.sin(d), np.cos(d))
        error = float(np.abs(d).max(initial=0.))
        group = path.split('[', 1)[0]
        self.maximum_numeric_error[group] = max(error, self.maximum_numeric_error.get(group, 0.))
        if error > (self.t.scalar if tolerance is None else tolerance):
            self.fail(path, 'numeric_tolerance_exceeded', float(error), float(tolerance or self.t.scalar))

    def tree(self, expected, actual, path='value', tolerance=None):
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(expected) != set(actual):
                self.fail(path, 'dictionary_keys', sorted(expected), sorted(actual) if isinstance(actual, dict) else None)
                return
            for key in sorted(expected):
                child_tolerance = self.t.timestamp_s if key in (
                    'timestamp', 'first_seen_time', 'last_seen_time', 'observation_age') else tolerance
                self.tree(expected[key], actual[key], path+'.'+key, child_tolerance)
        elif isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
                self.fail(path, 'list_length', len(expected), len(actual) if isinstance(actual, (list, tuple)) else None)
                return
            for i, (a, b) in enumerate(zip(expected, actual)):
                self.tree(a, b, path+'['+str(i)+']', tolerance)
        elif isinstance(expected, bool) or expected is None or isinstance(expected, (str, int)):
            if type(expected) is not type(actual) or expected != actual:
                self.fail(path, 'exact_value', expected, actual)
        elif isinstance(expected, (float, np.floating)):
            self.numeric(expected, actual, path, tolerance)
        else:
            self.fail(path, 'unsupported_comparison_type', type(expected).__name__, type(actual).__name__)

    def observation(self, expected, actual, path):
        if self.static_resolution is not None and self.static_expected is None:
            self.static_expected = PartialGraph(self.static_resolution, expected['robot_pose'][1])
            self.static_actual = PartialGraph(self.static_resolution, actual['robot_pose'][1])
        if set(expected) != set(actual):
            self.fail(path, 'observation_keys', sorted(expected), sorted(actual))
        self.numeric(expected['timestamp'], actual['timestamp'], path+'.timestamp', self.t.timestamp_s)
        self.numeric(expected['robot_pose'][:3], actual['robot_pose'][:3], path+'.robot_position', self.t.position_m)
        self.numeric(expected['robot_pose'][3], actual['robot_pose'][3], path+'.robot_yaw', self.t.yaw_rad, angular=True)
        self.numeric(expected['camera_origin'], actual['camera_origin'], path+'.camera_origin', self.t.position_m)
        self.numeric(expected['camera_rotation'], actual['camera_rotation'], path+'.camera_rotation', self.t.camera_rotation)
        et, at = expected['tracks'], actual['tracks']
        if len(et) != len(at):
            self.fail(path+'.tracks', 'track_count', len(et), len(at))
        for i, (e, a) in enumerate(zip(et, at)):
            base = path+'.tracks['+str(i)+']'
            for key in sorted(set(e) | set(a)):
                if key not in e or key not in a:
                    self.fail(base+'.'+key, 'track_key_missing')
                elif key == 'position':
                    self.numeric(e[key], a[key], base+'.position', self.t.position_m)
                elif key == 'estimated_velocity' and e[key] is not None and a[key] is not None:
                    self.numeric(e[key], a[key], base+'.velocity', self.t.velocity_m_s)
                else:
                    self.tree(e[key], a[key], base+'.'+key, self.t.timestamp_s)
        # Several legal wall rays can emit the identical occupied certificate.
        # Their multiplicity is not a map update semantic. Keep all distinct
        # records (including contradictory states or different clocks/heights),
        # with no rounding, and compare the resulting evidence sets strictly.
        def evidence_set(records):
            values = {json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False):r for r in records}
            return sorted(values.values(), key=lambda r:(tuple(r['cell']), r['state'], r['source'],
                          r['timestamp'], r['height'], json.dumps(r, sort_keys=True)))
        es, ac = evidence_set(expected['static_updates']), evidence_set(actual['static_updates'])
        if len(expected['static_updates']) != len(actual['static_updates']):
            if len(self.raw_static_multiplicity_differences) < self.t.max_reported_differences:
                self.raw_static_multiplicity_differences.append(dict(path=path,
                    expected_raw=len(expected['static_updates']), actual_raw=len(actual['static_updates']),
                    expected_unique=len(es), actual_unique=len(ac)))
        def structural_key(record):
            return tuple(record['cell']),record['state'],record['source']
        def remove_proven_refreshes(records, other, side):
            kept=[];other_keys={structural_key(r) for r in other}
            for r in records:
                key=tuple(r['cell']);p=self.static_expected.cells.get(key) if self.static_expected else None
                q=self.static_actual.cells.get(key) if self.static_actual else None
                wall='observed_static_wall_inflated_025m'
                # This exception cannot establish geometry: BOTH histories
                # already certify this exact occupied cell/height/source.
                # A missing heartbeat changes only static last_seen metadata.
                row=expected if side=='stored' else actual
                redundant=(structural_key(r) not in other_keys and p and q and
                    r['state']==KNOWN_OCCUPIED and r['source']==wall and
                    p['state']==q['state']==KNOWN_OCCUPIED and p['source']==q['source']==wall and
                    p['height']==q['height']==r['height'] and p['first_seen']==q['first_seen'] and
                    r['timestamp']==row['timestamp'] and
                    max(p['last_seen'],q['last_seen'])<min(expected['timestamp'],actual['timestamp']))
                if redundant:
                    if len(self.known_occupied_refresh_differences)<self.t.max_reported_differences:
                        self.known_occupied_refresh_differences.append(dict(path=path,present_in=side,
                            record=copy.deepcopy(r),stored_prior=copy.deepcopy(p),native_prior=copy.deepcopy(q),
                            stored_last_seen_after=r['timestamp'] if side=='stored' else p['last_seen'],
                            native_last_seen_after=r['timestamp'] if side=='native' else q['last_seen']))
                else:kept.append(r)
            return kept
        original_es,original_ac=es,ac
        es=remove_proven_refreshes(original_es,original_ac,'stored')
        ac=remove_proven_refreshes(original_ac,original_es,'native')
        self.tree(es, ac, path+'.static_evidence_set', self.t.position_m)
        for key in ('sensor_protocol', 'observation_valid'):
            self.tree(expected[key], actual[key], path+'.'+key)
        if self.static_expected is not None:
            self.static_expected.update(expected);self.static_actual.update(actual)

    def labels(self, expected, actual, path):
        a, b = dict(expected), dict(actual)
        a.pop('node_id', None); b.pop('node_id', None)
        self.tree(a, b, path, self.t.distance_m)

    def report(self):
        return dict(passed=self.count == 0, mismatch_count=self.count,
                    differences=self.differences, maximum_numeric_error=self.maximum_numeric_error,
                    raw_static_multiplicity_differences=self.raw_static_multiplicity_differences,
                    known_occupied_refresh_differences=self.known_occupied_refresh_differences)


def query_categories(query, pose):
    result = set()
    for p in query['provenance']:
        source = p['source']; result.add(source)
        if source == 'human':
            result.add('human:'+p['intended_band'])
    d = np.asarray(query['waypoints'][-1])-np.asarray(pose[:3])
    if d[0]*math.cos(pose[3])-d[2]*math.sin(pose[3]) < -.3:
        result.add('backtracking')
    return result


def select_branches(queries, pose, max_queries=12):
    """Cover social bands, each source, backtracking, and a same-target yaw pair."""
    categories = [query_categories(q, pose) for q in queries]
    selected = []
    available = set().union(*categories) if categories else set()
    required = available & {'human:safe', 'human:intrusion', 'current_position', 'frontier', 'topology', 'goal', 'backtracking'}
    for category in sorted(required):
        ids = [i for i, values in enumerate(categories) if category in values]
        if not any(i in selected for i in ids):
            selected.append(ids[0])
    groups = {}
    for i, q in enumerate(queries):
        key = tuple(tuple(p) for p in q['waypoints'])
        groups.setdefault(key, []).append(i)
    pairs = [(a, b) for group in groups.values() for j, a in enumerate(group) for b in group[j+1:]
             if abs(wrap(queries[a]['final_yaw']-queries[b]['final_yaw'])) >= .18]
    if pairs:
        a, b = min(pairs, key=lambda pair: (sum(i not in selected for i in pair), pair))
        for i in (a, b):
            if i not in selected:selected.append(i)
    if len(selected) > max_queries:
        raise ValueError('max_queries_too_small_for_available_required_coverage')
    for i in range(len(queries)):
        if len(selected) >= max_queries:break
        if i not in selected:selected.append(i)
    return selected, dict(available=sorted(available), required=sorted(required),
                          selected=sorted(set().union(*(categories[i] for i in selected))) if selected else [],
                          same_target_heading_pair_available=bool(pairs),
                          same_target_heading_pair_selected=any(a in selected and b in selected for a, b in pairs))


def source_versions():
    result = implementation()
    paths = ('eventwm_graph/partial/replay_review.py', 'eventwm_graph/partial/acceptance.py',
             'habitat-lab/habitat/core/env.py', 'habitat-lab/habitat/core/embodied_task.py',
             'habitat-lab/habitat/tasks/rearrange/actions/actions.py',
             'habitat-lab/habitat/tasks/rearrange/rearrange_sensors.py',
             'habitat-lab/habitat/articulated_agents/robots/spot_robot.py',
             'habitat-lab/habitat/articulated_agents/manipulator.py', 'falcon/rearrange_sim_v2.py')
    result.update({p:file_digest(REPO/p) for p in paths})
    return result


def reconstruct_root_lineage(legal,queries,configuration):
    """Rebuild retained topology at the producer's legal proposal checkpoints.

    This consumes saved legal inputs exactly, independently of fresh native
    floating-point pose tolerances. It establishes exact query/topology lineage.
    """
    saved=legal['partial_graph']
    graph=PartialGraph(saved['resolution'],saved['height'],saved['clearance'])
    checkpoints=[];proposed=None
    for i,row in enumerate(legal['history']):
        graph.update(row)
        if i in PRODUCER_QUERY_CHECKPOINTS:
            proposed,_=generate_queries(graph,row['tracks'],row,legal['pointgoal'],
                configuration['queries_per_root'],safe_margin=configuration.get('safe_margin_m',.2),
                intrusion_radius=configuration.get('intrusion_radius_m',.96))
            checkpoints.append(i)
    actual=graph.snapshot()
    if actual!=saved:raise ValueError('root_global_topology_lineage_not_reconstructable')
    if proposed!=queries:raise ValueError('root_full_query_generation_not_reconstructable')
    return dict(passed=True,producer_checkpoints=list(PRODUCER_QUERY_CHECKPOINTS),
                checkpoints_replayed=checkpoints,global_nodes=len(actual['global_nodes']),
                global_edges=len(actual['global_edges']),full_queries_reproduced=len(queries))


def _official_measure(rt):
    from falcon.additional_metric import PersonalSpaceCompliance
    measure = PersonalSpaceCompliance(sim=rt.sim, config=SimpleNamespace(use_geo_distance=True))
    # Appending this read-only measure lets real Env.step call the installed
    # update after num_steps. It never directs actions or changes termination.
    rt.task.measurements.measures['psc'] = measure
    measure.reset_metric(episode=rt.episode, task=rt.task, observations=rt.obs)
    return measure


def _guard(graph, pose, action):
    p, yaw = np.asarray(pose[:3]), pose[3]
    if action == 'forward':
        end = p+[FORWARD_STEP*math.cos(yaw), 0., -FORWARD_STEP*math.sin(yaw)]
        samples = max(1, int(math.ceil(FORWARD_STEP/.03)))
        return graph.segment_free(p, end) and all(
            static_body_clearance(graph, p+(end-p)*t, yaw) for t in np.linspace(0., 1., samples+1))
    if action in ('left', 'right'):
        return turn_clearance(graph, p, yaw, yaw+(TURN_STEP if action == 'left' else -TURN_STEP))
    raise ValueError('Unsupported query action: '+str(action))


def _replay_branch(hmap, episode, seed, legal, private, nodes, labels, tape, query, outcome, tolerances):
    from ..stage1.runtime import NativeEpisode
    compare = Comparison(tolerances, resolution=legal['partial_graph']['resolution'])
    began = time.perf_counter()
    with NativeEpisode(hmap, copy.deepcopy(episode), seed) as rt:
        sensor = GeometryObserver(rt, legal['partial_graph']['resolution'])
        rt.observation = sensor.observe
        measure = _official_measure(rt)
        current = sensor.observe()
        graph = PartialGraph(sensor.resolution, current['robot_pose'][1])
        graph.update(current)
        compare.observation(legal['history'][0], current, 'prefix[0]')
        prefix_compliant = 0
        for i, action in enumerate(legal['history_actions'], 1):
            if rt.env.episode_over:
                compare.fail('prefix', 'native_terminated_before_root', len(legal['history_actions']), i-1)
                break
            before = measure._compliant_steps
            current = rt.step(action); graph.update(current)
            actual = step_labels(rt, current['timestamp'])
            compliant = True if rt.count == 0 else measure._compliant_steps > before
            prefix_compliant += int(compliant)
            compare.tree(not private['prefix_labels'][i-1]['psc_violation'], compliant, 'prefix['+str(i)+'].installed_psc_compliant')
            compare.labels(private['prefix_labels'][i-1], actual, 'prefix['+str(i)+'].labels')
            compare.observation(legal['history'][i], current, 'prefix['+str(i)+']')
        if rt.frames != len(legal['history_actions']):
            return dict(query_id=query['query_id'], seconds=time.perf_counter()-began, **compare.report())
        compare.tree(legal['partial_graph']['cells'], graph.snapshot()['cells'], 'root.cells', tolerances.position_m)
        compare.tree(legal['partial_graph']['visited'], graph.visited, 'root.visited', tolerances.position_m)
        try:
            validate_global_graph(graph, legal['partial_graph'], current['robot_pose'])
        except Exception as error:
            compare.fail('root.global_graph', repr(error))
        human_positions = lambda: np.asarray([h['position'] for h in rt.truth()], dtype=float).reshape((-1, 3))
        compare.numeric(tape['positions'][0], human_positions(), 'root.human_positions', tolerances.position_m)
        compare.numeric(tape['timestamps'][0], current['timestamp'], 'root.timestamp', tolerances.timestamp_s)
        poses = [current['robot_pose']]; actual_labels = []; actual_actions = []
        reason = 'query_complete'
        horizon = private['identity']['collection_config']['max_query_steps']
        for i, action in enumerate(query['compiled']['actions']):
            if i >= horizon:
                reason = 'query_timeout'; break
            if not _guard(graph, current['robot_pose'], action):
                reason = 'route_invalidated'; break
            if rt.env.episode_over:
                reason = 'environment_termination'; break
            before = measure._compliant_steps
            current = rt.step(action); delta = graph.update(current)
            actual = step_labels(rt, current['timestamp'])
            hit = bool(rt.env.get_metrics()['human_collision'])
            actual.update(episode_over=bool(rt.env.episode_over), official_human_collision=hit)
            actual_labels.append(actual); poses.append(current['robot_pose']); actual_actions.append(action)
            compliant = True if rt.count == 0 else measure._compliant_steps > before
            path = 'query['+str(i+1)+']'
            compare.tree(not actual['psc_violation'], compliant, path+'.installed_psc_compliant')
            if i >= len(outcome['node_ids']):
                compare.fail(path, 'unexpected_native_extra_action', len(outcome['node_ids']), i+1)
            else:
                nid = outcome['node_ids'][i]; expected = nodes[nid]
                compare.observation(expected['observation'], current, path+'.observation')
                compare.labels(labels[nid], actual, path+'.labels')
                compare.tree(expected['graph_delta'], delta, path+'.graph_delta')
            if i+1 >= len(tape['timestamps']):
                compare.fail(path, 'native_action_beyond_saved_tape')
            else:
                compare.numeric(tape['positions'][i+1], human_positions(), path+'.human_positions', tolerances.position_m)
                compare.numeric(tape['timestamps'][i+1], current['timestamp'], path+'.timestamp', tolerances.timestamp_s)
            if rt.env.episode_over:
                reason = 'human_collision' if hit else 'episode_action_limit'; break
        compare.tree(len(outcome['node_ids']), len(actual_actions), 'executed_steps')
        compare.tree(outcome['summary']['termination'], reason, 'termination')
        successor = dict(timestamp=current['timestamp'], robot_pose=current['robot_pose'],
                         tracks=current['tracks'], graph_revision=graph.revision)
        compare.tree(outcome['successor'], successor, 'successor', tolerances.position_m)
        summary = summarize(actual_labels, poses, tape['timestamps'][0], query['compiled']['actions'],
                            reason, query, legal['pointgoal'], prefix_compliant, len(legal['history_actions']))
        compare.tree(outcome['summary'], summary, 'summary', tolerances.distance_m)
        installed_psc = float(measure.get_metric()) if rt.frames else None
        if installed_psc is not None:
            compare.numeric(outcome['summary']['episode_prefix_plus_query_psc'], installed_psc,
                            'installed_episode_prefix_plus_query_psc', 1e-12)
        return dict(query_id=query['query_id'], source=query['provenance'], prefix_steps=rt.frames-len(actual_actions),
                    prefix_observations_compared=len(legal['history']), fresh_environment=True,
                    executed_steps=len(actual_actions),
                    termination=reason, installed_episode_psc=installed_psc, seconds=time.perf_counter()-began,
                    **compare.report())


def review_root(folder, max_queries=12, repeats=2, tolerances=None):
    """Validate and repeatedly replay one train-scene root in fresh environments."""
    tolerances = tolerances or Tolerances()
    if repeats < 2:raise ValueError('At least two independent replays are required')
    folder = Path(folder).resolve()
    manifest = read_json(folder/'manifest.json')
    # Check only public scene identity before opening any episode/root contents.
    if scene_split(manifest['scene_id']) != 'train' or manifest['split'] != 'train':
        raise ValueError('held_out_scene_refused')
    sources = source_versions()
    for name, sha in manifest['version'].items():
        if sources.get(name) != sha:raise ValueError('stored_source_mismatch:'+name)
    validate_root(folder)
    legal = read_gzip(folder/'legal_root.json.gz')
    private = read_gzip(folder/'privileged_initialization.json.gz')
    queries = read_gzip(folder/'queries.json.gz'); outcomes = read_gzip(folder/'outcomes.json.gz')
    nodes = read_gzip(folder/'future_observations.json.gz')
    labels = read_gzip(folder/'privileged_step_labels.json.gz')
    lineage=reconstruct_root_lineage(legal,queries,manifest['collection_config'])
    with np.load(folder/'privileged_future.npz', allow_pickle=False) as f:
        tape = {key:f[key].copy() for key in ('timestamps', 'positions')}
    import habitat_sim  # Installed Habitat import order, before episode types.
    from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
    from ..benchmark.map import load_certified_map
    ep = private['episode']
    episode = NavigationEpisode(episode_id=ep['episode_id'], scene_id=ep['scene_id'],
        start_position=ep['start_position'], start_rotation=ep['start_rotation'],
        goals=[NavigationGoal(position=g['position']) for g in ep['goals']], info=copy.deepcopy(ep['info']))
    hmap = load_certified_map(private['source_map_manifest'])
    if hmap.metadata['scene_id'] != manifest['scene_id']:
        raise ValueError('private_map_scene_mismatch')
    if Path(hmap.navigation.path).resolve() != Path(ep['info']['graph_navmesh']).resolve():
        raise ValueError('private_navmesh_mismatch')
    selected, coverage = select_branches(queries, legal['history'][-1]['robot_pose'], max_queries)
    results = []
    for repeat in range(repeats):
        # Reverse order to detect any module-global state surviving fresh Env.
        order = selected if repeat % 2 == 0 else list(reversed(selected))
        for i in order:
            try:
                result = _replay_branch(hmap, episode, manifest['seed'], legal, private, nodes, labels, tape,
                                        queries[i], outcomes[i], tolerances)
            except Exception as error:
                result = dict(query_id=queries[i]['query_id'], passed=False, exception=repr(error))
            result.update(repeat=repeat, query_index=i)
            results.append(result)
    unchanged = sources == source_versions()
    return dict(schema=REVIEW_VERSION, root_id=manifest['root_id'], folder=str(folder),
                scene_id=manifest['scene_id'], status='PASS' if results and all(r['passed'] for r in results) and unchanged else 'FAIL',
                tolerances=asdict(tolerances), comparison_convention='per-scalar absolute; wrapped yaw',
                static_evidence_comparison=('distinct exact records; duplicate multiplicity diagnostic; unmatched occupied '
                    'refresh allowed ONLY with identical prior occupied state/height/source in BOTH histories'),
                official_psc_configuration=dict(use_geo_distance=True, radius_m=1., comparison='strict_less_than',
                                                updates='post_action_endpoints; root_excluded'),
                source_versions=sources, sources_unchanged=unchanged,
                manifest_sha256=file_digest(folder/'manifest.json'), source_map_manifest_sha256=file_digest(private['source_map_manifest']),
                native_navmesh_sha256=file_digest(hmap.navigation.path),
                root_lineage=lineage,selected_query_indices=selected, repeats=repeats, coverage=coverage, branches=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-queries', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    began = time.perf_counter(); reports = []
    for folder in args.roots:
        try:
            reports.append(review_root(folder, args.max_queries, args.repeats))
        except Exception as error:
            reports.append(dict(folder=str(Path(folder).resolve()), status='FAIL', exception=repr(error)))
    result = dict(schema=REVIEW_VERSION, status='PASS' if reports and all(r['status']=='PASS' for r in reports) else 'FAIL',
                  seconds=time.perf_counter()-began, roots=reports, training_started=False, production_started=False)
    write_json(args.output, result)
    print(result['status']+' '+str(Path(args.output).resolve()), flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
