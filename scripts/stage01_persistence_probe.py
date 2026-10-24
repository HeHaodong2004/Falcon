#!/usr/bin/env python
"""Bounded fresh-vs-retained native scene differential; no data publication."""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eventwm_graph.config import file_digest, read_json, write_json


def probe(output, mode='persistent'):
    import habitat_sim  # Installed build requires this import order.
    from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
    import numpy as np
    from eventwm_graph.benchmark.map import load_certified_map
    from eventwm_graph.partial.map import PartialGraph
    from eventwm_graph.partial.metrics import step_labels
    from eventwm_graph.partial.observer import GeometryObserver
    from eventwm_graph.partial.persistent_runtime import SceneEpisodePool
    from eventwm_graph.partial.replay_review import Comparison, source_versions
    from eventwm_graph.stage1.runtime import NativeEpisode
    from eventwm_graph.stage1.storage import read_gzip, scene_split

    roots = ('39/396f06cac79b03e323811d673a1de1108a7df9376c22cbdcefbae0ed70009c77',
             'f8/f8217644d7b3f87124c2cf111054a4be6f5b077053b9df827e8dc5dd8225eb0f',
             'ca/ca837a85ff371e4e26983a5a102971853edf04886488eb47ba2ac76eb445542e',
             '7e/7e42d5ee073de11c77d8404b5ddab2c7f984eeea7cf2c77de69673a1e92fb85d')
    base = ROOT/'output/stage01_partial_graph_query_20261020/pilot/dataset/roots'
    cases, maps = [], {}
    for suffix in roots:
        folder = base/suffix
        manifest = read_json(folder/'manifest.json')
        if manifest['split'] != 'train' or scene_split(manifest['scene_id']) != 'train':
            raise ValueError('held_out_scene_refused')
        private = read_gzip(folder/'privileged_initialization.json.gz')
        record = private['episode']
        episode = NavigationEpisode(episode_id=record['episode_id'], scene_id=record['scene_id'],
            start_position=record['start_position'], start_rotation=record['start_rotation'],
            goals=[NavigationGoal(position=g['position']) for g in record['goals']], info=copy.deepcopy(record['info']))
        path = private['source_map_manifest']
        if path not in maps:
            maps[path] = load_certified_map(path)
        cases.append((maps[path], episode, manifest['seed'], folder))
    actions = ('forward', 'left', 'forward', 'right', 'right', 'forward')

    def run(factory):
        records = []
        for hmap, episode, seed, folder in cases:
            start = time.perf_counter()
            with factory(hmap, copy.deepcopy(episode), seed) as rt:
                initialization = time.perf_counter()-start
                sensor = GeometryObserver(rt, hmap.robot_config.collision_reference/6)
                rt.observation = sensor.observe
                graph = None
                frames = []
                for action in (None, *actions):
                    if rt.env.episode_over:
                        break
                    row = sensor.observe() if action is None else rt.step(action)
                    if graph is None:
                        graph = PartialGraph(sensor.resolution, row['robot_pose'][1])
                    graph.update(row)
                    actors = []
                    for actor in rt.sim.agents_mgr.articulated_agents_iter:
                        actors.append(dict(transform=np.asarray(actor.sim_obj.transformation).tolist(),
                            joints=np.asarray(actor.sim_obj.joint_positions).tolist(),
                            velocities=np.asarray(actor.sim_obj.joint_velocities).tolist(),
                            offset=np.asarray(getattr(actor, 'offset_transform', np.eye(4))).tolist()))
                    frames.append(dict(observation=row, actors=actors,
                                       labels=step_labels(rt, row['timestamp']), graph=graph.snapshot(),
                                       episode_over=bool(rt.env.episode_over)))
                samples = [np.asarray(rt.sim.pathfinder.get_random_navigable_point()).tolist() for _ in range(6)]
                records.append(dict(seed=seed, folder=str(folder), frames=frames, future_rng_samples=samples,
                    simulator_identity=id(rt.sim), initialization_seconds=initialization,
                    total_seconds=time.perf_counter()-start))
        return records

    began = time.perf_counter()
    versions = source_versions()
    versions['eventwm_graph/partial/persistent_runtime.py'] = file_digest(ROOT/'eventwm_graph/partial/persistent_runtime.py')
    versions['scripts/stage01_persistence_probe.py'] = file_digest(__file__)
    fresh = run(NativeEpisode)
    if mode == 'persistent':
        pool = SceneEpisodePool()
        try:
            retained = run(pool)
        finally:
            pool.close()
        optimization = pool.performance
    elif mode == 'configuration-cache':
        from eventwm_graph.partial.configuration_cache import ExactConfigurationCache
        cache = ExactConfigurationCache()

        @contextmanager
        def cached(hmap, episode, seed):
            with cache.scope():
                with NativeEpisode(hmap, episode, seed) as rt:
                    yield rt

        retained = run(cached)
        optimization = cache.performance
        versions['eventwm_graph/partial/configuration_cache.py'] = file_digest(ROOT/'eventwm_graph/partial/configuration_cache.py')
        versions.update({'configuration/'+name:sha for name,sha in cache.sources.items()})
    else:
        raise ValueError('unknown_probe_mode')
    comparisons = []
    for reference, actual in zip(fresh, retained):
        check = Comparison()
        check.tree(len(reference['frames']), len(actual['frames']), 'frame_count')
        for i, (a, b) in enumerate(zip(reference['frames'], actual['frames'])):
            check.observation(a['observation'], b['observation'], 'frame['+str(i)+']')
            check.tree(a['actors'], b['actors'], 'actors['+str(i)+']', 1e-5)
            check.labels(a['labels'], b['labels'], 'labels['+str(i)+']')
            check.tree(a['graph'], b['graph'], 'graph['+str(i)+']', 1e-5)
            check.tree(a['episode_over'], b['episode_over'], 'episode_over['+str(i)+']')
        check.tree(reference['future_rng_samples'], actual['future_rng_samples'], 'native_rng_samples', 1e-7)
        comparisons.append(dict(seed=reference['seed'], frames=len(reference['frames']),
            fresh_initialization_seconds=reference['initialization_seconds'],
            retained_initialization_seconds=actual['initialization_seconds'],
            fresh_total_seconds=reference['total_seconds'], retained_total_seconds=actual['total_seconds'],
            **check.report()))
    report = dict(status='PASS' if all(row['passed'] for row in comparisons) else 'FAIL',
                  scope='Four train roots, two native scenes; same-scene crowd changes 1->0->1; six real native actions per root',
                  elapsed_seconds=time.perf_counter()-began, actions=list(actions),
                  mode=mode, optimization=optimization, source_versions=versions, cases=comparisons)
    write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('persistent', 'configuration-cache'), default='persistent')
    arguments = parser.parse_args()
    result = probe(arguments.output, arguments.mode)
    print(json.dumps(result, indent=2), flush=True)
    sys.exit(0 if result['status'] == 'PASS' else 1)
