"""OFFLINE PRIVILEGED ORACLE reference; not a real WM input contract.

Inputs are the current graph, base trajectory, trial camera poses, static scene
mesh and navigation certificate support. Recorded updates are scoring targets
only. No label/observation-result cache is an inference input. This first
reference intentionally excludes dynamic actor occlusion and measures its error.
"""
import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace
import time
import habitat_sim
import numpy as np
from .diagnose import OUT, HF, read, save, sha, source_rows, member
from .full_evaluate import labels
from eventwm_graph.native import StaticGeometry
from eventwm_graph.partial.fast_observer import FastGeometryObserver, StaticProbeIndex
from eventwm_graph.partial.observer import in_frustum
from eventwm_graph.partial.native_batch import NativeRayBatch
from eventwm_graph.stage2_segment.data import replay
from eventwm_graph.stage2_quality.features import Features
from scripts.collector.observation import ObservationConfig
from scripts.stage2_diagnostic.common import extended_rates


def asset_path(value):
    path = Path(value)
    if path.exists():
        return path.resolve()
    for root in [Path('/home/haodong/workspace/projects/eventwm-falcon'), Path('/home/haodong/workspace/baselines/Falcon')]:
        if (root/path).exists():
            return (root/path).resolve()
    raise FileNotFoundError(value)


class CertificateBank(FastGeometryObserver):
    def __init__(self, geometry, resolution, namespace):
        self.sim = geometry.sim
        self.resolution = float(resolution)
        self.cache_namespace = namespace
        self.cache = {}
        self.probe_cache = {}
        self.performance = dict(support_hits=0, support_misses=0, probe_hits=0, probe_misses=0, private_geometry_seconds=0., indexed_support_reuses=0)
        radius = .25+resolution/math.sqrt(2)
        self._halo = np.array([[math.cos(a)*radius, 0., math.sin(a)*radius] for a in np.linspace(0., 2*math.pi, 16, endpoint=False)])
        self.indices = {}

    def lookup(self, keys, height):
        index = self.indices.setdefault(round(height/.1), StaticProbeIndex())
        return index.lookup(keys, self, height)


class NativeStaticPredictor:
    def __init__(self, scene, navmesh, resolution, sensor):
        self.scene, self.navmesh = asset_path(scene), asset_path(navmesh)
        self.geometry = StaticGeometry(str(self.scene))
        if not self.geometry.sim.pathfinder.load_nav_mesh(str(self.navmesh)):
            raise ValueError('cannot load the actual collection navmesh')
        self.resolution = float(resolution)
        self.bank = CertificateBank(self.geometry, resolution, (str(self.scene), str(self.navmesh), resolution))
        os.environ.setdefault('EVENTWM_NATIVE_BATCH_BUILD', '/home/haodong/workspace/projects/eventwm-falcon/output/stage01_acceleration_20261021/native_batch/build')
        self.rays = NativeRayBatch(self.geometry.sim)
        self.sensor = SimpleNamespace(**sensor)
        r = self.resolution
        span = int(math.ceil(.25/r))+1
        self.wall_offsets = np.array([(x, z) for x in range(-span, span+1) for z in range(-span, span+1)])
        self.wall_directions = np.array([[x, y, -1.] for x in np.linspace(-.95, .95, 35) for y in (-.5, 0., .5)])

    def close(self):
        self.geometry.close()

    def predict(self, keys, path, origins, rotations):
        assert len(path) == len(origins) == len(rotations)
        n = len(keys)
        hard, seen = np.zeros(n, np.uint8), np.zeros(n, bool)
        heights = np.zeros(n, np.float64)
        lookup = {tuple(k): i for i, k in enumerate(keys)}
        r = self.resolution
        for frame in range(1, len(path)):
            base, origin, rotation = path[frame, 1:4], origins[frame], rotations[frame]
            centers = np.column_stack([(keys[:, 0]+.5)*r, np.full(n, base[1]+.55), (keys[:, 1]+.5)*r])
            eligible = np.flatnonzero(~seen & in_frustum(centers, origin, rotation, self.sensor))
            if len(eligible):
                selected, targets, yy = self.bank.lookup(keys[eligible], base[1])
                eligible = eligible[selected]
                mask = in_frustum(targets.reshape(-1, 3), origin, rotation, self.sensor).reshape(len(targets), -1).all(1) if len(targets) else np.zeros(0, bool)
                remaining = np.flatnonzero(mask)
                for probe in range(63):
                    if not len(remaining):
                        break
                    vectors = targets[remaining, probe]-origin
                    lengths = np.sqrt((vectors[:, None, :]@vectors[:, :, None])[:, 0, 0])
                    clear = self.rays.clear_rays(origin, vectors/lengths[:, None], lengths, ())
                    remaining = remaining[clear]
                ids = eligible[remaining]
                hard[ids] = 1
                seen[ids] = True
                heights[ids] = yy[remaining]
            directions = self.wall_directions@rotation.T
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            hits = self.rays.nearest_hits(origin, directions, np.full(len(directions), 10.), ())
            for point in hits['points'][hits['valid']]:
                if not base[1]+.2 < point[1] < base[1]+1.4:
                    continue
                near = np.floor(point[[0, 2]]/r).astype(np.int64)+self.wall_offsets
                dist = np.linalg.norm((near+.5)*r-point[[0, 2]], axis=1)
                for key in near[dist <= .25]:
                    i = lookup.get(tuple(key))
                    if i is not None:
                        hard[i] = 2
                        seen[i] = False
                        heights[i] = base[1]
        return hard, heights


def trial_cameras(legal, nodes, start, end):
    ancestry = []
    while end != start:
        ancestry.append(nodes[end]['observation'])
        end = nodes[end]['parent']
    first = legal['history'][-1] if start is None else nodes[start]['observation']
    observations = [first]+list(reversed(ancestry))
    return np.array([[o['timestamp'], *o['robot_pose']] for o in observations]), np.array([o['camera_origin'] for o in observations]), np.array([o['camera_rotation'] for o in observations])


def run(roots=4, windows=2, split='dev', name='native_geometry_probe',backend='native',device='cuda:0'):
    rows, manifest = source_rows(split)
    if split == 'dev':
        pilot = set(read(HF/'protocol.json')['pilot_selection_roots'])
        rows = [r for r in rows if r['root_id'] in pilot]
    cohorts = [[], []]
    for row in rows:
        private = member(row, 'privileged_initialization.json.gz')
        cohorts[int(private['episode']['info']['human_num'] > 0)].append((row, private))
        if roots>0 and min(map(len, cohorts)) >= (roots+1)//2:
            break
    selected = (cohorts[0][:(roots+1)//2]+cohorts[1][:roots//2]) if roots>0 else cohorts[0]+cohorts[1]
    reports = []
    tick = time.perf_counter()
    for row, private in selected:
        legal = member(row, 'legal_root.json.gz')
        nodes = member(row, 'future_observations.json.gz')
        ep = private['episode']
        print('load', row['scene_id'], 'human_count', ep['info']['human_num'], 'scene', ep['scene_id'], 'navmesh', ep['info']['graph_navmesh'], flush=True)
        cls=NativeStaticPredictor
        kwargs={}
        if backend=='gpu':
            from .geometry_api import GeometryPredictor
            cls=GeometryPredictor
            kwargs['device']=device
        engine = cls(ep['scene_id'], ep['info']['graph_navmesh'], legal['partial_graph']['resolution'], legal['sensor'],**kwargs)
        try:
            count = 0
            for ex in replay(row, 'mixed'):
                if 'audit' in ex:
                    continue
                p, origin, rotation = trial_cameras(legal, nodes, ex['start_node'], ex['end_node'])
                np.testing.assert_array_equal(p, ex['trajectory'])
                keys = Features(ex['graph'], p).candidates('both')
                before = time.perf_counter()
                if backend=='native':pred, heights = engine.predict(keys, p, origin, rotation)
                else:
                    delta=engine.predict(partial_graph=ex['graph'],trajectory=p,camera_origins=origin,camera_rotations=rotation)
                    np.testing.assert_array_equal(keys,delta['candidate_cells'])
                    pred,heights=delta['states'],delta['heights']
                    assert delta['future_partial_graph']['hypothetical'] is True
                seconds = time.perf_counter()-before
                y, _, missing = labels(keys, ex['target'])
                cm = np.bincount(3*y.astype(np.int64)+pred, minlength=9).reshape(3, 3)
                for state, height in missing:
                    cm[state, 0] += 1
                rr = dict(root_id=row['root_id'], scene_id=row['scene_id'], human_count=ep['info']['human_num'], start_node=ex['start_node'], end_node=ex['end_node'], poses=len(p), candidates=len(keys), target=len(ex['target']), confusion=cm.tolist(), metrics=extended_rates(cm), native_seconds=seconds)
                reports.append(rr)
                print('reference', row['scene_id'], ex['start_node'], ex['end_node'], rr['metrics']['label_aware_delta_iou'], 'seconds', seconds, flush=True)
                dest = OUT/'geometry'/name
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest/('%03d.npz'%len(reports)), keys=keys, y=y, prediction=pred, heights=heights, path=p, origins=origin, rotations=rotation)
                save(dest/'report.json', dict(contract='REVISED: current graph + base trajectory + supplied trial camera poses + static scene mesh/navmesh. Actual recorded updates are scoring targets only. Active-human occlusion excluded.',backend=backend,source_manifest_sha256=sha(manifest), windows=reports, seconds=time.perf_counter()-tick))
                count += 1
                if windows>0 and count >= windows:
                    break
        finally:
            engine.close()
    from collections import defaultdict
    scenes=defaultdict(list)
    root_confusions=defaultdict(list)
    root_scenes={r['root_id']:r['scene_id']for r in reports}
    for r in reports:root_confusions[r['root_id']].append(r['confusion'])
    for rid,values in root_confusions.items():scenes[root_scenes[rid]].append(np.mean(values,axis=0))
    cm=np.mean([np.mean(values,axis=0)for values in scenes.values()],axis=0)
    summary=dict(roots=len(root_confusions),scenes=len(scenes),windows=len(reports),hierarchical=extended_rates(cm),pooled=extended_rates(np.sum([r['confusion']for r in reports],axis=0)),pooled_confusion=np.sum([r['confusion']for r in reports],axis=0).tolist(),seconds=time.perf_counter()-tick)
    save(OUT/'geometry'/name/'summary.json',summary)
    print('SUMMARY',summary,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--roots', type=int, default=4)
    p.add_argument('--windows', type=int, default=2)
    p.add_argument('--split', default='dev')
    p.add_argument('--name', default='native_geometry_probe')
    p.add_argument('--backend',choices=['native','gpu'],default='native')
    p.add_argument('--device',default='cuda:0')
    run(**vars(p.parse_args()))
