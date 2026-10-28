"""Exactness and speed of the fast legal sensor (querygen/fastsense.py, v13).

(a) dual mode: during a short Oracle search (prefix cache off, so every step is observed) each observation is made by
    the accepted pure observer AND by the fast observer, each from its own copy of the sensor state; the rows
    (pose, camera, detections/tracks, certified FREE cells, inflated wall cells, ego cell -- content and order), the
    trackers and the certified-cell sets must be identical;
(b) end to end: the v12 C1-K10 Oracle search with the prefix cache, reference vs fast sensor: identical search digest
    (edges, outcomes, labels, final actions and decisions); sensor seconds of both.

  python scripts/querygen_run.py tests.querygen.test_fastsense KEY SUITE EDGES_DUAL EDGES_E2E [KEY SUITE ...]
"""
import json
import sys
import time

V12_GEN = dict(tc_variants=('goal', 'st_goal'), vc_variants=('goal', 'st_goal'), st_goal=True,
               maneuver_spawn_only=False, goal_unstick=6, bump_memory_m=0.25, evade=True, unstick_keep=True,
               bump_front=True, static_humans=True, stop_strict=True, unstick_progress=True, goal_relax_m=0.5,
               stop_closest_m=1.2)


def same_row(a, b):
    keys = ('timestamp', 'robot_pose', 'tracks', 'camera_origin', 'camera_rotation', 'sensor_protocol',
            'observation_valid')
    diff = [k for k in keys if a[k] != b[k]]
    if list(a['static_updates']) != list(b['static_updates']):
        diff.append('static_updates')
    return diff


def main():
    from eventwm_graph.querygen import episode as E
    from eventwm_graph.querygen.legal import LegalSensor
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
    from eventwm_graph.querygen.oracle import Oracle, OracleConfig
    from eventwm_graph.querygen.dynamics import FAMILIES
    from tests.querygen.test_resumable_portfolio import digest
    args = sys.argv[1:]
    ok = True
    dyn = FAMILIES['native']

    class Dual:
        """Reference + fast observer on the same simulator; returns the reference rows."""
        def __init__(self, task, resolution=None, private_cache=None, pure_caches=True, fast=None):
            self.ref = LegalSensor(task, private_cache=private_cache, pure_caches=pure_caches, fast=False)
            self.fast = LegalSensor(task, private_cache={}, pure_caches=pure_caches, fast=True)
            self.observer = self.ref.observer
            self.n, self.bad, self.t_ref, self.t_fast = 0, [], 0., 0.

        def observe(self, t):
            c = time.perf_counter(); a = self.ref.observe(t); self.t_ref += time.perf_counter()-c
            c = time.perf_counter(); b = self.fast.observe(t); self.t_fast += time.perf_counter()-c
            self.n += 1
            diff = same_row(a, b)
            ro, fo = self.ref.observer, self.fast.observer
            if ro.seen != fo.seen:
                diff.append('seen')
            if ro.tracker.__dict__ != fo.tracker.__dict__:
                diff.append('tracker')
            if diff and len(self.bad) < 20:
                self.bad.append((self.n, t, diff))
            return a

        def state(self):
            return (self.ref.state(), self.fast.state())

        def restore(self, st):
            self.ref.restore(st[0]); self.fast.restore(st[1])

    for i in range(0, len(args), 4):
        key, suite, e_dual, e_e2e = args[i], args[i+1], int(args[i+2]), int(args[i+3])
        spec = [s for s in load(suite) if s.key == key][0]
        gen = QueryGenerator(dyn, GeneratorConfig(**V12_GEN))
        # (a) dual
        if e_dual:
            E.LegalSensor = Dual
            rt = E.EpisodeRuntime(spec, 'orca', dyn, strict=False); rt.sim.check_static = False
            E.LegalSensor = LegalSensor
            cfg = OracleConfig(heading_h=True, escape_penalty=80, prefix_cache=0, max_edges=e_dual, max_seconds=40000,
                               portfolio=[{'weight': 3.5}, {'st_radius': 0.6, 'st_radius_opt': 0}])
            Oracle(rt, gen, cfg).search()
            d = rt.sensor
            perf = d.fast.observer.performance
            row = dict(key=key, mode='dual', observations=d.n, mismatches=len(d.bad), first=d.bad[:3],
                       ref_s=round(d.t_ref, 1), fast_s=round(d.t_fast, 1),
                       memo_skipped=perf.get('memo_skipped'), classified_rays=perf.get('classified_rays'),
                       ref_support_rays=d.ref.observer.performance.get('support_rays'),
                       fast_support_rays=perf.get('support_rays'))
            print(json.dumps(row), flush=True)
            ok = ok and not d.bad and d.n > 0
            rt.close()
        # (b) end to end
        if e_e2e:
            out = {}
            for fast in (False, True):
                LegalSensor.FAST_SENSE = fast
                rt = E.EpisodeRuntime(spec, 'orca', dyn, strict=False); rt.sim.check_static = False
                assert rt.sensor.fast == fast
                cfg = OracleConfig(heading_h=True, escape_penalty=80, prefix_cache=48, max_edges=e_e2e,
                                   max_seconds=40000, portfolio=[{'weight': 3.5}, {'st_radius': 0.6, 'st_radius_opt': 0}])
                t = time.perf_counter()
                res = Oracle(rt, QueryGenerator(dyn, GeneratorConfig(**V12_GEN)), cfg).search()
                dt = time.perf_counter()-t
                dg, payload = digest(res)
                out[fast] = dict(digest=dg, success=payload['success'], edges=payload['children'],
                                 search_s=round(dt, 1), sensor_s=round(rt.perf['sensor_s'], 1),
                                 legal_s=round(rt.perf['legal_s'], 1), steps=rt.perf['steps'])
                rt.close()
            LegalSensor.FAST_SENSE = False
            same = out[False]['digest'] == out[True]['digest']
            print(json.dumps(dict(key=key, mode='e2e', identical=same, ref=out[False], fast=out[True])), flush=True)
            ok = ok and same
    print('FASTSENSE_EXACT' if ok else 'FASTSENSE_MISMATCH')


if __name__ == '__main__':
    main()
