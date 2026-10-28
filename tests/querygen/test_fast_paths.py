"""Exactness of the throughput fast paths (2026-10-28):
1. cached benchmark config == freshly composed config (YAML-identical) for HM3D and MP3D episodes;
2. authoritative replay with a shared (warm) pure sensor cache and the cached human tape == the original replay
   (fresh caches, freshly recorded tape): identical actions, trajectory, official metrics, shadow checks;
3. lean replay (no shadow) == the same actions, trajectory poses and official metrics.

  python scripts/querygen_run.py tests.querygen.test_fast_paths RESULT_JSON [RESULT_JSON ...]
"""
import json
import os
import sys
import time


def main():
    from omegaconf import OmegaConf
    from eventwm_graph.querygen import native
    from eventwm_graph.querygen.native import EpisodeSpec
    from eventwm_graph.querygen.episode import replay, EpisodeRuntime
    from eventwm_graph.querygen.executor import Query
    from eventwm_graph.querygen.dynamics import FAMILIES
    native._install()
    files = sys.argv[1:]
    ok = True
    for f in files:
        r = json.load(open(f))
        spec = EpisodeSpec(**r['spec'])
        # 1. config cache
        native._BASE_CONFIGS.clear()
        a = OmegaConf.to_yaml(native.benchmark_config(spec))
        b = OmegaConf.to_yaml(native.benchmark_config(spec))          # from the cache
        native._BASE_CONFIGS.clear()
        c = OmegaConf.to_yaml(native.benchmark_config(spec))
        cfg_ok = a == b == c
        dec = [dict(query=Query(x['kind'], via=tuple(tuple(v) for v in x['via']), heading=x['heading'],
                                duration=x['duration'], speed=x['speed'], label=x['label'], meta=x['meta'] or {}),
                    cont=x['cont'], horizon=x['horizon']) for x in r['decisions']]
        dyn = FAMILIES[r['dynamics']]
        # 2. original replay: fresh caches, recorded tape
        saved = os.environ.pop('QG_TAPE_CACHE', None)
        t = time.perf_counter(); r0 = replay(spec, r['planner'], dyn, dec); t0 = time.perf_counter()-t
        if saved is not None:
            os.environ['QG_TAPE_CACHE'] = saved
        # warm cache from a runtime of the same episode, cached tape
        rt = EpisodeRuntime(spec, r['planner'], dyn, strict=False)
        t = time.perf_counter(); r1 = replay(spec, r['planner'], dyn, dec, private_cache=rt.private_cache); t1 = time.perf_counter()-t
        t = time.perf_counter(); r2 = replay(spec, r['planner'], dyn, dec, private_cache=rt.private_cache, lean=True); t2 = time.perf_counter()-t
        rt.close()
        keys = ('actions', 'trajectory', 'metrics', 'steps', 'shadow_pose_mismatch_steps', 'canonical_human_collision',
                'body_overlap_steps', 'min_rect_distance', 'static_contact_steps', 'final_geodesic')
        same1 = all(r0[k] == r1[k] for k in keys)
        same2 = r0['actions'] == r2['actions'] and r0['metrics'] == r2['metrics'] and \
            [row[:4] for row in r0['trajectory']] == [row[:4] for row in r2['trajectory']]
        print(json.dumps(dict(key=spec.key, config_identical=cfg_ok, warm_replay_identical=same1,
                              lean_replay_identical=same2, success=r0['metrics'].get('success'),
                              seconds_original=round(t0, 1), seconds_warm=round(t1, 1), seconds_lean=round(t2, 1))),
              flush=True)
        ok = ok and cfg_ok and same1 and same2
    print('FAST_PATHS_EXACT' if ok else 'FAST_PATHS_MISMATCH')


if __name__ == '__main__':
    main()
