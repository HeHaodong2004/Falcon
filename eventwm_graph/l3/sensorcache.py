"""Exact cross-episode cache of the legal sensor's static support certificates (a speed-up; outputs unchanged).

With ``pure_caches=True`` (every Layer 3 actor, labeller and replay) querygen's legal sensor computes each cell's
private support certificate at the 1 cm-quantised base height and keys it by (scene namespace incl. the quantised
height, cell, layer): the value is a pure function of the scene's navmesh geometry.  Each episode nevertheless starts
with an empty cache and recomputes it through ~10 native pathfinder calls per cell (~100 us).  Here the cache of a
scene is shared by every episode of that scene in the process and persisted to disk, so later episodes -- and the
counterfactual labellers, which re-simulate scenes the actors just visited -- reuse it.  Measured: legal sensor
67.5 -> 40.6 ms/step on a repeated episode.  Only the support certificates are persisted (the probe coordinates are
cheap concatenations of them and stay per process).

    from eventwm_graph.l3 import sensorcache
    sensorcache.attach(runtime)      # after EpisodeRuntime(...) / BranchEpisode(...).rt
    ...
    sensorcache.save(runtime)        # at episode end
"""
import hashlib
import os
import pickle
from collections import OrderedDict
from pathlib import Path

ROOT = Path(os.environ.get('L3_SUPPORT_CACHE', Path(__file__).resolve().parents[2]/'output/support_cache'))
_SCENES = OrderedDict()          # scene namespace -> shared cache dict (support entries + the probe-cache entry)
_LOADED = set()
MAX_SCENES = 6
MAX_ENTRIES = 600_000      # ~70 MB per scene file; larger caches stay in memory only


def _scene_key(obs):
    ns = obs.cache_namespace
    base = tuple(x for x in ns if not (isinstance(x, tuple) and x and x[0] == 'qheight'))
    return base


def _file(base):
    return ROOT/('%s.pkl' % hashlib.sha1(repr(base).encode()).hexdigest()[:20])


def attach(rt):
    """Point the runtime's legal-sensor observer at the shared cache of its scene (loading the disk copy once)."""
    obs = rt.sensor.observer
    base = _scene_key(obs)
    shared = _SCENES.get(base)
    if shared is None:
        shared = {}
        f = _file(base)
        if f.exists():
            try:
                shared.update(pickle.loads(f.read_bytes()))
            except Exception:
                pass
        _SCENES[base] = shared
        while len(_SCENES) > MAX_SCENES:
            _SCENES.popitem(last=False)
    _SCENES.move_to_end(base)
    # entries the runtime computed for its root observation, and its probe-cache entry
    probe_key = next((k for k, v in obs.cache.items() if v is obs.probe_cache), None)
    for k, v in obs.cache.items():
        if k != probe_key:
            shared.setdefault(k, v)
    if probe_key is not None:
        probes = shared.setdefault(probe_key, {})
        probes.update(obs.probe_cache)
        obs.probe_cache = probes
    obs.cache = shared
    rt.private_cache = shared
    return len(shared)


def save(rt):
    """Persist the support certificates of the runtime's scene (atomic; last writer wins -- entries are pure)."""
    obs = rt.sensor.observer
    base = _scene_key(obs)
    shared = _SCENES.get(base)
    if not shared:
        return 0
    data = {k: v for k, v in shared.items() if not (isinstance(k, tuple) and k and k[0] == 'static_probe_coordinates_v1')}
    if len(data) > MAX_ENTRIES:
        return 0
    ROOT.mkdir(parents=True, exist_ok=True)
    f = _file(base)
    tmp = f.with_suffix('.%d.tmp' % os.getpid())
    tmp.write_bytes(pickle.dumps(data, protocol=4))
    os.replace(tmp, f)
    return len(data)
