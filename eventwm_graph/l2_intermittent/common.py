"""Frozen research paths and offline utilities; never read production payloads."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]
OLD = BASE/'output/l2_graph_wm_120k_20261021'
SNAP = OLD/'snapshot'
OUT = BASE/'output/l2_intermittent_wm_20261022'
DOC = BASE/'docs/l2/intermittent_wm_20261022'
AGE_BINS = (('visible', 0., 1e-7), ('0-.25', 1e-7, .25),
            ('.25-.5', .25, .5), ('.5-1', .5, 1.), ('1-1.5', 1., 1.5),
            ('1.5-2', 1.5, 2.), ('over2', 2., float('inf')))
HORIZONS = (.5, 1., 2., 3.)


def read(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_for(root, base=221):
    return int.from_bytes(hashlib.sha256(('intermittent-v1:%s:%s' % (base, root)).encode()).digest()[:4], 'little')


def age_bin(age):
    if age <= 1e-7:
        return 'visible'
    for name, lo, hi in AGE_BINS[1:]:
        if age <= hi+1e-8:
            return name
    return 'over2'


def truth_at(times, positions, t):
    if t < times[0]-1e-7 or t > times[-1]+1e-7:
        return None
    j = min(int(np.searchsorted(times, t)), len(times)-1)
    if j == 0 or abs(times[j]-t) < 1e-8:
        return positions[j]
    a = (t-times[j-1])/(times[j]-times[j-1])
    return positions[j-1]*(1-a)+positions[j]*a


def centers(ob):
    if not ob.get('human_sensor_available', True) or not ob.get('observation_valid', True):
        return np.empty((0, 2))
    return np.array([np.asarray(t['position'])[[0, 2]] for t in ob.get('tracks', [])
                     if t.get('visible_now', False) and t.get('valid', True)
                     and abs(t.get('last_seen_time', ob['timestamp'])-ob['timestamp']) < 1e-7]).reshape(-1, 2)


def canonical(folder):
    queries, outcomes = read(folder/'queries.json.gz'), read(folder/'outcomes.json.gz')
    qi = max(range(len(queries)), key=lambda i: (len(queries[i]['compiled']['actions']), queries[i]['query_id']))
    nodes = read(folder/'future_observations.json.gz')
    return queries[qi], [nodes[i]['observation'] for i in outcomes[qi]['node_ids']]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))
