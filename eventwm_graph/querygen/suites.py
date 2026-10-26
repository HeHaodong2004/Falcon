"""Frozen evaluation suites (declared before any Generator/Oracle iteration).

Selection is a deterministic salted hash (no inspection of episodes or
outcomes): Social-HM3D val scenes are ordered by sha256(SALT|scene); the first
half form DEV (development + primary acceptance gate), the rest HOLDOUT.
Within each scene the episode with the smallest sha256(SALT|scene|episode_id)
is chosen.  All Social-MP3D val scenes join HOLDOUT the same way (cross-dataset).
HOLDOUT stays untouched until the Generator is frozen.
"""
import gzip
import hashlib
import json
from pathlib import Path

SALT = 'querygen-oracle-suites-20261026'
DATA = Path('data/datasets/pointnav')


def _h(*parts):
    return hashlib.sha256('|'.join([SALT, *map(str, parts)]).encode()).hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _episodes(dataset, split='val'):
    root = DATA/f'social-{dataset}'/split/'content'
    out = {}
    for f in sorted(root.glob('*.json.gz')):
        scene = f.name.split('.')[0]
        eps = json.load(gzip.open(f))['episodes']
        out[scene] = (f, eps)
    return out


def _pick(dataset, scene, eps):
    e = min(eps, key=lambda e: _h(dataset, scene, e['episode_id']))
    seed = int(_h('seed', dataset, scene, e['episode_id'])[:8], 16) % (2**31-1)
    return dict(dataset=dataset, split='val', scene=scene, episode_id=str(e['episode_id']), seed=seed,
                human_num=int(e['info']['human_num']),
                geodesic=round(float(e['info']['geodesic_distance']), 3))


def build():
    hm3d = _episodes('hm3d')
    mp3d = _episodes('mp3d')
    scenes = sorted(hm3d, key=lambda s: _h('scene', 'hm3d', s))
    half = len(scenes)//2
    dev = [_pick('hm3d', s, hm3d[s][1]) for s in scenes[:half]]
    hold = [_pick('hm3d', s, hm3d[s][1]) for s in scenes[half:]]
    hold += [_pick('mp3d', s, mp3d[s][1]) for s in sorted(mp3d, key=lambda s: _h('scene', 'mp3d', s))]
    files = {str(f): _file_sha(f) for d in (hm3d, mp3d) for f, _ in d.values()}
    return dict(salt=SALT, rule=__doc__.strip(), dev=dev, holdout=hold, dataset_file_sha256=files)


def load(name, path='docs/querygen/suites.json'):
    from .native import EpisodeSpec
    data = json.loads(Path(path).read_text())
    return [EpisodeSpec(r['dataset'], r['split'], r['scene'], r['episode_id'], r['seed']) for r in data[name]]


if __name__ == '__main__':
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'docs/querygen/suites.json')
    if out.exists():
        raise SystemExit(f'{out} exists: suites are frozen')
    s = build()
    out.write_text(json.dumps(s, indent=1))
    for k in ('dev', 'holdout'):
        rows = s[k]
        print(k, len(rows), 'humans', {n: sum(r['human_num'] == n for r in rows) for n in (2, 4, 6)},
              'geo mean %.2f min %.2f max %.2f' % (sum(r['geodesic'] for r in rows)/len(rows),
                                                   min(r['geodesic'] for r in rows), max(r['geodesic'] for r in rows)))
