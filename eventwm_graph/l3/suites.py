"""Layer 3 episode suites (declared 2026-10-29, before any Layer 3 training or evaluation).

Separation (scene level; salted-hash selection, no inspection of episodes or outcomes):

* ``test``  -- the official Falcon test data: Social-HM3D **val** and Social-MP3D **val** (the original paper trains on
  ``train`` and tests on ``val``; see ``dataset_split_explanation.txt``).  ``test_full`` lists every val episode
  (1,087 + 317); ``test`` is a fixed subsample for the configuration matrix (per scene the episodes with the smallest
  hash: 3 per HM3D val scene, 12 per MP3D val scene).  Frozen now; never run before the Layer 3 freeze.
* ``dev``   -- Layer 3 development / model selection: Social-HM3D **train** scenes that no querygen suite used and that
  are not Social-HM3D minival scenes (60 scenes with people, 1 episode each, human_num >= 2) plus 6 reserved Social-MP3D train
  scenes (2 episodes each).  Dev scenes never enter training.
* ``train`` -- every other Social-HM3D / Social-MP3D train scene (minival scenes excluded): up to ``per_scene``
  hash-chosen episodes with human_num >= 2 each, plus every querygen unseen-suite episode (train split) whose scene is
  not a dev scene (their v14 Oracle solutions are warm-up material).

The querygen DEV / HOLDOUT suites are val episodes: they are part of ``test_full`` and never used for Layer 3
training (the querygen generator was developed on the DEV scenes; results are also reported without those scenes).
"""
import gzip
import hashlib
import json
from pathlib import Path

SALT = 'l3-shared-policy-suites-20261029'
DATA = Path('data/datasets/pointnav')
PATH = Path('docs/l3/suites.json')
QG_SUITES = ['docs/querygen/suites_unseen.json', 'docs/querygen/suites_unseen2.json', 'docs/querygen/suites_unseen3.json',
             'docs/querygen/suites_unseen4.json', 'docs/querygen/suites_unseen5.json', 'docs/querygen/suites_unseen6.json']
QG_NAMES = ['unseen', 'unseen2', 'unseen3', 'unseen4', 'unseen5', 'unseen6']


def _h(*parts):
    return hashlib.sha256('|'.join([SALT, *map(str, parts)]).encode()).hexdigest()


def _seed(dataset, split, scene, episode_id):
    return int(_h('seed', dataset, split, scene, episode_id)[:8], 16) % (2**31-1)


def _row(dataset, split, scene, e):
    eid = str(e['episode_id'])
    return dict(dataset=dataset, split=split, scene=scene, episode_id=eid, seed=_seed(dataset, split, scene, eid),
                human_num=int(e['info']['human_num']), geodesic=round(float(e['info']['geodesic_distance']), 3))


def _scenes(dataset, split):
    root = DATA/f'social-{dataset}'/split/'content'
    return sorted(f.name.split('.')[0] for f in root.glob('*.json.gz'))


def _episodes(dataset, split, scene):
    return json.load(gzip.open(DATA/f'social-{dataset}'/split/'content'/f'{scene}.json.gz'))['episodes']


def _querygen_rows():
    rows = []
    for p, name in zip(QG_SUITES, QG_NAMES):
        for r in json.loads(Path(p).read_text())[name]:
            rows.append(dict(r, querygen_suite=name))
    return rows


def build(per_scene=6, mp3d_per_scene=20, dev_hm3d=60, dev_mp3d=6, test_hm3d=3, test_mp3d=12):
    qg = _querygen_rows()
    qg_scenes = {(r['dataset'], r['scene']) for r in qg}
    minival = {s.split('_ep')[0] for s in _scenes('hm3d', 'minival')}
    out = dict(salt=SALT, rule=__doc__.strip(), created='2026-10-29')
    # ---- test (official val splits)
    full, test = [], []
    for dataset, k in (('hm3d', test_hm3d), ('mp3d', test_mp3d)):
        for sc in _scenes(dataset, 'val'):
            eps = _episodes(dataset, 'val', sc)
            rows = [_row(dataset, 'val', sc, e) for e in eps]
            full += rows
            test += sorted(rows, key=lambda r: _h('test', dataset, sc, r['episode_id']))[:k]
    out['test_full'] = full
    out['test'] = test
    # ---- dev (train-split scenes unused by querygen and minival)
    # train-split scenes with at least one episode with people (about a quarter of the scenes have none)
    human_eps = {}
    for dataset in ('hm3d', 'mp3d'):
        for sc in _scenes(dataset, 'train'):
            if dataset == 'hm3d' and sc in minival:
                continue
            eps = [e for e in _episodes(dataset, 'train', sc) if int(e['info']['human_num']) >= 2]
            if eps:
                human_eps[(dataset, sc)] = eps
    hm3d_train = sorted(sc for d, sc in human_eps if d == 'hm3d')
    mp3d_train = sorted(sc for d, sc in human_eps if d == 'mp3d')
    free = [s for s in hm3d_train if ('hm3d', s) not in qg_scenes]
    dev_scenes = sorted(free, key=lambda s: _h('dev-scene', 'hm3d', s))[:dev_hm3d]
    dev_mp3d_scenes = sorted(mp3d_train, key=lambda s: _h('dev-scene', 'mp3d', s))[:dev_mp3d]
    dev = []
    for dataset, scenes, k in (('hm3d', dev_scenes, 1), ('mp3d', dev_mp3d_scenes, 2)):
        for sc in scenes:
            eps = sorted(human_eps[(dataset, sc)], key=lambda e: _h('dev', dataset, sc, e['episode_id']))[:k]
            dev += [_row(dataset, 'train', sc, e) for e in eps]
    out['dev'] = dev
    dev_set = {('hm3d', s) for s in dev_scenes} | {('mp3d', s) for s in dev_mp3d_scenes}
    # ---- train
    train = []
    for dataset, scenes, k in (('hm3d', hm3d_train, per_scene), ('mp3d', mp3d_train, mp3d_per_scene)):
        for sc in scenes:
            if (dataset, sc) in dev_set:
                continue
            eps = sorted(human_eps[(dataset, sc)], key=lambda e: _h('train', dataset, sc, e['episode_id']))[:k]
            train += [_row(dataset, 'train', sc, e) for e in eps]
    seen = {(r['dataset'], r['scene'], r['episode_id']) for r in train}
    qg_train = []
    for r in qg:
        if (r['dataset'], r['scene']) in dev_set or r['split'] != 'train':
            continue
        rr = {k: r[k] for k in ('dataset', 'split', 'scene', 'episode_id', 'seed', 'human_num', 'geodesic')}
        rr['querygen_suite'] = r['querygen_suite']
        qg_train.append(rr)                     # keeps the querygen seed: its Oracle solutions replay exactly
        seen.add((r['dataset'], r['scene'], r['episode_id']))
    out['train'] = train
    out['train_querygen'] = qg_train
    out['excluded'] = dict(minival_scenes=sorted(minival), dev_scenes=sorted('%s/%s' % x for x in dev_set))
    return out


def load(name, path=PATH):
    """EpisodeSpecs of a Layer 3 suite (``train``, ``train_querygen``, ``dev``, ``test``, ``test_full``)."""
    from eventwm_graph.querygen.native import EpisodeSpec
    data = json.loads(Path(path).read_text())
    rows = data['train']+data['train_querygen'] if name == 'train_all' else data[name]
    return [EpisodeSpec(r['dataset'], r['split'], r['scene'], r['episode_id'], r['seed']) for r in rows]


def rows(name, path=PATH):
    return json.loads(Path(path).read_text())[name]


if __name__ == '__main__':
    if PATH.exists():
        raise SystemExit(f'{PATH} exists: suites are frozen')
    s = build()
    PATH.write_text(json.dumps(s, indent=0))
    for k in ('test_full', 'test', 'dev', 'train', 'train_querygen'):
        r = s[k]
        hum = {n: sum(x['human_num'] == n for x in r) for n in (2, 4, 6)}
        print(k, len(r), 'scenes', len({(x['dataset'], x['scene']) for x in r}), 'humans', hum,
              'geo mean %.2f' % (sum(x['geodesic'] for x in r)/max(1, len(r))))
    print('sha256', hashlib.sha256(PATH.read_bytes()).hexdigest())
