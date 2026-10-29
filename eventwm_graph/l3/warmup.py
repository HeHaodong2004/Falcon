"""Oracle-guided warm-up data.

1. ``oracle-map``: the v14 Oracle's decision label sequences of every L3 training episode it solved (PASS), per
   configuration (``cflabel.ORACLE_DIRS``).  The actor's ``--selector oracle`` replays them in the full Layer 3 loop
   (generator -> Sandbox -> Graph Compiler -> the Oracle's candidate), which yields the compiled graphs of the Oracle's
   decision states with the real reward components -- demonstrations in exactly the policy's input space.
2. ``load_warm``: inserts those episodes into the learner's ring as ``source`` 1 with their counterfactual candidate
   labels (``cflabel.py``) joined by (episode, configuration, decision index); candidate sets are checked label by label.

Oracle information is used only as training supervision (labels, demonstration actions), never as policy input.
"""
import argparse
import json
from pathlib import Path

from .cflabel import ORACLE_RUNS, ORACLE_DIRS


def oracle_map(train_keys):
    out, jobs = {}, []
    for d, gen in sorted(ORACLE_DIRS.items()):
        for p in sorted(Path(ORACLE_RUNS, d).glob('*.json')):
            if p.name == 'summary.json':
                continue
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            if r.get('verdict') != 'PASS' or r['key'] not in train_keys:
                continue
            cfg = '%s-%s-%s' % (r['planner'], r['dynamics'], gen)
            out['%s|%s' % (r['key'], cfg)] = ['continue' if x['cont'] else x['label'] for x in r['decisions']]
            jobs.append([r['spec'], cfg])
    return out, jobs


def cf_index(cf_dir):
    idx = {}
    for p in Path(cf_dir).glob('*.json'):
        r = json.loads(p.read_text())
        cfg = '%s-%s-%s' % (r['planner'], r['dynamics'], r['config'])
        idx['%s|%s' % (r['key'], cfg)] = {s['decision']: s for s in r['states']}
    return idx


def load_warm(learner, warm_dir, cf_dir=None):
    from .actor import read_episode
    cfi = cf_index(cf_dir) if cf_dir else {}
    n = 0
    for p in sorted(Path(warm_dir).glob('*.ep')):
        data = read_episode(p)
        m = data['metrics']
        if m.get('diverged') or m['outcome'] != 'success':
            continue
        key = '%s/%s/%s/%s' % (data['spec']['dataset'], data['spec']['split'], data['spec']['scene'], data['spec']['episode_id'])
        cf = cfi.get('%s|%s' % (key, data['config']))
        n += learner.ingest_file(p, source=1, cf=cf if cf is not None else {})
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['map'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from . import suites
    keys = {s.key for s in suites.load('train_querygen')}
    m, jobs = oracle_map(keys)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out/'oracle_map.json').write_text(json.dumps(m))
    (out/'oracle_jobs.json').write_text(json.dumps(jobs))
    print(len(m), 'oracle solutions;', sum(len(v) for v in m.values()), 'decisions')


if __name__ == '__main__':
    main()
