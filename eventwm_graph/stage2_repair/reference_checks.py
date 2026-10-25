"""Independent dictionary/set metric checks and descriptive human cohorts."""
from collections import defaultdict
import gzip
import json
import numpy as np
from .diagnose import OUT, read, save
from scripts.stage2_diagnostic.common import extended_rates


def set_metrics(keys, truth, prediction):
    actual = {(tuple(k), int(s)) for k, s in zip(keys, truth) if s}
    predicted = {(tuple(k), int(s)) for k, s in zip(keys, prediction) if s}
    aset = {k for k, s in actual}
    pset = {k for k, s in predicted}
    return dict(label_aware_delta_iou=len(actual & predicted)/max(len(actual | predicted), 1),
                revelation_iou=len(aset & pset)/max(len(aset | pset), 1),
                missed_revelations=len(aset-pset), false_revelations=len(pset-aset),
                incorrect_free_blocked=sum((k, s) not in predicted for k, s in actual if k in pset))


def metrics():
    done = []
    for i, e in enumerate(read(OUT/'trace.json')['examples'], 1):
        with np.load(OUT/'examples'/('%02d.npz'%i), allow_pickle=False) as z:
            keys, y, hard = z['keys'], z['y'], z['prediction']
        cm = np.zeros((3, 3), np.int64)
        for actual, predicted in zip(y, hard):
            cm[int(actual), int(predicted)] += 1
        independent = set_metrics(keys, y, hard)
        existing = extended_rates(cm)
        for k, value in independent.items():
            assert abs(value-existing[k]) < 1.e-12, (i, k)
        done.append(dict(example=i, metrics=independent, confusion=cm.tolist()))
    save(OUT/'independent_metrics.json', dict(method='Independent sets of (exact cell, certificate label) and cell-only sets; no production replay/label/rates helper used for reference.', windows=done))


def cohorts(name='selected'):
    source = OUT/'full_evaluation'/name
    buckets = defaultdict(lambda: [[], []])
    roots = [set(), set()]
    scenes = [set(), set()]
    for path in source.glob('*.json.gz'):
        r = json.loads(gzip.decompress(path.read_bytes()))
        cohort = int(r['human_count'] != 0)
        roots[cohort].add(r['root_id'])
        scenes[cohort].add(r['scene_id'])
        for e in r['records']:
            motion = next(s for s in e['groups'] if s.startswith('motion:'))
            duration = next(s for s in e['groups'] if s.startswith('duration:'))
            target = next(i for i, limit in enumerate([0, 19, 99, 499, 10**20]) if e['new_cells'] <= limit)
            buckets[r['scene_id'], motion, duration, target][cohort].append(np.array([e['baseline_cm'], e['cm']], np.float64))
    shared = []
    sums = np.zeros((2, 2, 3, 3), np.float64)
    mass = 0
    for key, values in buckets.items():
        n0, n1 = map(len, values)
        if not n0 or not n1:
            continue
        weight = min(n0, n1)
        sums += weight*np.array([np.mean(v, axis=0) for v in values])
        mass += weight
        shared.append(dict(scene=key[0], motion=key[1], duration=key[2], target_size_bin=key[3], zero_windows=n0, present_windows=n1))
    save(OUT/('human_cohorts_'+name+'.json'), dict(role='Descriptive standardized comparison, not causal attribution. Human presence is recorded episode human_num, never inferred from missing tracks. Same checkpoint/scoring procedure within each model.',
        root_counts=list(map(len, roots)), scene_counts=list(map(len, scenes)), common_strata=len(shared), weight='min(cohort window counts) per common scene/motion/duration/target-size stratum; mean confusion within stratum before standardization.',
        common_support_windows=[sum(s[k] for s in shared) for k in ['zero_windows', 'present_windows']],
        matched={cohort: {model: extended_rates(sums[c, m]/mass) for m, model in enumerate(['baseline', 'candidate'])} for c, cohort in enumerate(['human_zero', 'human_present'])}, strata=shared))


if __name__ == '__main__':
    metrics()
    cohorts()
