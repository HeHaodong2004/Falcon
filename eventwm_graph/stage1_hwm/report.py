"""Paired candidate-vs-baseline comparisons with scene-bootstrap 95% intervals.

Rows are matched by identity (Root, anchor, stream, horizon/physical or probe/H/radius),
reduced anchor->Root->scene exactly as the legacy evaluator, and resampled by scene.
Relative change = (candidate - baseline) / baseline of the matched aggregate; negative is
better for every listed score.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np

EVAL = Path(__file__).resolve().parents[2] / 'output/stage1_hwm/eval'


def load(name):
    return np.load(EVAL / (name + '_rows.npy'), allow_pickle=True).item()


def paired(cand, base, key, ident, select=lambda r:True):
    a = {ident(r):r for r in cand if key in r and r[key] is not None and select(r)}
    b = {ident(r):r for r in base if key in r and r[key] is not None and select(r)}
    common = sorted(a.keys() & b.keys())
    roots = collections.defaultdict(list);scene = {}
    for k in common:
        roots[a[k]['root_id']].append((a[k][key], b[k][key]));scene[a[k]['root_id']] = a[k]['scene_id']
    by_scene = collections.defaultdict(list)
    for root, v in roots.items():by_scene[scene[root]].append(np.mean(v, 0))
    means = np.array([np.mean(by_scene[s], 0) for s in sorted(by_scene)])
    if not len(means):return None
    rng = np.random.default_rng(731);draw = means[rng.integers(len(means), size=(2000, len(means)))].mean(1)
    pair = means.mean(0)
    rel = (draw[:, 0] - draw[:, 1]) / draw[:, 1]
    return dict(candidate=float(pair[0]), baseline=float(pair[1]), relative=float((pair[0] - pair[1]) / pair[1]),
                lo=float(np.quantile(rel, .025)), hi=float(np.quantile(rel, .975)), matched=len(common), roots=len(roots), scenes=len(means))


PATH_ID = lambda r:(r['root_id'], r['anchor'], r['stream'], r.get('physical', -1), r.get('horizon'))
POP_ID = lambda r:(r['root_id'], r['anchor'], r['stream'], r['horizon'])
RISK_ID = lambda r:(r['root_id'], r['anchor'], r['stream'], r['probe'], r['H'], r['radius'])


def compare(cand, base):
    c, b = load(cand), load(base);out = {}
    for name, sel in (('all', lambda r:'path_es' in r), ('visible', lambda r:r.get('visible') and 'path_es' in r),
                      ('practical_gap', lambda r:r.get('practical_gap')), ('reappearance', lambda r:r.get('reappearance') and 'path_es' in r)):
        out['path_es/' + name] = paired(c['path'], b['path'], 'path_es', PATH_ID, sel)
    for h in (.5, 1., 2., 3.):
        out['path_es/h%g' % h] = paired(c['path'], b['path'], 'path_es', PATH_ID, lambda r, h=h:r.get('horizon') == h)
    out['joint_es'] = paired(c['path'], b['path'], 'joint_es', PATH_ID)
    for k in ('count_nll', 'count_rps', 'occupancy_brier', 'joint_set_kernel'):
        out[k] = paired(c['population'], b['population'], k, POP_ID)
    for probe in ('executed', 'stationary'):
        for H in (1., 2., 3.):
            for rad in (.5, 1., 2.):
                out['risk_brier/%s_H%g_r%g' % (probe, H, rad)] = paired(c['risk'], b['risk'], 'brier', RISK_ID,
                    lambda r, probe=probe, H=H, rad=rad:r['probe'] == probe and r['H'] == H and r['radius'] == rad)
        out['risk_step_brier/' + probe] = paired(c['risk'], b['risk'], 'step_brier', RISK_ID, lambda r, probe=probe:r['probe'] == probe and r['H'] == -1)
    return out


def main():
    p = argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('baseline', nargs='?', default='legacy_final')
    a = p.parse_args()
    out = compare(a.candidate, a.baseline)
    (EVAL / ('paired_%s_vs_%s.json' % (a.candidate, a.baseline))).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if v is None:continue
        print('%-34s cand %.5f base %.5f rel %+.2f%% [%+.2f%%, %+.2f%%] roots %d scenes %d' % (
            k, v['candidate'], v['baseline'], 100 * v['relative'], 100 * v['lo'], 100 * v['hi'], v['roots'], v['scenes']))


if __name__ == '__main__':
    main()
