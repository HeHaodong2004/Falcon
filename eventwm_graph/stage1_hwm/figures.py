"""Qualitative side-by-side banks (legacy vs candidate) on development anchors.

Each panel: legal map raster (world frame, robot-centred), true future paths (white), 40
sampled worlds (coloured thin lines; tracked keys orange, residual people cyan), robot.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .data import Store
from .metrics import iter_banks, legacy_records


def bank_index(path, stream=41):
    out = {}
    for b in iter_banks(path):
        if int(b['stream']) == stream:out[(b['root_id'], int(b['anchor']))] = b
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('candidate');p.add_argument('--baseline', default='legacy_final');p.add_argument('--n', type=int, default=6)
    p.add_argument('--seed', type=int, default=3);p.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'docs/stage1_hwm/figures'))
    a = p.parse_args()
    root = Path('/var/tmp/falcon-stage1-claude/banks')
    cand = bank_index(root / a.candidate);base = bank_index(root / a.baseline)
    store = Store('dev');recs = legacy_records()
    anchor_of = {(store.root_ids[store.a['anchor_root'][i]], int(store.a['j'][i])): i for i in range(store.n)}
    rng = np.random.default_rng(a.seed)
    keys = [k for k in cand if k in base and (recs[k]['physical'] >= 0).any() and recs[k]['times'][5] > 0]
    pick = [keys[i] for i in rng.choice(len(keys), a.n, replace=False)]
    fig, ax = plt.subplots(2, a.n, figsize=(4.2 * a.n, 8.6))
    for c, key in enumerate(pick):
        i = anchor_of[key];ri = store.a['anchor_root'][i];T, truth = store.truth[ri]
        t0 = store.a['t'][i];sel = (T >= t0 - 1e-9) & (T <= t0 + 3.03)
        corner = store.a['corner'][i];mp = store.a['map'][i]
        rgb = np.zeros(mp.shape + (3,));rgb[mp == 1] = (.55, .85, .55);rgb[mp == 2] = (.85, .3, .3);rgb[mp == 0] = (.12, .12, .2)
        ext = [corner[0], corner[0] + .2 * mp.shape[0], corner[1], corner[1] + .2 * mp.shape[1]]
        robot = store.a['robot'][i][:2]
        for row, (name, bank) in enumerate(((a.baseline, base[key]), (a.candidate, cand[key]))):
            axx = ax[row, c];axx.imshow(rgb.transpose(1, 0, 2), origin='lower', extent=ext)
            pos = bank['positions'];pres = bank['present'];n_keys = bank['key_cols'].shape[1]
            tracked_cols = set(np.unique(bank['key_cols']).tolist())
            for w in range(pos.shape[0]):
                for n in np.flatnonzero(pres[w]):
                    col = 'orange' if n in tracked_cols else 'cyan'
                    axx.plot(pos[w, :, n, 0], pos[w, :, n, 1], '-', color=col, alpha=.18, lw=.8)
            for h in range(truth.shape[1]):axx.plot(truth[sel, h, 0], truth[sel, h, 1], 'w-', lw=2)
            axx.plot(*robot, 'm^', ms=9)
            axx.set_xlim(robot[0] - 6, robot[0] + 6);axx.set_ylim(robot[1] - 6, robot[1] + 6)
            axx.set_title('%s  %s j=%d' % (name, key[0][:6], key[1]), fontsize=9)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dest = Path(a.out) / ('qualitative_%s_vs_%s_seed%d.png' % (a.candidate, a.baseline, a.seed))
    plt.savefig(dest, dpi=70, bbox_inches='tight');print(dest)


if __name__ == '__main__':
    main()
