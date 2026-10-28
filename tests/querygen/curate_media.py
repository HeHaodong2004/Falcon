"""Pick and render navigation recordings for the results page (top-down, CPU).

  python scripts/querygen_run.py tests.querygen.curate_media --oracle-root R --floor-arm A --out DIR [--cfg c1k10]
         [--n-social 2] [--n-static 1] [--extra RUN_JSON|LABEL|SUITE|CAPTION ...]

Selection (deterministic, from finished runs): episodes the Oracle solved (PASS) where the default policy of the
same foundation failed -- by a human collision (social decisions; prefers a moderate minimum robot-human distance
on the Oracle branch: a visible interaction that stays clear) and by a timeout (static deadlock) -- rendered side by
side (default policy | Oracle-selected queries) with the same recorded human motion.  Writes DIR/media.json
([{file, caption}]) for build_page.py.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

SUITE_OF = {'dev': 'dev', 'unseen': 'unseen', 'unseen2': 'unseen2', 'unseen3': 'unseen3'}


def load_runs(pattern):
    out = {}
    for f in glob.glob(pattern):
        if f.endswith(('run.json', 'summary.json')):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if 'verdict' in r:
            out[r['key']] = (f, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle-root', required=True)
    ap.add_argument('--floor-arm', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cfg', default='c1k10')
    ap.add_argument('--n-social', type=int, default=2)
    ap.add_argument('--n-static', type=int, default=1)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--extra', nargs='*', default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fl = 'c1' if a.cfg.startswith('c1') else 'c2'
    planner = 'ORCA' if fl == 'c1' else 'DWA'
    picks = []
    for suite in SUITE_OF:
        O = load_runs(f'{a.oracle_root}/{suite}_{a.cfg}/*.json')
        D = load_runs(f'{a.floor_arm}/{suite}_{fl}/*.json')
        for k, (fo, ro) in O.items():
            if ro['verdict'] != 'PASS' or k not in D:
                continue
            fd, rd = D[k]
            outcome = rd['branch']['outcome']
            if outcome == 'success':
                continue
            rect = [row[4] for row in ro['replay']['trace']['trajectory'] if row[4] is not None]
            dmin = min(rect) if rect else 9.
            picks.append(dict(suite=suite, key=k, oracle=fo, default=fd, default_outcome=outcome,
                              default_steps=rd['branch']['steps'], oracle_steps=ro['replay']['steps'],
                              spl=ro['replay']['metrics'].get('spl', 0.), dmin=dmin))
    social = sorted([p for p in picks if p['default_outcome'] == 'human_collision' and 0.3 <= p['dmin'] <= 1.0],
                    key=lambda p: (abs(p['dmin']-0.5), p['oracle_steps']))[:a.n_social]
    static = sorted([p for p in picks if p['default_outcome'] == 'timeout'],
                    key=lambda p: (p['oracle_steps'], p['key']))[:a.n_static]
    media = []
    for p in social+static:
        scene = p['key'].split('/')[2]
        why = ('walks into a human' if p['default_outcome'] == 'human_collision' else 'deadlocks (timeout)')
        mp4 = os.path.join(a.out, f"{scene}_{a.cfg}_default_vs_oracle.mp4")
        title = (f"{scene} ({p['suite']}): the default policy {why} after {p['default_steps']} steps; "
                 f"the Oracle-selected queries reach the goal in {p['oracle_steps']} steps (SPL {p['spl']:.2f})")
        cmd = [sys.executable, 'scripts/querygen_run.py', 'tests.querygen.render_topdown',
               f"{p['default']}:Default policy ({planner}, no selection)",
               f"{p['oracle']}:Oracle-selected queries ({planner}, K={a.cfg[-2:].lstrip('k')})",
               '--suite', p['suite'], '--out', mp4, '--every', str(a.every), '--title', title]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        media.append(dict(file=mp4, caption=title, key=p['key'], kind=p['default_outcome']))
        print(json.dumps(media[-1]), flush=True)
    for item in a.extra:
        run, label, suite, caption = item.split('|', 3)
        scene = json.load(open(run))['key'].split('/')[2]
        mp4 = os.path.join(a.out, f"{scene}_extra_{len(media)}.mp4")
        cmd = [sys.executable, 'scripts/querygen_run.py', 'tests.querygen.render_topdown', f'{run}:{label}',
               '--suite', suite, '--out', mp4, '--every', str(a.every), '--title', caption]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        media.append(dict(file=mp4, caption=caption, key=scene, kind='extra'))
        print(json.dumps(media[-1]), flush=True)
    json.dump(dict(picks=picks, media=media), open(os.path.join(a.out, 'media.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
