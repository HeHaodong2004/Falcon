"""Privileged collision context (ANALYSIS ONLY, never policy input): what was the robot doing when people walked into it,
and had it ever seen them?

Replays collision episodes of an evaluation spool decision by decision (same generator options, legal sensor, recorded
human tape; no Layer 2), then from the robot's legal pose history and the TRUE human positions of the tape reports per
collision: the robot's motion in the last 1 s (moving / rotating in place / standing), the colliding person's bearing
relative to the robot's heading at contact and 1 s / 2 s before, the person's speed and approach direction, and
whether a legal track was visible within 0.6 m of that person in the last 1 s / 3 s / ever.

    python scripts/l3_run.py eventwm_graph.l3.collctx --dir output/eval/floor_dev24/spool --out output/collctx.json \
        [--configs orca-native-k10,dwa-native-k10,sfm-native-k10] [--limit 60]
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def _bearing(pose, pt):
    dx, dz = pt[0]-pose[0], pt[1]-pose[2]
    fw = dx*math.cos(pose[3])-dz*math.sin(pose[3])
    lf = -dx*math.sin(pose[3])-dz*math.cos(pose[3])
    return math.degrees(math.atan2(lf, fw)), math.hypot(dx, dz)


def analyse(path):
    from eventwm_graph.sandbox.episode import BranchEpisode, generator
    from eventwm_graph.querygen.native import EpisodeSpec
    from eventwm_graph.querygen.dynamics import STEP_SECONDS
    from .actor import read_episode
    from .env import ConfigSpec, _NullRange, _NullStream
    from .cfonline import _query
    d = read_episode(path)
    m = d['metrics']
    cfg = ConfigSpec.parse(d['config'])
    generator(cfg.gen, cfg.dynamics)                     # process-wide executor options before any replay
    ep = BranchEpisode(EpisodeSpec(**d['spec']), cfg.planner, cfg.dynamics, range_sensor=_NullRange(), gc_freeze=True,
                       stream=_NullStream())
    try:
        for dec in m['decision_list']:
            if ep.over:
                break
            ep.run(None if dec['cont'] else _query(dec), horizon=int(dec['horizon']))
        legal = ep.legal
        poses = np.asarray(legal.poses, np.float64)
        times = np.asarray(legal.times, np.float64)
        tape = np.asarray(ep.rt.tape.positions, np.float64)          # [steps, N, 3] true positions (privileged)
        outcome = ep.state.outcome
        T = len(poses)-1
        if outcome != 'human_collision' or T < 1:
            return dict(file=Path(path).name, outcome=outcome, replay_steps=T, recorded=m['outcome'])
        hp = tape[min(T, len(tape)-1)][:, [0, 2]]
        dist = np.hypot(hp[:, 0]-poses[-1, 0], hp[:, 1]-poses[-1, 2])
        h = int(np.argmin(dist))
        n1, n2 = int(round(1.0/STEP_SECONDS)), int(round(2.0/STEP_SECONDS))
        a1 = max(T-n1, 0)
        trans = float(np.hypot(*(poses[-1, [0, 2]]-poses[a1, [0, 2]])))
        rot = float(np.abs(np.angle(np.exp(1j*np.diff(poses[a1:, 3])))).sum())
        motion = 'moving' if trans > 0.5 else ('rotating' if rot > 0.3 else 'standing')
        out = dict(file=Path(path).name, config=d['config'], outcome=outcome, steps=T, motion=motion,
                   robot_trans_1s=round(trans, 2), robot_rot_1s=round(math.degrees(rot), 1))
        for name, k in (('contact', T), ('t-1s', a1), ('t-2s', max(T-n2, 0))):
            hpos = tape[min(k, len(tape)-1)][h, [0, 2]]
            b, r = _bearing(poses[k], hpos)
            out['bearing_'+name] = round(b, 1)
            out['dist_'+name] = round(r, 2)
        hv = tape[min(T, len(tape)-1)][h, [0, 2]]-tape[min(a1, len(tape)-1)][h, [0, 2]]
        out['human_speed'] = round(float(np.hypot(*hv))/max((T-a1)*STEP_SECONDS, 1e-6), 2)
        # was a legal track visible near that person (0.6 m) in the last 1 s / 3 s / ever?
        seen = {'1s': False, '3s': False, 'ever': False}
        for (tm, trs), k in zip(legal.track_log, range(1, len(legal.track_log)+1)):
            hpos = tape[min(k, len(tape)-1)][h, [0, 2]]
            vis = any(t.get('visible_now') and math.hypot(t['position'][0]-hpos[0], t['position'][2]-hpos[1]) < 0.6
                      for t in trs)
            if vis:
                seen['ever'] = True
                age = times[-1]-tm
                if age <= 3.0:
                    seen['3s'] = True
                if age <= 1.0:
                    seen['1s'] = True
        out.update({'seen_'+k: v for k, v in seen.items()})
        return out
    finally:
        ep.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--configs', default='orca-native-k10,dwa-native-k10,sfm-native-k10')
    ap.add_argument('--limit', type=int, default=60)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from .actor import read_episode
    keep = set(a.configs.split(','))
    files = []
    for p in sorted(Path(a.dir).glob('*.ep')):
        if p.name.split('__')[1].replace('.ep', '') not in keep:
            continue
        files.append(p)
    rows = []
    out = Path(a.out)
    done = {}
    if out.exists():
        done = {r['file']: r for r in json.loads(out.read_text())}
    n = 0
    for i, p in enumerate(files):
        if i % a.shards != a.shard:
            continue
        if p.name in done:
            rows.append(done[p.name]); n += 1
            continue
        d = read_episode(p)
        if d['metrics']['outcome'] != 'human_collision':
            continue
        try:
            r = analyse(p)
        except Exception as exc:
            r = dict(file=p.name, error=repr(exc))
        rows.append(r); n += 1
        out.write_text(json.dumps(rows))
        print(json.dumps(r), flush=True)
        if n >= a.limit:
            break
    ok = [r for r in rows if r.get('outcome') == 'human_collision' and 'motion' in r]
    if ok:
        from collections import Counter
        mot = Counter(r['motion'] for r in ok)
        sect = Counter('front' if abs(r['bearing_contact']) <= 45 else ('behind' if abs(r['bearing_contact']) >= 135 else 'side')
                       for r in ok)
        print(json.dumps(dict(collisions=len(ok), motion=mot, bearing=sect,
                              seen_ever=float(np.mean([r['seen_ever'] for r in ok])),
                              seen_3s=float(np.mean([r['seen_3s'] for r in ok])),
                              seen_1s=float(np.mean([r['seen_1s'] for r in ok])),
                              unseen_by_motion={k: float(np.mean([not r['seen_3s'] for r in ok if r['motion'] == k]))
                                                for k in mot})))


if __name__ == '__main__':
    main()
