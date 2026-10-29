"""Diagnostic (privileged, never policy input): is candidate success at critical states predictable from the
imagined trajectories if the human future were known exactly?

At the labelled critical states of a run (cfonline labels), the episode is replayed (BranchEpisode + in-process
Sandbox with continuation imagination) and, for every candidate, the imagined trajectory (interval + continuation)
is scored against (a) the Stage 1 prediction (bank mean and the 40 worlds, as the policy sees it) and (b) the TRUE
human positions from the recorded tape.  Within-state AUC of each score for "candidate, then default, succeeds"
separates two explanations of the weak selector: a human-prediction bottleneck (true-future scores discriminate,
predicted ones do not) or trajectories that do not determine the outcome (neither discriminates).

    python scripts/l3_run.py eventwm_graph.l3.oracle_features --run output/runs/R5 --out DIR [--limit 60] [--shard i --shards n]
"""
import argparse
import json
import math
import os
import zlib
from pathlib import Path

import numpy as np


def rect_dist(pose, pts):
    from eventwm_graph.sandbox.imagine import rectangle_clearance
    return rectangle_clearance(pose, pts) if len(pts) else math.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=60)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--continuation', type=int, default=24)
    a = ap.parse_args()
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.sandbox.core import WorldModelSandbox, SandboxConfig, Candidate
    from eventwm_graph.sandbox.episode import BranchEpisode, generator
    from eventwm_graph.querygen.native import EpisodeSpec
    from eventwm_graph.querygen.dynamics import STEP_SECONDS
    from .actor import read_episode
    from .env import ConfigSpec
    from .cfonline import _query
    from . import sensorcache
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sb = WorldModelSandbox.from_release(SandboxConfig(materialize=False, continuation=a.continuation),
                                        map_device='cuda:%d' % (a.shard % 2))
    files = sorted(Path(a.run, 'cf_online').glob('*.json'))
    files = [f for f in files if zlib.crc32(f.name.encode()) % a.shards == a.shard][:a.limit]
    for f in files:
        name = f.name[:-len('.json')]
        if (out/(name+'.json')).exists():
            continue
        lab = json.loads(f.read_text())
        sts = {int(k): v for k, v in lab['states'].items() if 'candidates' in v}
        mixed = {i: s for i, s in sts.items() if any(c['outcome'] == 'success' for c in s['candidates'])
                 and any(c['outcome'] != 'success' for c in s['candidates'])}
        if not mixed:
            continue
        d = read_episode(Path(a.run, 'spool', name))
        m = d['metrics']
        cfg = ConfigSpec.parse(d['config'])
        gen = generator(cfg.gen, cfg.dynamics)
        ep = BranchEpisode(EpisodeSpec(**d['spec']), cfg.planner, cfg.dynamics, range_sensor='fast-driver', gc_freeze=True)
        sensorcache.attach(ep.rt)
        tape = ep.rt.tape.positions                        # [steps, N, 3] true human positions (privileged)
        rows = []
        try:
            for i, dec in enumerate(m['decision_list']):
                if i > max(mixed) or ep.over:
                    break
                if i in mixed:
                    queries, diag = gen.propose(ep.legal, ep.execution)
                    if [q.label for q in queries] != mixed[i]['labels']:
                        break
                    cands = [Candidate(None, cfg.planner, cfg.dynamics, execution=ep.execution, level=ep.level)
                             if q.kind == 'continue' else Candidate(q, cfg.planner, cfg.dynamics, level=ep.level)
                             for q in queries]
                    res = sb.evaluate(ep.decision_state(), ep.legal, cands)
                    k0 = ep.steps
                    bank = res.humans
                    W = np.asarray(bank.weights, np.float64); W = W/W.sum()
                    for k, (c, lbl) in enumerate(zip(res.candidates, mixed[i]['candidates'])):
                        ro = c.rollout
                        poses = np.asarray(ro.poses)
                        cont = getattr(ro, 'cont', None)
                        seq = np.concatenate([poses, cont]) if cont is not None and len(cont) else poses
                        # true future: tape positions at the same absolute steps (privileged)
                        true_min = math.inf
                        for j, p in enumerate(seq):
                            s = min(k0+j, len(tape)-1)
                            true_min = min(true_min, rect_dist(p, tape[s][:, [0, 2]]))
                        # predicted: Stage 1 worlds (min over time per world; weighted P(< 0.3 m) and mean of mins)
                        rel = np.arange(1, min(len(seq)-1, 62)+1)*STEP_SECONDS
                        HP = np.asarray(bank.sandbox_positions(rel), np.float64) if len(bank.track_keys) else None
                        pred_min, p_close = 10., 0.
                        if HP is not None and HP.shape[2]:
                            mins = np.full(len(W), np.inf)
                            for j in range(len(rel)):
                                pts = HP[:, j]                                   # [W, N, 2]
                                for w in range(len(W)):
                                    q = pts[w][np.isfinite(pts[w]).all(-1)]
                                    if len(q):
                                        mins[w] = min(mins[w], rect_dist(seq[j+1], q))
                            fin = np.where(np.isfinite(mins), mins, 10.)
                            pred_min = float((W*fin).sum()); p_close = float((W*(fin < 0.3)).sum())
                        rows.append(dict(decision=i, k=k, label=lbl['label'], success=lbl['outcome'] == 'success',
                                         true_min=float(true_min), pred_min=pred_min, p_close=p_close,
                                         x_pcol=float(c.humans['p_collision']) if c.humans else 0.))
                ep.run(None if dec['cont'] else _query(dec), horizon=int(dec['horizon']))
        finally:
            sensorcache.save(ep.rt)
            ep.close()
        (out/(name+'.json')).write_text(json.dumps(rows))
        print(json.dumps(dict(episode=name, rows=len(rows))), flush=True)
    sb.close()


def summarize(out):
    rows = [r for f in Path(out).glob('*.json') for r in json.loads(f.read_text())]
    by = {}
    for r in rows:
        by.setdefault((r.get('episode'), r['decision'], id), []).append(r)
    return rows


if __name__ == '__main__':
    main()
