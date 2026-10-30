"""State bank for the closed-loop Sandbox (analysis): the LEGAL inputs of every probed decision, saved with the exact
outcome of every candidate, so that Sandbox / Layer 2 variants can be scored offline against ground truth.

For each episode with a short-horizon probe record (``shortprobe.py``: the exact 16 / 32 / 64-step outcome of every
candidate at sub-sampled floor decisions), the floor episode is replayed in the Layer 3 environment with a LOCAL
Sandbox (same generator, Stage 1 / Stage 2, legal stream; the floor's choices) and at the probed decisions the bank
stores:

* legal (decision-time): the LegalState (map, tracks, track log, pose history, bumps, goal), the running execution,
  the speed level, the queries and the default index, and the Layer 2 DecisionState (frames, partial graph, range
  packet) -- everything a Sandbox variant may consume;
* the Stage 1 bank of the decision (40 worlds, positions over its horizon, weights, track keys, presence);
* privileged, for EVALUATION only: every person's true position over the next 64 steps (the recorded tape) and the
  probe's exact outcomes per candidate.

    L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.statebank --probe output/shortprobe/orca_dev \
        --out output/statebank/orca_dev [--shard i --shards n --device cuda:0]

Each state is one pickle ``<out>/<episode>__d<decision>.pkl``.
"""
import argparse
import json
import os
import pickle
import time
import zlib
from pathlib import Path

import numpy as np

W_MAX = 64


def bank_arrays(bank, steps=W_MAX):
    rel = np.arange(1, steps+1)*0.048
    inside = rel[rel <= float(bank.times[-1])+1e-9]
    keys = list(getattr(bank, 'track_keys', ()))
    return dict(track_keys=[str(k) for k in keys], weights=np.asarray(bank.weights, np.float64),
                present=np.asarray(getattr(bank, 'present', np.ones((len(bank.weights), len(keys)))), bool),
                unseen=np.asarray(getattr(bank, 'unseen', np.zeros(len(keys))), bool),
                times=np.asarray(bank.times, np.float64),
                positions=np.asarray(bank.sandbox_positions(inside), np.float32) if keys else np.zeros((len(bank.weights), len(inside), 0, 2), np.float32))


def capture_episode(sb, spec, cfg, records, out, compiler):
    from .env import L3Episode
    # sb None (--blind): no Layer 2 -- the legal inputs, the executor and the candidates only (state / bank None); the
    # floor's choices and the generator do not depend on the Sandbox, so the probed decisions are the same
    ep = L3Episode(spec, cfg, sb, compiler=compiler, imagination=sb is not None)
    want = {int(r['decision']): r for r in records}
    saved = 0
    try:
        tape = np.asarray(ep.rt.tape.positions, np.float64)
        while not ep.over and ep.decisions <= max(want):
            d = ep.observe()
            k = ep.decisions
            if k in want:
                r = want[k]
                if list(d.labels) != list(r['labels']) or int(ep.ep.steps) != int(r['step']):
                    print(json.dumps(dict(key=spec.key, decision=k, error='state differs from the probe record')), flush=True)
                    break
                i0 = int(ep.ep.steps)
                idx = np.minimum(np.arange(i0, i0+W_MAX+1), len(tape)-1)
                ex = ep.ep.execution
                item = dict(key=spec.key, config=cfg.name, decision=k, step=i0,
                            legal=ep.ep.legal.copy() if hasattr(ep.ep.legal, 'copy') else ep.ep.legal,
                            execution=ex.copy() if ex is not None else None, level=int(ep.ep.level),
                            queries=list(d.queries), labels=list(d.labels), default=int(d.default),
                            state=getattr(ep, 'last_state', None),
                            bank=bank_arrays(d.result.humans) if d.result is not None else None,
                            truth_people=tape[idx][:, :, [0, 2]].astype(np.float32),          # privileged (evaluation)
                            probe=r)
                name = '%s__d%02d.pkl' % (spec.key.replace('/', '_'), k)
                with open(Path(out, name+'.tmp'), 'wb') as f:
                    pickle.dump(item, f, protocol=4)
                os.replace(Path(out, name+'.tmp'), Path(out, name))
                saved += 1
            ep.act(d.default)
    finally:
        ep.close()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default=None)
    ap.add_argument('--blind', action='store_true', help='CPU only: no Layer 2 (items without state / bank)')
    a = ap.parse_args()
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.sandbox.core import WorldModelSandbox, SandboxConfig
    from eventwm_graph.querygen.native import EpisodeSpec
    from .env import ConfigSpec
    from .compiler import GraphCompiler
    from . import suites
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    specs = {s.key: s for name in ('dev', 'train') for s in suites.load(name)}
    files = sorted(Path(a.probe).glob('*.json'))
    files = [f for f in files if zlib.crc32(f.name.encode()) % a.shards == a.shard]
    sb = None if a.blind else WorldModelSandbox.from_release(SandboxConfig(materialize=False, route=True),
                                                              map_device=a.device or 'cuda:%d' % (a.shard % 2))
    comp = GraphCompiler(brake_tail=True)
    for f in files:
        rec = json.loads(f.read_text())
        if not rec.get('probe') or rec['key'] not in specs:
            continue
        name0 = rec['key'].replace('/', '_')
        if all(Path(out, '%s__d%02d.pkl' % (name0, int(r['decision']))).exists() for r in rec['probe']):
            continue
        t0 = time.perf_counter()
        try:
            n = capture_episode(sb, specs[rec['key']], ConfigSpec.parse(rec['config']), rec['probe'], out, comp)
        except Exception as exc:
            import traceback
            print(json.dumps(dict(key=rec['key'], error=repr(exc), tb=traceback.format_exc()[-800:])), flush=True)
            continue
        print(json.dumps(dict(key=rec['key'], saved=n, of=len(rec['probe']), seconds=round(time.perf_counter()-t0, 1))),
              flush=True)
    if sb is not None:
        sb.close()


if __name__ == '__main__':
    main()
