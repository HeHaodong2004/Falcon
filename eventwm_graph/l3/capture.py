"""Development tool: capture real decision contexts (legal state, candidates, full SandboxResult) to disk.

The Graph Compiler is developed and unit-tested offline on these captures, so iterating on it never re-runs the
environment or Layer 2.

    python scripts/l3_run.py eventwm_graph.l3.capture --out output/captures/c1 --episodes unseen:31,unseen2:47 \
        --decisions 6 --planner orca --dynamics native --config k10
"""
import argparse
import os
import time
from multiprocessing.reduction import ForkingPickler
from pathlib import Path


def capture_episode(sb, spec, planner, dynamics, config, decisions, out, range_sensor='fast-driver'):
    from eventwm_graph.sandbox.core import Candidate
    from eventwm_graph.sandbox.episode import BranchEpisode, generator, default_query
    gen = generator(config, dynamics)
    ep = BranchEpisode(spec, planner, dynamics, range_sensor=range_sensor, gc_freeze=True)
    q = default_query()
    n = 0
    for d in range(int(decisions)+1):
        if ep.over:
            break
        if d > 0:
            r = ep.run(q if ep.execution is None or ep.execution.status != 'running' else None, 16)
            if ep.over or r['steps'] == 0:
                break
        state = ep.decision_state()
        queries, diag = gen.propose(ep.legal, ep.execution)
        cands = [Candidate(None, planner, dynamics, execution=ep.execution, level=ep.level, query_id='continue')
                 if qq.kind == 'continue' else Candidate(qq, planner, dynamics, level=ep.level, query_id=qq.label)
                 for qq in queries]
        t = time.perf_counter()
        res = sb.evaluate(state, ep.legal, cands)
        rec = dict(key=spec.key, spec=spec.asdict(), planner=planner, dynamics=dynamics, config=config, decision=d,
                   steps=ep.steps, legal=ep.legal.copy(), queries=queries, diag={k: v for k, v in diag.items()
                                                                                 if isinstance(v, (int, float, bool, str))},
                   result=res, sandbox_s=time.perf_counter()-t, geodesic=float(ep.state.geodesic),
                   history=[dict(query=x['query'].describe() if hasattr(x['query'], 'describe') else None,
                                 cont=x['cont'], steps=x['steps'], status=x['status']) for x in ep.decisions])
        name = '%s_%s_%s_d%02d.pkl' % (spec.key.replace('/', '_'), planner, dynamics, d)
        (out/name).write_bytes(bytes(ForkingPickler.dumps(rec)))
        n += 1
    ep.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--decisions', type=int, default=6)
    ap.add_argument('--planner', default='orca')
    ap.add_argument('--dynamics', default='native')
    ap.add_argument('--config', default='k10')
    a = ap.parse_args()
    from eventwm_graph.sandbox import service  # noqa: F401  (registers the transport reducers used for pickling)
    from eventwm_graph.sandbox.core import WorldModelSandbox, SandboxConfig
    from eventwm_graph.querygen import suites
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sb = WorldModelSandbox.from_release(SandboxConfig(), map_device='cuda:0')
    for item in a.episodes.split(','):
        suite, idx = item.split(':')
        spec = suites.load(suite)[int(idx)]
        n = capture_episode(sb, spec, a.planner, a.dynamics, a.config, a.decisions, out)
        print('captured', spec.key, n, flush=True)
    sb.close()


if __name__ == '__main__':
    main()
