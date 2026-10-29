"""On-policy counterfactual labellers: exact outcomes of EVERY candidate at the critical states of the policy's own
episodes (privileged training supervision only; never policy input).

A labeller takes finished actor episodes from a run's spool (failures first, newest first), rebuilds the episode in
querygen's exact branch simulator by replaying its recorded decisions (deterministic: same generator, same legal
sensor, recorded human tape; the replayed actions are checked against the recorded ones), and at the chosen decision
states executes every candidate for its interval followed by the default (no-selection) policy -- to the official
episode end or for at most ``cap`` steps.  Targets per episode:
  * human collision / STOP failure: the last ``k`` decisions before the end (the critical window);
  * timeout: ``k`` decisions spread over the episode;
  * success: one random decision with probability ``p_success`` (efficiency, calibration).
Output: ``<run>/cf_online/<episode>.json`` = {decision index: {labels, candidates: [outcome, steps, path, social,
alive, interval outcome]}}.  The learner writes them into the replay ring next to the episode's transitions.

    python scripts/l3_run.py eventwm_graph.l3.cfonline --run output/runs/R5 --worker 0 [--k 4 --cap 160]
"""
import argparse
import json
import os
import random
import time
from pathlib import Path


def _query(x):
    from eventwm_graph.querygen.executor import Query
    return Query(x['kind'], via=tuple(tuple(v) for v in x['via']), heading=x['heading'], duration=x['duration'],
                 speed=x['speed'], label=x['label'], meta=x['meta'] or {})


def default_continue(rt, st, goal, until):
    while not st.terminal and st.index < until:
        cont = st.execution is not None and st.execution.status == 'running'
        h = max(1, min(16, until-st.index))
        st = rt.run(st, None if cont else goal, horizon=h, cont=cont, geodesic=False)
    return st


def targets_for(m, T, k, p_success, rng):
    o = m['outcome']
    if T == 0:
        return []
    if o in ('human_collision', 'stop_fail'):
        return sorted(range(max(0, T-k), T))
    if o == 'timeout':
        step = max(1, T//k)
        return sorted(set(range(T-1, -1, -step)[:k]))
    if rng.random() < p_success:
        return [int(rng.integers(T))]
    return []


def label_episode(path, out_dir, k=4, cap=160, p_success=0.25, rng=None):
    from eventwm_graph.querygen.native import EpisodeSpec, MAX_EPISODE_STEPS
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.oracle import horizon_for
    from eventwm_graph.sandbox.episode import generator, default_query
    from .actor import read_episode
    from .env import ConfigSpec
    t0 = time.perf_counter()
    d = read_episode(path)
    m, rows = d['metrics'], d['rows']
    T = len(rows['actions'])
    targets = targets_for(m, T, k, p_success, rng)
    rec = dict(episode=Path(path).name, outcome=m['outcome'], config=d['config'], targets=targets, states={})
    if not targets or 'decision_list' not in m:
        return rec
    cfg = ConfigSpec.parse(d['config'])
    dyn = FAMILIES[cfg.dynamics]
    rt = EpisodeRuntime(EpisodeSpec(**d['spec']), cfg.planner, dyn, strict=False)
    rt.sim.check_static = False
    rt.prefix_cache = 48
    from . import sensorcache
    sensorcache.attach(rt)
    gen = generator(cfg.gen, cfg.dynamics)          # also sets the process-wide executor options
    goal = default_query()
    H = type('H', (), {'horizon': 16})()
    acts = m.get('actions')
    st = rt.root
    try:
        for i, dec in enumerate(m['decision_list']):
            if i > max(targets) or st.terminal:
                break
            if i in targets:
                queries, _ = gen.propose(st.legal, st.execution)
                labels = [q.label for q in queries]
                if labels != rows['labels'][i]:
                    rec['states'][i] = dict(error='candidate set differs from the recorded one')
                    break
                cands = []
                for q in queries:
                    cont = q.kind == 'continue'
                    h = min(horizon_for(q, H) if not cont else 16, MAX_EPISODE_STEPS-st.index)
                    child = rt.run(st, None if cont else q, horizon=h, cont=cont, geodesic=False)
                    fin = default_continue(rt, child, goal, min(MAX_EPISODE_STEPS, st.index+cap))
                    cands.append(dict(label=q.label, outcome=fin.outcome, alive=fin.outcome is None,
                                      interval=child.outcome, steps=int(fin.index-st.index),
                                      path=float(fin.path_length-st.path_length), social=int(fin.psc_steps-st.psc_steps)))
                rec['states'][i] = dict(labels=labels, candidates=cands, step=int(st.index))
            st = rt.run(st, None if dec['cont'] else _query(dec), horizon=int(dec['horizon']), cont=bool(dec['cont']),
                        geodesic=False)
            if acts is not None:
                mine = [x if isinstance(x, str) else [float(v) for v in x] for x in st.actions]
                if mine != acts[:len(mine)]:
                    rec['diverged_at'] = i
                    rec['states'] = {j: s for j, s in rec['states'].items() if j <= i}
                    break
    finally:
        try:
            sensorcache.save(rt)
        except Exception:
            pass
        rt.close()
    rec['seconds'] = round(time.perf_counter()-t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--cap', type=int, default=160)
    ap.add_argument('--p-success', type=float, default=0.25)
    ap.add_argument('--seconds', type=float, default=1e12)
    a = ap.parse_args()
    os.environ.setdefault('QG_FAST_SENSE', '1')             # the actors' exact fast legal sensor (identical rows)
    import numpy as np
    from eventwm_graph.querygen import native
    native._install()
    run = Path(a.run)
    out = run/'cf_online'; out.mkdir(exist_ok=True)
    claim = run/'cf_claims'; claim.mkdir(exist_ok=True)
    log = open(run/('cflabel_%02d.log' % a.worker), 'a')
    rng = np.random.default_rng(1234+a.worker)
    t_start = time.time()
    while time.time()-t_start < a.seconds:
        files = sorted(Path(run/'spool').glob('*.ep'), key=lambda p: p.stat().st_mtime, reverse=True)
        todo = None
        for p in files[:2000]:                               # newest first: the current policy's mistakes
            if (claim/p.name).exists():
                continue
            try:
                fd = os.open(str(claim/p.name), os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # atomic claim
                os.close(fd)
            except FileExistsError:
                continue
            todo = p
            break
        if todo is None:
            time.sleep(10)
            continue
        try:
            rec = label_episode(todo, out, a.k, a.cap, a.p_success, rng)
            if rec['states']:
                tmp = out/('.%s.tmp' % todo.name)
                tmp.write_text(json.dumps(rec))
                os.replace(tmp, out/(todo.name+'.json'))
            log.write(json.dumps(dict(t=time.time(), episode=todo.name, outcome=rec['outcome'], states=len(rec['states']),
                                      diverged=rec.get('diverged_at'), seconds=rec.get('seconds')))+'\n')
        except Exception as exc:
            import traceback
            log.write(json.dumps(dict(t=time.time(), episode=todo.name, error=repr(exc), tb=traceback.format_exc()[-1500:]))+'\n')
        log.flush()


if __name__ == '__main__':
    main()
