"""Oracle evaluation driver: search, authoritative replay, strict verdict.

Single episode:
  python scripts/querygen_run.py eventwm_graph.querygen.evaluate one KEY SEED --out DIR [opts]
Suite (one subprocess per episode):
  python scripts/querygen_run.py eventwm_graph.querygen.evaluate suite dev --out DIR --workers 8 [opts]

Verdict PASS requires: Oracle success, replay success == 1 with STOP inside
the 500-step limit, official human_collision == 0, zero body-proxy overlap
steps, zero static-contact steps, and replayed actions identical to the
Oracle branch actions.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .native import EpisodeSpec


def git_identity():
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--', 'eventwm_graph/querygen'], text=True).strip())
        return dict(commit=head, querygen_dirty=dirty)
    except Exception as exc:  # pragma: no cover
        return dict(error=str(exc))


def snapshot_identity():
    """Content hash of the querygen package actually imported by this process."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return dict(path=str(root), sha256=h.hexdigest())


def _decisions_json(decisions):
    out = []
    for d in decisions:
        q = d['query']
        out.append(dict(q.describe(), cont=d['cont'], start=d['start'], end=d['end'], horizon=d['horizon'],
                        status=d['status'], reason=d['reason'], outcome=d['outcome']))
    return out


def compact_trace(rep):
    """Per-step replay trace for behaviour metrics: actions + (x, z, yaw, geodesic, min rect)."""
    acts = ''.join(a[0] if isinstance(a, str) else 'v' for a in rep['actions'])
    vel = [list(a) for a in rep['actions'] if not isinstance(a, str)]
    traj = [[round(v, 4) if v != float('inf') else None for v in row] for row in rep['trajectory']]
    return dict(actions=acts, velocity_commands=vel, trajectory=traj, per_decision_steps=rep['per_decision_steps'])


def run_one(spec, out_dir, planner='orca', dynamics='native', oracle_kw=None, gen_kw=None, plot=True):
    from .episode import EpisodeRuntime, replay
    from .generator import QueryGenerator, GeneratorConfig
    from .oracle import Oracle, OracleConfig
    from .dynamics import FAMILIES
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    key = spec.key.replace('/', '_')
    dyn = FAMILIES[dynamics]
    t0 = time.perf_counter()
    rt = EpisodeRuntime(spec, planner, dyn, strict=True, trace=False)
    gen = QueryGenerator(dyn, GeneratorConfig(**(gen_kw or {})))
    oracle = Oracle(rt, gen, OracleConfig(**(oracle_kw or {})))
    res = oracle.search()
    st = res['state']
    record = dict(spec=spec.asdict(), key=spec.key, planner=planner, dynamics=dynamics,
                  generator=asdict(gen.cfg), oracle_config=asdict(oracle.cfg), code=git_identity(),
                  code_snapshot=snapshot_identity(),
                  start_geodesic=rt.start_geodesic, setup_s=rt.setup_s, spawn_contact=rt.spawn_contact,
                  oracle=dict(success=res['success'], steps=res['steps'], outcome=res['outcome'],
                              final_geodesic=res['final_geodesic'], seconds=res['seconds'],
                              stats=res['stats'], runtime_perf=rt.perf, sim_perf=rt.sim.performance,
                              min_rect=st.min_rect_dist, psc_steps=st.psc_steps,
                              path_length=st.path_length, static_contacts=st.static_contacts,
                              escape_contacts=st.escape_contacts),
                  decisions=_decisions_json(st.decisions))
    if plot:
        try:
            from .viz import plot_branch
            plot_branch(rt, st, out_dir/f'{key}.png',
                        title=f"{spec.key} {planner}/{dynamics} oracle={'OK' if res['success'] else st.outcome}")
        except Exception as exc:  # pragma: no cover
            record['plot_error'] = str(exc)
    rt.close()
    verdict = 'FAIL_SEARCH'
    if res['success']:
        t = time.perf_counter()
        try:
            rep = replay(spec, planner, dyn, st.decisions)
        except RuntimeError as exc:
            # the authoritative replay could not follow the Oracle branch (divergence)
            record['replay'] = dict(error=str(exc), seconds=time.perf_counter()-t)
            record['verdict'] = 'FAIL_REPLAY'
            record['wall_seconds'] = time.perf_counter()-t0
            (out_dir/f'{key}.json').write_text(json.dumps(record, indent=1, default=float))
            return record
        m = rep['metrics']
        identical = rep['actions'] == st.actions
        ok = (identical and m.get('success') == 1.0 and m.get('human_collision') == 0.0 and
              rep['static_contact_steps'] == 0 and rep['body_overlap_steps'] == 0 and
              rep['shadow_pose_mismatch_steps'] == 0 and not rep['canonical_human_collision'] and
              rep['dynamics_violation_steps'] == 0)
        verdict = 'PASS' if ok else 'FAIL_REPLAY'
        record['replay'] = dict(metrics=m, actions_identical=identical, steps=rep['steps'],
                                static_contact_steps=rep['static_contact_steps'],
                                dynamics_violation_steps=rep['dynamics_violation_steps'],
                                spawn_contact=rep['spawn_contact'], escape_contact_steps=rep['escape_contact_steps'],
                                body_overlap_steps=rep['body_overlap_steps'],
                                min_rect_distance=rep['min_rect_distance'],
                                shadow_pose_mismatch_steps=rep['shadow_pose_mismatch_steps'],
                                official_vs_canonical_human_mismatch_steps=rep['official_vs_canonical_human_mismatch_steps'],
                                final_geodesic=rep['final_geodesic'], seconds=time.perf_counter()-t,
                                trace=compact_trace(rep))
    record['verdict'] = verdict
    record['wall_seconds'] = time.perf_counter()-t0
    (out_dir/f'{key}.json').write_text(json.dumps(record, indent=1, default=float))
    return record


def run_executor_only(spec, out_dir, planner='orca', dynamics='native', horizon=16):
    """No high-level selection: the legal goal query, continued every decision."""
    from .episode import EpisodeRuntime, replay
    from .executor import Query
    from .dynamics import FAMILIES
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dyn = FAMILIES[dynamics]
    t0 = time.perf_counter()
    rt = EpisodeRuntime(spec, planner, dyn, strict=False)
    st = rt.root
    q = Query('goal', label='goal')
    while not st.terminal:
        cont = st.execution is not None and st.execution.status == 'running'
        st = rt.run(st, None if cont else q, horizon=horizon, cont=cont)
    search_s = time.perf_counter()-t0
    record = dict(spec=spec.asdict(), key=spec.key, planner=planner, dynamics=dynamics, mode='executor_only',
                  code=git_identity(), code_snapshot=snapshot_identity(), start_geodesic=rt.start_geodesic,
                  spawn_contact=rt.spawn_contact,
                  branch=dict(outcome=st.outcome, steps=st.index, static_contacts=st.static_contacts,
                              escape_contacts=st.escape_contacts, body_overlaps=st.body_overlaps,
                              min_rect=st.min_rect_dist, path_length=st.path_length, seconds=search_s,
                              runtime_perf=rt.perf))
    rt.close()
    rep = replay(spec, planner, dyn, st.decisions)
    m = rep['metrics']
    record['replay'] = dict(metrics=m, actions_identical=rep['actions'] == st.actions, steps=rep['steps'],
                            static_contact_steps=rep['static_contact_steps'], body_overlap_steps=rep['body_overlap_steps'],
                            min_rect_distance=rep['min_rect_distance'], final_geodesic=rep['final_geodesic'],
                            escape_contact_steps=rep['escape_contact_steps'],
                            canonical_human_collision=rep['canonical_human_collision'], trace=compact_trace(rep))
    record['decisions'] = _decisions_json(st.decisions)
    strict_ok = (m.get('success') == 1.0 and m.get('human_collision') == 0.0 and rep['static_contact_steps'] == 0
                 and rep['body_overlap_steps'] == 0 and rep['dynamics_violation_steps'] == 0)
    record['verdict'] = 'PASS' if strict_ok else ('NATIVE_SUCCESS_ONLY' if m.get('success') == 1.0 else 'FAIL')
    record['wall_seconds'] = time.perf_counter()-t0
    (out_dir/f"{spec.key.replace('/', '_')}.json").write_text(json.dumps(record, indent=1, default=float))
    return record


def summarize(out_dir):
    rows = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob('*.json'))
            if p.name not in ('summary.json', 'run.json')]
    rows = [r for r in rows if 'verdict' in r]
    n = len(rows)
    passed = [r for r in rows if r['verdict'] == 'PASS']
    summ = dict(n=n, passed=len(passed), pass_rate=len(passed)/max(n, 1),
                verdicts={v: sum(r['verdict'] == v for r in rows) for v in sorted({r['verdict'] for r in rows})},
                search_outcomes={k: sum(str(r.get('oracle', r.get('branch'))['outcome']) == k for r in rows)
                                 for k in sorted({str(r.get('oracle', r.get('branch'))['outcome']) for r in rows})},
                mean_spl=sum(r.get('replay', {}).get('metrics', {}).get('spl', 0.) for r in rows)/max(n, 1),
                mean_steps_pass=(sum(r['replay']['steps'] for r in passed)/len(passed)) if passed else None,
                native_success=sum(r.get('replay', {}).get('metrics', {}).get('success', 0.) for r in rows),
                human_collision_eps=sum(r.get('replay', {}).get('metrics', {}).get('human_collision', 0.) for r in rows),
                mean_psc=(sum(r['replay']['metrics'].get('psc', 0.) for r in rows if 'replay' in r)/
                          max(1, sum('replay' in r for r in rows))),
                mean_oracle_seconds=sum(r.get('oracle', r.get('branch'))['seconds'] for r in rows)/max(n, 1),
                failures=[dict(key=r['key'], verdict=r['verdict'], outcome=r.get('oracle', r.get('branch'))['outcome'],
                               final_geodesic=r.get('oracle', {}).get('final_geodesic', r.get('replay', {}).get('final_geodesic')))
                          for r in rows if r['verdict'] != 'PASS'])
    (Path(out_dir)/'summary.json').write_text(json.dumps(summ, indent=1))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['one', 'suite', 'summarize', 'exec_one'])
    ap.add_argument('--policy', default='oracle', choices=['oracle', 'executor'])
    ap.add_argument('args', nargs='*')
    ap.add_argument('--out', required=True)
    ap.add_argument('--planner', default='orca')
    ap.add_argument('--dynamics', default='native')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--oracle', default='{}', help='JSON OracleConfig overrides')
    ap.add_argument('--gen', default='{}', help='JSON GeneratorConfig overrides')
    ap.add_argument('--only', default='', help='comma-separated scene names (suite mode)')
    a = ap.parse_args()
    out = Path(a.out)
    if a.mode == 'one':
        key, seed = a.args
        ds, split, scene, ep = key.split('/')
        r = run_one(EpisodeSpec(ds, split, scene, ep, int(seed)), out, a.planner, a.dynamics,
                    json.loads(a.oracle), json.loads(a.gen))
        print(json.dumps(dict(key=r['key'], verdict=r['verdict'], oracle=r['oracle']['outcome'],
                              steps=r['oracle']['steps'], seconds=round(r['wall_seconds'], 1))))
        return
    if a.mode == 'exec_one':
        key, seed = a.args
        ds, split, scene, ep = key.split('/')
        r = run_executor_only(EpisodeSpec(ds, split, scene, ep, int(seed)), out, a.planner, a.dynamics)
        print(json.dumps(dict(key=r['key'], verdict=r['verdict'], steps=r['replay']['steps'])))
        return
    if a.mode == 'summarize':
        print(json.dumps(summarize(out), indent=1))
        return
    from .suites import load
    specs = load(a.args[0])
    if a.only:
        keep = set(a.only.split(','))
        specs = [s for s in specs if s.scene in keep]
    out.mkdir(parents=True, exist_ok=True)
    # Freeze the code for the whole suite: every episode subprocess imports
    # eventwm_graph.querygen from this snapshot (see scripts/querygen_run.py).
    base = Path(__file__).resolve().parents[2]
    snap = (out/'code_snapshot'/'eventwm_graph').resolve()
    if not (snap/'querygen').exists():
        shutil.copytree(base/'eventwm_graph'/'querygen', snap/'querygen', ignore=shutil.ignore_patterns('__pycache__'))
    (out/'run.json').write_text(json.dumps(dict(argv=sys.argv, code=git_identity(), started=time.time(),
                                                snapshot=str(snap)), indent=1))
    pending = [s for s in specs if not (out/f"{s.key.replace('/', '_')}.json").exists()]
    procs = []
    py = sys.executable
    runner = str(Path(__file__).resolve().parents[2]/'scripts/querygen_run.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '1'), QG_SNAPSHOT=str(snap))
    logs = out/'logs'; logs.mkdir(exist_ok=True)
    while pending or procs:
        while pending and len(procs) < a.workers:
            s = pending.pop(0)
            cmd = ['nice', '-n', '10', py, runner, 'eventwm_graph.querygen.evaluate',
                   'one' if a.policy == 'oracle' else 'exec_one', s.key, str(s.seed),
                   '--out', str(out), '--planner', a.planner, '--dynamics', a.dynamics,
                   '--oracle', a.oracle, '--gen', a.gen]
            f = open(logs/f"{s.key.replace('/', '_')}.log", 'w')
            procs.append((s, subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env), f))
        time.sleep(2.)
        still = []
        for s, p, f in procs:
            if p.poll() is None:
                still.append((s, p, f))
            else:
                f.close()
                print(f'[{time.strftime("%H:%M:%S")}] done {s.key} rc={p.returncode}', flush=True)
        procs = still
    print(json.dumps(summarize(out), indent=1))


if __name__ == '__main__':
    main()
