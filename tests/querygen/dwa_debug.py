"""Diagnostic: why does the DWA planner hold?  Replays a decision prefix, then evaluates every DWA candidate of
one query's first step and prints admissibility (first body failure step), min human clearance, progress
and score.

  python scripts/querygen_run.py tests.querygen.dwa_debug KEY --prefix 'goal:16,...' --label goal
      [--suite dev] [--dynamics inertia] [--gen JSON]
"""
import argparse
import json
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--prefix', default='')
    ap.add_argument('--label', default='goal')
    ap.add_argument('--dynamics', default='inertia')
    ap.add_argument('--gen', default='{}')
    a = ap.parse_args()
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen import planners as P
    from eventwm_graph.querygen.control import body_cost
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    dyn = FAMILIES[a.dynamics]
    rt = EpisodeRuntime(spec, 'dwa', dyn, strict=True)
    gen = QueryGenerator(dyn, GeneratorConfig(**json.loads(a.gen)))
    st = rt.root
    for item in [x for x in a.prefix.split(',') if x]:
        lab, n = item.split(':')
        qs, _ = gen.propose(st.legal, st.execution)
        q = next((x for x in qs if x.label == lab), None)
        st = rt.run(st, q, horizon=int(n))
    p = st.legal.pose
    print(f't={st.index} pose=({p[0]:.2f},{p[2]:.2f},{math.degrees(p[3]):.0f}) level={st.dyn_state} geo={st.geodesic:.2f}')
    qs, _ = gen.propose(st.legal, st.execution)
    q = next(x for x in qs if x.label == a.label)
    orig = P.DWAPlanner._act

    def traced(self, ctx, adapter):
        cfg = self.cfg
        budget0 = body_cost(ctx.gmap, ctx.P, ctx.psi) if ctx.gmap is not None else 0
        start_cost = ctx.progress_fn(ctx.P) if (ctx.progress_fn and not ctx.hold) else 0.
        print(f'  ctx: hold={ctx.hold} budget0={budget0} plan_prims={list(ctx.plan_prims or [])[:12]} '
              f'route_dir={ctx.route_dir} start_cost={start_cost:.2f} humans={len(ctx.humans or [])}')
        self._memo = {}
        for name, intents, cap in self._sequences(ctx):
            steps, tail = self._rollout(ctx, intents, cap, budget0)
            traj = steps+tail
            bad = None
            for k, (Pk, psi, _, _) in enumerate(traj):
                if ctx.gmap is not None and k % 2 == 0 and not self._body_ok(ctx.gmap, Pk, psi, budget0):
                    bad = k
                    break
            Pend = steps[-1][0]
            prog = 0. if ctx.hold or not ctx.progress_fn else start_cost-ctx.progress_fn(Pend)
            print(f'    {name:22s} body_fail_step={bad} progress={prog:+.2f} first={steps[0][3]}')
        return orig(self, ctx, adapter)
    P.DWAPlanner._act = traced
    ex = rt.new_execution(q, None, None)
    prim = ex.step(st.legal, st.dyn_state)
    print('chosen', prim, 'mode', getattr(ex.planner, 'mode', None))
    rt.close()


if __name__ == '__main__':
    main()
