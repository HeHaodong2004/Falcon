"""Plot the legal map, the goal route and the generator's candidates at a decision state (diagnostic).

  python scripts/querygen_run.py tests.querygen.plot_candidates KEY --prefix 'goal:16' --out PNG [--gen JSON]
"""
import argparse, json, math
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--suite', default='dev'); ap.add_argument('--prefix', default='')
    ap.add_argument('--gen', default='{}'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.gridplan import PlanningView
    from eventwm_graph.querygen.legal import FREE, BLOCKED
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    dyn = FAMILIES['native']
    rt = EpisodeRuntime(spec, 'orca', dyn, strict=False)
    gen = QueryGenerator(dyn, GeneratorConfig(**json.loads(a.gen)))
    st = rt.root
    for item in [x for x in a.prefix.split(',') if x]:
        lab, n = item.split(':')[:2]
        if lab == 'cont':
            st = rt.run(st, None, horizon=int(n), cont=True)
            continue
        qs, _ = gen.propose(st.legal, st.execution)
        q = next(x for x in qs if x.label == lab)
        st = rt.run(st, q, horizon=int(n))
    legal = st.legal
    qs, diag = gen.propose(legal, st.execution)
    g = legal.map
    img = np.full(g.state.shape+(3,), 1.)
    img[g.state == FREE] = (.82, .9, .82)
    img[g.state == BLOCKED] = (.25, .25, .25)
    x0 = g.ox*g.res; z0 = g.oz*g.res
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.transpose(img, (1, 0, 2)), origin='lower', interpolation='nearest',
              extent=(x0, x0+g.state.shape[0]*g.res, z0, z0+g.state.shape[1]*g.res))
    pose = legal.pose; goal = legal.goal
    view = PlanningView(g, (pose[0], pose[2]), (goal[0], goal[2]))
    route = view.route(view.field([(goal[0], goal[2])]), (pose[0], pose[2]))
    if route is not None:
        ax.plot(route[:, 0], route[:, 1], 'b-', lw=1.5, label='legal goal route')
    for q in qs:
        for v in q.via:
            ax.plot(v[0], v[1], 'o', ms=5)
            ax.annotate(q.label, (v[0], v[1]), fontsize=7)
    ax.arrow(pose[0], pose[2], .4*math.cos(pose[3]), -.4*math.sin(pose[3]), width=.03, color='k')
    ax.plot(goal[0], goal[2], 'r*', ms=16)
    P = np.asarray(legal.poses)
    ax.plot(P[:, 0], P[:, 2], 'k.', ms=2)
    ax.set_xlim(pose[0]-4, pose[0]+4); ax.set_ylim(pose[2]-4, pose[2]+4); ax.set_aspect('equal')
    ax.set_title(f'{spec.key} t={st.index} candidates: ' + ', '.join(q.label for q in qs), fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out, dpi=90)
    rt.close()


if __name__ == '__main__':
    main()
