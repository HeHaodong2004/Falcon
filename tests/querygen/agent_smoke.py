"""End-to-end agent smoke: goal-only policy in branch mode vs real replay.

Usage: python scripts/querygen_run.py tests.querygen.agent_smoke [planner] [dataset/split/scene/episode ...]
"""
import json
import sys
import time
from eventwm_graph.querygen.native import EpisodeSpec
from eventwm_graph.querygen.episode import EpisodeRuntime, replay
from eventwm_graph.querygen.executor import Query
from eventwm_graph.querygen.dynamics import NATIVE, FAMILIES

DEFAULT = ['hm3d/val/3t8DB4Uzvkt/1', 'hm3d/val/5cdEh9F2hJL/3', 'hm3d/val/3t8DB4Uzvkt/2']


def main():
    planner = sys.argv[1] if len(sys.argv) > 1 else 'orca'
    keys = [k for k in sys.argv[2:] if not k.startswith('--')] or DEFAULT
    dyn = NATIVE
    for a in sys.argv:
        if a.startswith('--dyn='):
            dyn = FAMILIES[a.split('=', 1)[1]]
    ok_all = True
    for k, key in enumerate(keys):
        ds, split, scene, ep = key.split('/')
        spec = EpisodeSpec(ds, split, scene, ep, 100+k)
        t0 = time.perf_counter()
        rt = EpisodeRuntime(spec, planner, dyn, strict=False, trace=True)
        st = rt.root
        q = Query('goal', label='goal')
        while not st.terminal:
            cont = st.execution is not None and st.execution.status == 'running'
            st = rt.run(st, None if cont else q, horizon=20, cont=cont)
        branch_s = time.perf_counter()-t0
        from pathlib import Path
        from eventwm_graph.querygen.viz import plot_branch
        out = Path('output/runs/smoke'); out.mkdir(parents=True, exist_ok=True)
        plot_branch(rt, st, out/f"{planner}_{dyn.name}_{key.replace('/', '_')}.png", rt.trace)
        modes = {}
        for t in rt.trace:
            modes[str(t['prim'])] = modes.get(str(t['prim']), 0)+1
            modes['m:'+str(t.get('mode'))] = modes.get('m:'+str(t.get('mode')), 0)+1
        perf = {k: round(v, 2) for k, v in rt.perf.items()}
        sim_perf = {k: round(v, 2) for k, v in rt.sim.performance.items()}
        rt.close()
        t0 = time.perf_counter()
        rep = replay(spec, planner, dyn, st.decisions) if '--no-replay' not in sys.argv else dict(
            actions=st.actions, steps=st.index, metrics={}, static_contact_steps=None, body_overlap_steps=None)
        replay_s = time.perf_counter()-t0
        same = rep['actions'] == st.actions
        m = rep['metrics']
        row = dict(episode=key, planner=planner, branch_outcome=st.outcome, steps=st.index,
                   static_steps=st.static_contacts, min_center=round(st.min_center_dist, 3),
                   psc_steps=st.psc_steps, start_geodesic=round(rt.start_geodesic, 2),
                   final_geodesic=round(st.geodesic, 3) if st.geodesic == st.geodesic else None,
                   replay_actions_identical=same, replay_steps=rep['steps'],
                   replay_success=m.get('success'), replay_human_collision=m.get('human_collision'),
                   replay_spl=m.get('spl'), replay_psc=m.get('psc'), replay_static_steps=rep['static_contact_steps'],
                   branch_s=round(branch_s, 1), replay_s=round(replay_s, 1), perf=perf, sim_perf=sim_perf,
                   actions_hist=modes, min_rect=round(st.min_rect_dist, 3), body_overlaps=st.body_overlaps,
                   replay_body_overlaps=rep['body_overlap_steps'])
        ok_all &= same
        print(json.dumps(row), flush=True)
    print('REPLAY_IDENTICAL' if ok_all else 'REPLAY_DIVERGED')


if __name__ == '__main__':
    main()
