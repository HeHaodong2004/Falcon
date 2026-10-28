"""Re-verify recorded solutions of a robot with momentum under the v10 motion rules.

  python scripts/querygen_run.py tests.querygen.rebrake_check RUN_DIR [RUN_DIR ...]

For every replay-verified PASS whose executed commands break the dynamics model (dynamics.dynamics_violations:
STOP while moving; v9 DWA "blocked" fallback = instantaneous stop), the recorded *query sequence* is replayed
with the current executor/planners (STOP brakes to rest at the brake limit; a blocked DWA brakes) and the
strict verdict, now including zero dynamics violations, is recomputed.
"""
import argparse
import glob
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    from eventwm_graph.querygen.native import EpisodeSpec
    from eventwm_graph.querygen.episode import replay
    from eventwm_graph.querygen.executor import Query
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.native import SUCCESS_DISTANCE
    out = []
    for d in a.runs:
        for f in sorted(glob.glob(f'{d}/*.json')):
            if f.endswith('run.json'):
                continue
            r = json.loads(Path(f).read_text())
            if r.get('verdict') != 'PASS':
                continue
            dyn = FAMILIES[r['dynamics']]
            if not dyn.stateful:
                continue
            tr = r['replay']['trace']
            vc = tr['velocity_commands']
            it = iter(vc)
            cmds = ['stop' if x == 's' else next(it) if x == 'v' else x for x in tr['actions']]
            from eventwm_graph.querygen.dynamics import dynamics_violations
            nviol = dynamics_violations(dyn, cmds)
            if nviol == 0:
                continue
            lvl = int(round(vc[-1][0]/120./dyn.levels[1])) if vc and vc[-1] else 0
            decisions = []
            for x in r['decisions']:
                q = Query(x['kind'], via=tuple(tuple(v) for v in x['via']), heading=x['heading'],
                          duration=x['duration'], speed=x['speed'], label=x['label'], meta=x['meta'] or {})
                decisions.append(dict(query=q, cont=x['cont'], horizon=x['horizon']))
            decisions[-1]['horizon'] = 1+dyn.stop_steps(len(dyn.levels)-1)
            s = r['spec']
            spec = EpisodeSpec(s['dataset'], s['split'], s['scene'], s['episode_id'], s['seed'])
            rep = replay(spec, r['planner'], dyn, decisions)
            m = rep['metrics']
            ok = (bool(m.get('success')) and not m.get('human_collision') and rep['body_overlap_steps'] == 0 and
                  rep['static_contact_steps'] == 0 and rep['shadow_pose_mismatch_steps'] == 0 and
                  rep['dynamics_violation_steps'] == 0)
            row = dict(key=r['key'], run=d, v9_violations=nviol, stop_level=lvl,
                       old_final_geodesic=r['replay']['final_geodesic'], new_final_geodesic=rep['final_geodesic'],
                       success=bool(m.get('success')), overlaps=rep['body_overlap_steps'],
                       statics=rep['static_contact_steps'], violations=rep['dynamics_violation_steps'],
                       steps=rep['steps'], strict_pass=ok)
            out.append(row)
            print(json.dumps(row), flush=True)
    n = len(out)
    print(f'{sum(x["strict_pass"] for x in out)}/{n} solutions with v9 dynamics violations still pass under the '
          f'v10 motion rules (success radius {SUCCESS_DISTANCE} m)')


if __name__ == '__main__':
    main()
