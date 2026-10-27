"""Re-verify recorded solutions of a robot with momentum when STOP must first brake to rest (v10 rule).

  python scripts/querygen_run.py tests.querygen.rebrake_check RUN_DIR [RUN_DIR ...]

For every replay-verified PASS whose final STOP was issued while the robot still had a speed level > 0,
the recorded decision sequence is replayed (authoritative continuous replay, current executor: the STOP
decision brakes at the brake limit and then stops) and the strict verdict is recomputed.
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
            vc = r['replay']['trace']['velocity_commands']
            lvl = int(round(vc[-1][0]/120./dyn.levels[1])) if vc and vc[-1] else 0
            if lvl == 0:
                continue
            decisions = []
            for x in r['decisions']:
                q = Query(x['kind'], via=tuple(tuple(v) for v in x['via']), heading=x['heading'],
                          duration=x['duration'], speed=x['speed'], label=x['label'], meta=x['meta'] or {})
                decisions.append(dict(query=q, cont=x['cont'], horizon=x['horizon']))
            decisions[-1]['horizon'] = 1+dyn.stop_steps(lvl)
            s = r['spec']
            spec = EpisodeSpec(s['dataset'], s['split'], s['scene'], s['episode_id'], s['seed'])
            rep = replay(spec, r['planner'], dyn, decisions)
            m = rep['metrics']
            ok = (bool(m.get('success')) and not m.get('human_collision') and rep['body_overlap_steps'] == 0 and
                  rep['static_contact_steps'] == 0 and rep['shadow_pose_mismatch_steps'] == 0)
            row = dict(key=r['key'], run=d, stop_level=lvl, old_final_geodesic=r['replay']['final_geodesic'],
                       new_final_geodesic=rep['final_geodesic'], success=bool(m.get('success')),
                       overlaps=rep['body_overlap_steps'], statics=rep['static_contact_steps'], strict_pass=ok)
            out.append(row)
            print(json.dumps(row), flush=True)
    n = len(out)
    print(f'{sum(x["strict_pass"] for x in out)}/{n} moving-STOP solutions still pass after braking to rest '
          f'(success radius {SUCCESS_DISTANCE} m)')


if __name__ == '__main__':
    main()
