"""Offline measurements of retained release traces; no counterfactual outcomes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import math
import numpy as np


def diagnose(root):
    rows = []
    for folder in sorted(root.glob('*_B1')):
        report = json.loads((folder / 'report.json').read_text())
        trace = json.loads((folder / 'actual.json').read_text())
        reference = folder / 'reference.json'
        row = dict(case_id=report['case_id'], status=report['status'],
                   reason=report.get('feedback', {}).get('blocked_reason'),
                   trace=str(folder / 'actual.json'),
                   trace_sha256=hashlib.sha256((folder / 'actual.json').read_bytes()).hexdigest())
        actions = trace['actions']
        row['action_counts'] = dict(Counter(actions))
        row['adjacent_turn_reversals'] = sum(a != b and a in ('left', 'right') and b in ('left', 'right')
                                             for a, b in zip(actions, actions[1:]))
        run = longest = 0
        for a in actions:
            run = run + 1 if a in ('left', 'right') else 0
            longest = max(longest, run)
        row['longest_turn_run'] = longest
        decisions = trace['decisions']
        row['precise_fallback_calls'] = sum(bool(d['collision_check'].get('precise_static_fallback')) for d in decisions)
        row['static_rejections'] = sum(d['collision_check'].get('static_rejected', 0) for d in decisions)
        row['human_rejections'] = sum(d['collision_check'].get('human_rejected', 0) for d in decisions)
        row['nonforward_when_nominal_forward'] = 0
        row['nonforward_when_nominal_forward_no_tracks'] = 0
        if reference.exists():
            ref = json.loads(reference.read_text()); q = ref['query']; route = np.asarray(q['reference_route'])
            target = 1 if len(route) > 1 else 0
            for pose, ob, d in zip(trace['poses'], trace['observations'], decisions):
                pose = np.asarray(pose)
                while target < len(route)-1 and np.linalg.norm(pose[:3]-route[target]) <= q['position_tolerance']:
                    target += 1
                delta = route[target]-pose[:3]
                desired = q['terminal_orientation'] if np.linalg.norm(delta) <= q['position_tolerance'] else math.atan2(-delta[2], delta[0])
                error = math.atan2(math.sin(desired-pose[3]), math.cos(desired-pose[3]))
                if abs(error) <= q['orientation_tolerance'] and d['action'] != 'forward':
                    row['nonforward_when_nominal_forward'] += 1
                    row['nonforward_when_nominal_forward_no_tracks'] += int(not ob['tracks'])
            fb = report.get('feedback', {})
            row.update(final_next_waypoint=target, next_waypoint_distance_m=float(np.linalg.norm(np.asarray(trace['poses'][-1])[:3]-route[target])),
                       final_route_deviation_m=fb.get('route_deviation'), detours=fb.get('detours'), rejoins=fb.get('rejoins'))
        rows.append(row)
    return dict(source_release='167a53a92b93305306519f1e47c8393ffd0475ed', rows=rows,
                caveat='Nominal override counts measure action disagreement, not proof that counterfactual forward actions would succeed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = diagnose(a.root)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:sum(r[k] for r in result['rows']) for k in ('adjacent_turn_reversals','precise_fallback_calls','nonforward_when_nominal_forward','nonforward_when_nominal_forward_no_tracks')}, indent=2))
