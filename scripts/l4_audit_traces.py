"""Independent offline audit of actual protocol evidence and completion claims."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def audit(folder):
    r=json.loads((folder/'report.json').read_text());t=json.loads((folder/'actual.json').read_text())
    actions=t['actions'];poses=np.asarray(t['poses']);times=np.asarray(t['timestamps'])
    assert len(poses)==len(times)==len(actions)+1
    assert all(a in ('forward','left','right','stop') for a in actions)
    assert 'stop' not in actions[:-1]
    assert np.all(np.diff(times)>0)
    assert np.all((np.diff(times)>=.048-1e-6)&(np.diff(times)<=.056+1e-6))
    assert abs(r['actual_duration']-(times[-1]-times[0]))<1e-7
    assert len(t['observations'])==len(times)
    for ob,p,stamp in zip(t['observations'],poses,times):
        assert abs(ob['timestamp']-stamp)<1e-7 and np.allclose(ob['robot_pose'],p)
        for h in ob['tracks']:
            assert h['last_seen_time']<=stamp+1e-7
            if h['visible_now']:assert abs(h['last_seen_time']-stamp)<1e-7
        assert not {'offline_humans','human_goals','human_future','oracle_positions'}&set(ob)
    fb=r.get('feedback')
    if fb:
        n=len(fb['actual_actions'])
        assert fb['actual_actions']==actions[:n] and 'stop' not in fb['actual_actions']
        assert np.array_equal(np.asarray(fb['actual_poses']),poses[:n+1])
        assert np.array_equal(np.asarray(fb['actual_timestamps']),times[:n+1])
        assert len(fb['actual_observation_refs'])==n
        assert abs(fb['actual_duration']-(times[n]-times[0]))<1e-7
        if fb['query_status']=='QUERY_COMPLETED':
            assert not any(x['human_mesh_contact'] or x['static_obstacle_contact'] for x in t['offline_labels'][:n])
    if r.get('query_completed'):
        ref=json.loads((folder/'reference.json').read_text());q=ref['query']
        p=poses[len(fb['actual_actions'])] if fb else poses[-1]
        assert np.linalg.norm(p[:3]-np.asarray(q['reference_route'][-1]))<=q['position_tolerance']+1e-6
        assert abs(math.atan2(math.sin(p[3]-q['terminal_orientation']),math.cos(p[3]-q['terminal_orientation'])))<=q['orientation_tolerance']+1e-6
        index=0
        for point in q['reference_route']:
            while index<len(poses) and np.linalg.norm(poses[index,:3]-point)>q['position_tolerance']+1e-6:index+=1
            assert index<len(poses),'Skipped ordered waypoint'
    if r['metrics']['success']:
        assert actions[-1]=='stop' and r['metrics']['distance_to_goal']<.2
    return dict(case_id=r['case_id'],variant=r['variant'],actions=len(actions),pass_audit=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
    for root in a.roots:
        for f in sorted(root.glob('*/report.json')):
            try:rows.append(audit(f.parent))
            except Exception as exc:rows.append(dict(folder=str(f.parent),pass_audit=False,error=repr(exc)))
    a.output.write_text(json.dumps(dict(results=rows,passed=sum(r['pass_audit'] for r in rows),total=len(rows)),indent=2))
    print(f"{sum(r['pass_audit'] for r in rows)}/{len(rows)} trace audits passed")
    if not rows or not all(r['pass_audit'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
