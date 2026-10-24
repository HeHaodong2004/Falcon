"""Audit progress-first native ledgers and driver termination, independently."""
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
        assert not {'offline_humans','human_goals','human_future','oracle_positions'}&set(ob)
        for person in ob['tracks']:
            assert person['last_seen_time']<=stamp+1e-7
            if person['visible_now']:assert abs(person['last_seen_time']-stamp)<1e-7
    fb=r.get('feedback')
    if fb:
        n=len(fb['actual_actions'])
        assert fb['actual_actions']==actions[:n] and 'stop' not in fb['actual_actions']
        assert np.array_equal(np.asarray(fb['actual_poses']),poses[:n+1])
        assert np.array_equal(np.asarray(fb['actual_timestamps']),times[:n+1])
        assert len(fb['actual_observation_refs'])==len(fb['transition_events'])==n
        assert fb['query_status'] not in ('BLOCKED','EXECUTOR_LIMIT')
        assert not any(d['collision_check'].get('precise_static_fallback') for d in t['decisions'])
        if actions and actions[-1]=='stop':
            assert r['query_completed'], 'Supported active query ended artificially'
        if r['query_completed']:
            assert not fb['episode_termination'] and not fb['episode_truncation']
    if r.get('query_completed'):
        ref=json.loads((folder/'reference.json').read_text());q=ref['query']
        used=poses[:len(fb['actual_actions'])+1] if fb else poses
        p=used[-1]
        assert np.linalg.norm(p[:3]-q['reference_route'][-1])<=q['position_tolerance']+1e-6
        assert abs(math.atan2(math.sin(p[3]-q['terminal_orientation']),math.cos(p[3]-q['terminal_orientation'])))<=q['orientation_tolerance']+1e-6
        # Verify ordered gates independently: disc visit, crossing its narrow
        # route corridor, or local arrival on the adjacent outgoing segment.
        route=np.asarray(q['reference_route']);index=0
        for i,b in enumerate(route[1:-1],1):
            v=b-route[i-1];length=np.linalg.norm(v);unit=v/max(length,1e-12)
            w=route[i+1]-b;wl=np.linalg.norm(w);wu=w/max(wl,1e-12)
            found=False
            while index<len(used):
                point=used[index,:3];along=np.dot(point-route[i-1],unit)
                lateral=np.linalg.norm(point-route[i-1]-along*unit)
                na=np.dot(point-b,wu);nl=np.linalg.norm(point-b-na*wu)
                reached=np.linalg.norm(point-b)<=q['position_tolerance']+1e-6
                passed=length<=along<=length+1.2 and lateral<=.25+1e-6
                corner=along>=max(0.,length-.25) and 0.<na<=min(wl,1.2) and nl<=.20+1e-6 and np.dot(unit,wu)>-.5
                if reached or passed or corner:found=True;break
                index+=1
            assert found,'Skipped ordered route gate'
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
