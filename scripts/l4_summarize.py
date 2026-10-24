"""Aggregate all predeclared roots, including failures; never filter by outcome."""
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import numpy as np


def percentiles(values):
    values=np.asarray(values,float)
    return dict(n=len(values),**{name:float(np.percentile(values,p)) if len(values) else None
                               for name,p in [('p50',50),('p95',95),('p99',99),('max',100)]})


def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def cluster_interval(rows,key):
    """Exact scene-cluster resampling for the small, balanced four-scene panel."""
    clusters=defaultdict(list)
    for r in rows:
        if r.get(key) is not None:clusters[r['scene_id']].append(float(r[key]))
    names=sorted(clusters)
    if not 2<=len(names)<=5:return None
    estimates=[np.mean(sum((clusters[n] for n in draw),[]))
               for draw in itertools.product(names,repeat=len(names))]
    return [float(x) for x in np.percentile(estimates,[2.5,97.5])]


def diagnostics(trace,report,reference):
    actions=trace['actions'];times=np.asarray(trace['timestamps']);poses=np.asarray(trace['poses'])
    turn_run=longest=0;switches=0;last=None
    for a in actions:
        turn_run=turn_run+1 if a in ('left','right') else 0;longest=max(longest,turn_run)
        switches+=int(last in ('left','right') and a in ('left','right') and last!=a);last=a
    visibility=defaultdict(list)
    for ob in trace['observations']:
        for h in ob['tracks']:
            if h['visible_now']:visibility[h['track_id']].append(ob['timestamp'])
    gaps=[float(b-a) for ts in visibility.values() for a,b in zip(ts,ts[1:]) if .5<=b-a<=2.]
    # Observation IDs can change on reassociation: this measures confirmed
    # same-ID gaps only and cannot establish identity recall of disappeared IDs.
    fb=report.get('feedback',{});path=np.asarray(reference['nominal_waypoints']) if reference else None
    nominal_length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()) if path is not None else None
    nominal_time=.048*len(reference['native_action_schedule']) if reference else None
    query_duration=fb.get('actual_duration',times[-2]-times[0] if actions and actions[-1]=='stop' else times[-1]-times[0])
    recovery=[];unrejoined=None
    if path is not None and len(path)>1:
        next_waypoint=1;off_route_since=None
        for pose,stamp in zip(poses,times):
            while next_waypoint<len(path)-1 and np.linalg.norm(pose[:3]-path[next_waypoint])<=reference['query']['position_tolerance']:
                next_waypoint+=1
            starts=path[:next_waypoint];vectors=path[1:next_waypoint+1]-starts
            u=np.clip(np.sum((pose[:3]-starts)*vectors,axis=1)/np.maximum(np.sum(vectors*vectors,axis=1),1e-12),0,1)
            deviation=float(np.linalg.norm(starts+u[:,None]*vectors-pose[:3],axis=1).min())
            if deviation>.32 and off_route_since is None:off_route_since=stamp
            elif deviation<.20 and off_route_since is not None:recovery.append(float(stamp-off_route_since));off_route_since=None
        if off_route_since is not None:unrejoined=float(times[-1]-off_route_since)
    mode_seconds=defaultdict(float)
    for d,dt in zip(trace['decisions'],np.diff(times)):mode_seconds[d['mode']]+=float(dt)
    bad=trace['offline_labels'];dt=np.diff(times)
    dmin=np.array([x['official_dmin_m'] if x['official_dmin_m'] is not None else np.inf for x in bad])
    severity=float(np.sum(np.maximum(0,1.-dmin)*dt)) if len(dt)==len(dmin) else None
    return dict(longest_turn_run=longest,adjacent_turn_reversals=switches,
        turn_reversals_per_action=switches/max(1,len(actions)),route_progress=fb.get('route_progress'),
        recovery_time_completed_s=recovery,unrejoined_duration_censored_s=unrejoined,
        confirmed_reobservation_gaps_05_to_2_s=gaps,
        detours=fb.get('detours'),rejoins=fb.get('rejoins'),mode_seconds=dict(mode_seconds),
        nominal_length=nominal_length,nominal_duration=nominal_time,
        extra_path_completed=(report['path_length']-nominal_length) if report.get('query_completed') and nominal_length is not None else None,
        extra_time_completed=(query_duration-nominal_time) if report.get('query_completed') and nominal_time is not None else None,
        integrated_psc_intrusion_m_s=severity,
        visible_observation_count=sum(len(t) for t in visibility.values()),
        precise_steps=sum(bool(d['collision_check'].get('precise_static_fallback')) for d in trace['decisions']),
        forecast_valid_steps=sum(d['forecast_validity']=='valid' for d in trace['decisions']))


def aggregate(protocol,roots,variants):
    rows=[];timing=defaultdict(lambda:defaultdict(list));missing=[]
    references=defaultdict(dict);integrity=[]
    for root in roots:
        receipt=root/'completed_receipt.json'
        integrity.append(dict(path=str(root),completed=receipt.exists(),
            source_unchanged=json.loads(receipt.read_text()).get('source_unchanged') if receipt.exists() else None))
    for c in protocol['cases']:
        for v in variants:
            folders=[root/(c['case_id']+'_'+v) for root in roots]
            found=[f for f in folders if (f/'report.json').exists()]
            if len(found)>1:raise ValueError('Duplicate result '+c['case_id']+' '+v)
            row=dict(case_id=c['case_id'],scene_id=c['scene_id'],variant=v)
            if not found:
                row.update(status='MISSING_OR_INTEGRATION_ERROR',success=0.,spl=0.,stl=0.,psc=None,
                           human_collision=None,static_collision=None,query_completed=False)
                missing.append([c['case_id'],v]);rows.append(row);continue
            folder=found[0];r=json.loads((folder/'report.json').read_text());m=r['metrics']
            row.update({k:r.get(k) for k in ('status','human_collision','static_collision','query_completed','blocked','timeout',
                 'native_actions','path_length','actual_duration','wall_seconds','compile_error')})
            row.update({k:m.get(k) for k in ('success','spl','stl','psc')})
            row['query_completed']=bool(r.get('query_completed')) if v!='B3' else None
            row['blocked_reason']=r.get('feedback',{}).get('blocked_reason')
            ref=json.loads((folder/'reference.json').read_text()) if (folder/'reference.json').exists() else None
            if ref:references[c['case_id']][v]=ref['query']
            trace=json.loads((folder/'actual.json').read_text());row.update(diagnostics(trace,r,ref));rows.append(row)
            for key in ('l4_ms','environment_ms','l2_update_ms','l2_refresh_ms'):timing[v][key].extend(r[key])
            timing[v]['l4_excluding_episode_first_ms'].extend(r['l4_ms'][1:])
            timing[v]['episode_first_control_ms'].extend(r['l4_ms'][:1])
            timing[v]['episode_first_l2_refresh_ms'].extend(r['l2_refresh_ms'][:1])
            timing[v]['precise_control_ms'].extend(d['control_ms'] for d in trace['decisions'] if d['collision_check'].get('precise_static_fallback'))
            timing[v]['grid_control_ms'].extend(d['control_ms'] for d in trace['decisions'] if d['action'] and not d['collision_check'].get('precise_static_fallback'))
            timing[v]['candidate_counts'].extend(d['collision_check'].get('expanded',0) for d in trace['decisions'])
    summary={}
    for v in variants:
        rs=[r for r in rows if r['variant']==v];completed=[r for r in rs if r.get('success')]
        summary[v]=dict(n=len(rs),successes=sum(bool(r.get('success')) for r in rs),
            observed_native_outcomes=sum(r.get('human_collision') is not None for r in rs),
            human_collision_count=sum(bool(r.get('human_collision')) for r in rs),
            static_collision_count=sum(bool(r.get('static_collision')) for r in rs),
            **{k:mean([r.get(k) for r in rs]) for k in ('success','spl','stl','psc','human_collision','static_collision','query_completed','blocked','timeout')},
            statuses=dict(Counter(r['status'] for r in rs)),blocked_reasons=dict(Counter(r['blocked_reason'] for r in rs if r.get('blocked_reason'))),
            completion_time_success_only=mean([r['actual_duration'] for r in completed]),
            duration_all=mean([r.get('actual_duration') for r in rs]),
            mean_extra_path_completed=mean([r.get('extra_path_completed') for r in rs]),
            mean_extra_time_completed=mean([r.get('extra_time_completed') for r in rs]),
            total_detours=sum(r.get('detours') or 0 for r in rs),total_rejoins=sum(r.get('rejoins') or 0 for r in rs),
            total_turn_reversals=sum(r.get('adjacent_turn_reversals') or 0 for r in rs),
            mean_completed_recovery_time_s=mean([t for r in rs for t in r.get('recovery_time_completed_s',[])]),
            incomplete_rejoin_episodes=sum(r.get('unrejoined_duration_censored_s') is not None for r in rs),
            max_turn_run=max(r.get('longest_turn_run') or 0 for r in rs),
            confirmed_reobservation_gaps_05_to_2_s=[g for r in rs for g in r.get('confirmed_reobservation_gaps_05_to_2_s',[])],
            timing={k:percentiles(vals) for k,vals in timing[v].items()},
            success_scene_cluster_95_interval=cluster_interval(rs,'success'))
        wall=sum(r.get('wall_seconds') or 0 for r in rs);steps=sum(r.get('native_actions') or 0 for r in rs)
        summary[v].update(wall_seconds=wall,native_steps=steps,native_steps_per_wall_second=steps/wall if wall else None,
                          episodes_per_hour=3600*len(rs)/wall if wall else None)
    matched=all(len({json.dumps(q,sort_keys=True) for q in vs.values()})==1 for vs in references.values())
    return dict(summary=summary,rows=rows,missing=missing,receipts=integrity,
                query_semantics_matched=matched,matched_reference_roots=len(references),
                interval_method='exact percentile scene-cluster bootstrap; four scenes, exploratory panel; not a universal safety bound')


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--roots',type=Path,nargs='+',required=True);p.add_argument('--variants',default='B0,B1,B2,B3')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=aggregate(json.loads(a.protocol.read_text()),a.roots,a.variants.split(','))
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
