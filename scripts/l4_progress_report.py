"""Matched progress-first development results; failures remain in denominators."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from l4_summarize import aggregate, percentiles


def pair(rows, baseline):
    before={r['case_id']:r for r in baseline};after={r['case_id']:r for r in rows if r['variant']=='B1'}
    result=[]
    for case,b in before.items():
        a=after[case];causes=[]
        if not a.get('success'):
            if a.get('human_collision'):causes.append('human_collision')
            if a.get('static_collision'):causes.append('static_interaction_observed')
            if a.get('compile_error'):causes.append('query_geometry_compilation_limitation')
            if a.get('timeout'):causes.append('native_episode_limit')
            if not causes:causes.append('other_native_outcome')
        result.append(dict(case_id=case,baseline_success=bool(b.get('success')),candidate_success=bool(a.get('success')),
            category=('success' if b.get('success') else 'failure')+'_to_'+('success' if a.get('success') else 'failure'),
            candidate_failure_evidence=causes,baseline_human_collision=b.get('human_collision'),candidate_human_collision=a.get('human_collision')))
    counted=Counter(r['category'] for r in result)
    counts={k:counted[k] for k in ('success_to_success','success_to_failure','failure_to_success','failure_to_failure')}
    return dict(counts=counts,recovered_minus_lost=counts.get('failure_to_success',0)-counts.get('success_to_failure',0),episodes=result,
                caveat='Static contact is an observed interaction, not proof of its causal role. Native-limit failures include progress diagnostics in rows.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True,type=Path);p.add_argument('--roots',required=True,type=Path,nargs='+')
    p.add_argument('--output',required=True,type=Path);p.add_argument('--historical',type=Path);p.add_argument('--variants',default='B0,B1,B2')
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());result=aggregate(protocol,a.roots,a.variants.split(','))
    if a.historical:
        old=json.loads(a.historical.read_text());result['historical_summary']=old['summary']
        result['historical_sha256']=hashlib.sha256(a.historical.read_bytes()).hexdigest()
    profile=defaultdict(lambda:defaultdict(list))
    evidence=[]
    for root in a.roots:
        for folder in sorted(root.glob('*')):
            if not (folder/'report.json').exists():continue
            r=json.loads((folder/'report.json').read_text());t=json.loads((folder/'actual.json').read_text());v=r['variant']
            row=next(x for x in result['rows'] if x['case_id']==r['case_id'] and x['variant']==v)
            counts=Counter(t['actions']);alternating=maximum=0
            for x,y in zip(t['actions'],t['actions'][1:]):
                alternating=alternating+1 if x!=y and x in ('left','right') and y in ('left','right') else 0
                maximum=max(maximum,alternating)
            row.update(action_counts=dict(counts),longest_consecutive_turn_reversals=maximum,
                       avoidance_attempts=r.get('feedback',{}).get('avoidance_attempts'),
                       avoidance_failures=r.get('feedback',{}).get('avoidance_failures'))
            n=len(t['actions'])-int(bool(t['actions']) and t['actions'][-1]=='stop')
            row.update(query_execution_native_actions=n,
                       query_execution_human_mesh_contact=any(x['human_mesh_contact'] for x in t['offline_labels'][:n]),
                       post_query_stop_human_mesh_contact=any(x['human_mesh_contact'] for x in t['offline_labels'][n:]))
            if (folder/'reference.json').exists():
                from eventwm_graph.l4 import QuerySpec,CompiledReference
                from eventwm_graph.l4.progress import RouteProgress
                ref=json.loads((folder/'reference.json').read_text());ref['query']=QuerySpec(**ref['query'])
                tracker=RouteProgress(CompiledReference(**ref))
                for pose,stamp in zip(t['poses'],t['timestamps']):tracker.update(pose,stamp)
                row.update(actual_route_progress_m=tracker.progress,actual_route_deviation_m=tracker.deviation,
                           actual_path_length_m=r['path_length'],detours=tracker.detours,rejoins=tracker.rejoins,
                           actual_query_endpoint_distance_m=float(np.linalg.norm(np.asarray(t['poses'][-1])[:3]-ref['query'].target)))
            for file in ('case.json','reference.json','actual.json','report.json'):
                path=folder/file
                if path.exists():evidence.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for d in t['decisions']:
                c=d['collision_check'];profile[v]['nominal_ms' if d['reason']=='nominal_tracking' else 'reactive_ms'].append(d['control_ms'])
                if d['reason'] in ('reactive_avoidance','avoidance_unavailable_nominal'):
                    profile[v]['avoidance_attempt_control_ms'].append(d['control_ms'])
                elif d['reason']=='committed_avoidance':
                    profile[v]['committed_maneuver_control_ms'].append(d['control_ms'])
                for key in ('tracking_ms','static_ms','reactive_ms'):
                    if key in c:profile[v][key+'_component'].append(c[key])
            for key in ('l4_transition_ms','l4_online_ms','l2_request_ms'):
                profile[v][key].extend(r.get(key,[]))
    result['native_profile']={v:{k:percentiles(values) for k,values in cols.items()} for v,cols in profile.items()}
    result['paired_current_B0']=pair(result['rows'],[r for r in result['rows'] if r['variant']=='B0'])
    if a.historical:
        result['paired_original_B0']=pair(result['rows'],[r for r in old['rows'] if r['variant']=='B0'])
        result['paired_historical_B1']=pair(result['rows'],[r for r in old['rows'] if r['variant']=='B1'])
    comparisons=[]
    for c in protocol['cases']:
        arms={r['variant']:r for r in result['rows'] if r['case_id']==c['case_id']}
        b=arms['B0'];x=arms['B1']
        comparisons.append(dict(case_id=c['case_id'],nominal_query_completed=b.get('query_completed'),reactive_query_completed=x.get('query_completed'),
            additional_native_actions=x.get('native_actions',0)-b.get('native_actions',0),
            additional_query_execution_actions=x.get('query_execution_native_actions',0)-b.get('query_execution_native_actions',0),
            additional_path_length_m=(x.get('path_length') or 0.)-(b.get('path_length') or 0.),
            nominal_route_progress_m=b.get('actual_route_progress_m'),reactive_route_progress_m=x.get('actual_route_progress_m'),
            nominal_human_collision=b.get('human_collision'),reactive_human_collision=x.get('human_collision'),
            reactive_detours=x.get('detours'),reactive_rejoins=x.get('rejoins')))
    result['paired_query_execution']=comparisons
    result['paired_query_completion_counts']={k:0 for k in ('success_to_success','success_to_failure','failure_to_success','failure_to_failure')}
    for r in comparisons:
        key=('success' if r['nominal_query_completed'] else 'failure')+'_to_'+('success' if r['reactive_query_completed'] else 'failure')
        result['paired_query_completion_counts'][key]+=1
    result['evidence_manifest']=evidence
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({v:{k:r.get(k) for k in ('n','successes','human_collision_count','static_collision_count','spl','stl','psc','native_steps','native_steps_per_wall_second','total_turn_reversals','max_turn_run')} for v,r in result['summary'].items()},indent=2))
    print(json.dumps(result['paired_current_B0']['counts'],indent=2))


if __name__=='__main__':main()
