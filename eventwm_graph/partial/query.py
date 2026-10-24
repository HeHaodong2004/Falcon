"""Deterministic proposals: ordered points + terminal yaw, and known-space A*.

No simulator, navmesh, hidden actor identity, oracle heading, or future tape is
accepted by either public entry point. A fixed discrete route follower is an
engineering collection executor; it is not the final social L4 controller.
"""
import copy
import math
from functools import lru_cache
import numpy as np
from ..config import digest

EXECUTOR_VERSION='known_space_discrete_route_follower_1'
FORWARD_STEP=25./120.
TURN_STEP=10./120.
NOMINAL_STEP_SECONDS=.048
HUMAN_ENVELOPE=.616  # Independently audited all six assets / all walking poses.
HUMAN_SPEED_BOUND=3.0  # Declared post-projection horizontal bound; checked on native tapes.
ROBOT_BOUNDS=(-.646495,.584202,-.242846,.238225)  # all 756 leg-animation frames


def wrap(a):return math.atan2(math.sin(a),math.cos(a))


def yaw_to(a,b):
    d=np.asarray(b)-a
    return math.atan2(-d[2],d[0])


def physical_clearance(position,yaw,tracks,margin=.03):
    """Known robot oriented rectangle versus all-phase human enclosing disks.

    Positions are last LEGAL measurements. Stale positions get a motion envelope.
    Invisible current GT, hidden human yaw and simulator contacts are not read.
    This checks the proposed terminal pose, not the future execution outcome.
    """
    xmin,xmax,zmin,zmax=ROBOT_BOUNDS
    c,s=math.cos(yaw),math.sin(yaw)
    for h in tracks:
        if not h.get('valid',True):continue
        d=np.asarray(h['position'])-position
        local_x=c*d[0]-s*d[2];local_z=s*d[0]+c*d[2]
        dx=max(xmin-local_x,0.,local_x-xmax)
        dz=max(zmin-local_z,0.,local_z-zmax)
        age=float(h.get('observation_age',0.))
        uncertainty=age*HUMAN_SPEED_BOUND
        if math.hypot(dx,dz)<=HUMAN_ENVELOPE+margin+uncertainty:
            return False
    return True


@lru_cache(maxsize=16)
def _body_offsets(resolution):
    spacing=min(.10,resolution/2)
    n=int(math.ceil((ROBOT_BOUNDS[1]-ROBOT_BOUNDS[0])/spacing))+1
    return tuple(np.linspace(ROBOT_BOUNDS[0],ROBOT_BOUNDS[1],n).tolist())


def static_body_clearance(graph,position,yaw):
    """Cover the all-animation rectangle by certified .25m support disks.

    Longitudinal sample spacing <= .10m leaves >= .2449m lateral coverage,
    exceeding the measured .242846m half-width. Each sampled disk center must
    be a visible, full-square native configuration-space certificate. An exact
    ego singleton never substitutes for these translated body configurations.
    """
    # Exact-key memoization only: no rounding, interpolation, or reuse after
    # an occupancy/height revision or ego singleton change.
    ego=tuple(graph.ego_support) if graph.ego_support is not None else None
    version=(graph.revision,graph.timestamp,ego)
    if getattr(graph,'_body_cache_version',None)!=version:
        graph._body_cache={};graph._body_cache_version=version
    signature=(*position,float(yaw))
    cache=graph._body_cache
    if signature in cache:return cache[signature]
    c,s=math.cos(yaw),math.sin(yaw)
    px,py,pz=map(float,position);resolution=graph.resolution;result=True
    for x in _body_offsets(resolution):
        a,b=px+x*c,pz-x*s
        cell=graph.cells.get((math.floor(a/resolution),math.floor(b/resolution)))
        if cell is None or cell['state']!=1:
            result=False;break
        if not (abs(py-cell['height'])<=.22) and not graph.point_free([a,py,b]):
            result=False;break
    cache[signature]=result
    return result


def turn_clearance(graph,position,a,b):
    # Bound movement of the measured .67m outer body between checks to 3.4mm.
    difference=wrap(b-a)
    n=max(1,int(math.ceil(abs(difference)/.005)))
    return all(static_body_clearance(graph,position,a+difference*t) for t in np.linspace(0.,1.,n+1))


def compile_query(partial_graph,query,robot_state,max_actions=180,goal_tolerance=.155):
    pose=np.asarray(robot_state.get('robot_pose',robot_state.get('pose')),dtype=float).copy()
    points=query['waypoints']
    if not 1 <= len(points) <= 2:
        return {'ok':False,'reason':'one_or_two_ordered_waypoints_required'}
    yaw=float(query['final_yaw'])
    if not np.isfinite(pose).all() or not math.isfinite(yaw):return {'ok':False,'reason':'nonfinite_query'}
    if not static_body_clearance(partial_graph,pose[:3],pose[3]):
        return {'ok':False,'reason':'initial_full_body_support_not_observed'}
    paths=[];start=pose[:3]
    try:
        for p in points:
            path=partial_graph.route(start,p);paths.append(path);start=np.asarray(p)
    except ValueError as e:return {'ok':False,'reason':str(e)}
    route=[]
    for path in paths:
        for p in path:
            if not route or np.linalg.norm(np.asarray(route[-1])-p)>1e-6:route.append(p.tolist())
    trace=[pose.tolist()];actions=[];waypoint_arrivals=[]
    for waypoint_index,path in enumerate(paths):
        target=1 if len(path)>1 else 0
        goal=np.asarray(points[waypoint_index]);visited={}
        while np.linalg.norm(pose[:3]-goal)>goal_tolerance:
            if len(actions)>=max_actions:return {'ok':False,'reason':'compile_action_budget'}
            while target<len(path)-1 and np.linalg.norm(pose[:3]-path[target])<.22:target+=1
            look=target
            while look<len(path)-1 and partial_graph.segment_free(pose[:3],path[look+1]):look+=1
            target=look
            desired=yaw_to(pose[:3],path[target]);angle=wrap(desired-pose[3])
            if abs(angle)>TURN_STEP/2+1e-5:
                action='left' if angle>0 else 'right'
                next_yaw=wrap(pose[3]+(TURN_STEP if angle>0 else -TURN_STEP))
                if not turn_clearance(partial_graph,pose[:3],pose[3],next_yaw):
                    return {'ok':False,'reason':'intermediate_turn_body_clearance_not_certified'}
                pose[3]=next_yaw
            else:
                action='forward';nxt=pose.copy()
                nxt[0]+=FORWARD_STEP*math.cos(pose[3]);nxt[2]-=FORWARD_STEP*math.sin(pose[3])
                # Height is measured map support, not an oracle native step.
                k=partial_graph.key(nxt[:3]);cell=partial_graph.cells.get(k)
                if cell:nxt[1]=cell['height']
                if not partial_graph.segment_free(pose[:3],nxt[:3]):
                    return {'ok':False,'reason':'discrete_step_leaves_confirmed_space'}
                samples=max(1,int(math.ceil(FORWARD_STEP/.03)))
                if not all(static_body_clearance(partial_graph,pose[:3]+(nxt[:3]-pose[:3])*t,pose[3]) for t in np.linspace(0.,1.,samples+1)):
                    return {'ok':False,'reason':'forward_body_clearance_not_certified'}
                pose=nxt
            actions.append(action);trace.append(pose.tolist())
            key=tuple(np.round(pose,3))+(target,)
            visited[key]=visited.get(key,0)+1
            if visited[key]>3:return {'ok':False,'reason':'route_follower_cycle'}
        waypoint_arrivals.append(len(actions))
    # Orientation applies only NOW, after all ordered waypoint arrivals.
    while abs(wrap(yaw-pose[3]))>TURN_STEP/2+1e-5:
        if len(actions)>=max_actions:return {'ok':False,'reason':'terminal_turn_action_budget'}
        left=wrap(yaw-pose[3])>0
        next_yaw=wrap(pose[3]+(TURN_STEP if left else -TURN_STEP))
        if not turn_clearance(partial_graph,pose[:3],pose[3],next_yaw):
            return {'ok':False,'reason':'terminal_turn_body_clearance_not_certified'}
        actions.append('left' if left else 'right');pose[3]=next_yaw
        trace.append(pose.tolist())
    if not actions:return {'ok':False,'reason':'no_op_duplicate'}
    route=np.asarray(route)
    return {'ok':True,'route':route.tolist(),'route_distance':float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum()),
            'actions':actions,'planned_poses':trace,'waypoint_arrival_action_indices':waypoint_arrivals,
            'terminal_turn_actions':len(actions)-waypoint_arrivals[-1],
            'nominal_seconds':len(actions)*NOMINAL_STEP_SECONDS,
            'nominal_clock':'measured pinned Habitat .048s/action; actual timestamps authoritative',
            'final_yaw_error':abs(wrap(yaw-pose[3])),'position_tolerance':goal_tolerance,
            'yaw_tolerance':TURN_STEP/2+1e-5,'executor_version':EXECUTOR_VERSION,
            'graph_revision':partial_graph.revision,'status':'known_space_route_and_discrete_motion_certified'}


def generate_queries(partial_graph,human_tracks,ego_state,pointgoal,limit=32,safe_margin=.2,intrusion_radius=.96):
    if not math.isfinite(safe_margin) or safe_margin<=0:raise ValueError('safe_margin_must_be_positive')
    if not math.isfinite(intrusion_radius) or not 0<intrusion_radius<1.:raise ValueError('intrusion_radius_must_be_inside_psc')
    pose=np.asarray(ego_state.get('robot_pose',ego_state.get('pose')),dtype=float)
    now=float(ego_state['timestamp']);goal=np.asarray(pointgoal,dtype=float)
    if partial_graph.timestamp != now:raise ValueError('Map and query decision timestamp differ')
    for h in human_tracks:
        if h['last_seen_time']>now+1e-8:raise ValueError('Future human estimate')
    reachable=partial_graph.reachable(pose[:3])
    spatial=[];rejections=[]
    def propose(p,provenance,extra_yaws=()):
        p=np.asarray(p,dtype=float);source=copy.deepcopy(provenance)
        reason=None
        if not partial_graph.point_free(p):reason='not_certified_known_free'
        elif partial_graph.key(p) not in reachable and provenance['source']!='current_position':reason='disconnected_known_free_component'
        if reason:
            rejections.append({'position':p.tolist(),'provenance':source,'reason':reason});return False
        for candidate in spatial:
            if np.linalg.norm((np.asarray(candidate['position'])-p)[[0,2]])<.10:
                rejections.append({'position':p.tolist(),'provenance':source,'reason':'near_duplicate_spatial_anchor'});return False
        spatial.append({'position':p.tolist(),'provenance':source,'extra_yaws':list(extra_yaws)})
        return True
    # The exact ego anchor takes precedence over nearby discretized anchors;
    # otherwise position deduplication can erase in-place observation queries.
    propose(pose[:3],{'source':'current_position'},[pose[3]+.25,pose[3]-.25,pose[3]+math.pi/2,pose[3]-math.pi/2,pose[3]+math.pi])
    # Up to 8 spatial proposals/person. Distance bands never change semantics.
    for h in sorted(human_tracks,key=lambda x:x['track_id']):
        if not h.get('valid',True) or not h.get('visible_now',False):continue
        center=np.asarray(h['position'],dtype=float)
        velocity=np.asarray(h.get('estimated_velocity') or [0.,0.,0.])
        speed=float(np.linalg.norm(velocity[[0,2]]))
        reliable=h.get('velocity_valid',False) and h.get('num_observations',0)>=3 and speed>.1
        front=velocity.copy() if reliable else pose[:3]-center
        front[1]=0
        if np.linalg.norm(front)<1e-8:front=np.array([math.cos(pose[3]),0.,-math.sin(pose[3])])
        front=front/np.linalg.norm(front);left=np.array([front[2],0.,-front[0]])
        for direction,axis in (('FRONT',front),('BACK',-front),('LEFT',left),('RIGHT',-left)):
            for band,radius in (('safe',1.+safe_margin),('intrusion',intrusion_radius)):
                p=center+axis*radius
                tangent=yaw_to(center,p)+math.pi/2
                provenance={'source':'human','track_id':h['track_id'],'direction':direction,'intended_band':band,
                    'nominal_radius_m':radius,'psc_radius_m':1.,'safe_margin_m':safe_margin,
                    'reference_timestamp':now,'last_seen_time':h['last_seen_time'],
                    'track_age_s':h.get('observation_age',0.),'estimated_velocity':velocity.tolist(),
                    'confidence':1. if h.get('visible_now') else max(0.,1.-h.get('observation_age',0.)/2.),
                    'heading_basis':'observed_motion' if reliable else 'robot_relative_fallback',
                    'reference_position':center.tolist(),'frozen_coordinates':True}
                propose(p,provenance,[tangent])
    frontiers=partial_graph.frontiers(pose[:3])
    for f in frontiers:
        boundary=np.asarray(f['position'])
        # The viewpoint is inside known space, far enough back for the entire
        # robot to face the unknown corridor without its rear/front entering it.
        feasible=[k for k in reachable if np.linalg.norm(partial_graph.center(k)-boundary)<1.2
                  and static_body_clearance(partial_graph,partial_graph.center(k),f['unknown_yaw'])]
        if not feasible:continue
        k=min(feasible,key=lambda k:(np.linalg.norm(partial_graph.center(k)-boundary),k))
        propose(partial_graph.center(k),{'source':'frontier','frontier_component_cells':f['component_cells'],
            'cell':list(k),'boundary_cell':f['cell'],'unknown_yaw':f['unknown_yaw']},[f['unknown_yaw']])
    if partial_graph.point_free(goal) and partial_graph.key(goal) in reachable:
        propose(goal,{'source':'goal','actual_goal_known_connected':True})
    elif reachable:
        # Distance is a heuristic only; reachability was already established.
        keys=sorted(reachable,key=lambda k:(np.linalg.norm((partial_graph.center(k)-goal)[[0,2]]),k))
        for k in keys:
            if (np.linalg.norm(partial_graph.center(k)-pose[:3])>.5 and
                    static_body_clearance(partial_graph,partial_graph.center(k),yaw_to(pose[:3],partial_graph.center(k)))):
                if all(np.linalg.norm(partial_graph.center(k)-c['position'])>.35 for c in spatial):
                    if propose(partial_graph.center(k),{'source':'goal','actual_goal_known_connected':False,'cell':list(k)}):break
    topology=partial_graph.rebuild_global(pose[:3])
    # Retain a useful local graph stopping point behind the robot even when
    # a sparse open-room skeleton has no nearby global vertex there.
    behind=[]
    for k in reachable:
        p=partial_graph.center(k);d=p-pose[:3]
        progress=float(d[0]*math.cos(pose[3])-d[2]*math.sin(pose[3]))
        distance=float(np.linalg.norm(d[[0,2]]))
        if -.3>progress and .35<distance<2.5 and static_body_clearance(partial_graph,p,yaw_to(pose[:3],p)):
            lateral=abs(d[0]*math.sin(pose[3])+d[2]*math.cos(pose[3]))
            behind.append((lateral+abs(distance-.8),k))
    if behind:
        k=min(behind)[1]
        propose(partial_graph.center(k),{'source':'topology','node_id':f'local_{k[0]}_{k[1]}',
            'kind':'observed_backtrack_support','behind_robot':True})
    for n in sorted(topology['nodes'],key=lambda n:(n['kind']!='historical_decision',n['id'])):
        if np.linalg.norm(np.asarray(n['position'])-pose[:3])<.4:continue
        d=np.asarray(n['position'])-pose[:3]
        propose(n['position'],{'source':'topology','node_id':n['id'],'kind':n['kind'],
                               'behind_robot':float(d[0]*math.cos(pose[3])-d[2]*math.sin(pose[3]))<0})
        if sum(c['provenance']['source']=='topology' for c in spatial)>=8:break
    queries=[];seen=set();proposal_count=0
    # Generate a bounded but diverse orientation set. Human tangent orientation
    # makes physically clear lateral intrusion possible with the audited hull.
    for c in spatial:
        p=np.asarray(c['position']);source=c['provenance']['source']
        yaws=list(c['extra_yaws'])
        if source=='current_position':yaws=yaws[:5]
        else:
            yaws.append(yaw_to(pose[:3],p))
            yaws.append(yaw_to(p,goal))
            if source=='human':yaws.append(yaw_to(p,c['provenance']['reference_position']))
            if not yaws:yaws=[pose[3]]
        unique=[]
        for yaw in yaws:
            yaw=wrap(yaw)
            if any(abs(wrap(yaw-v))<.18 for v in unique):continue
            unique.append(yaw)
            if len(unique)>= (5 if source=='current_position' else 2):break
        for yaw in unique:
            proposal_count+=1
            if not physical_clearance(p,yaw,human_tracks):
                rejections.append({'position':p.tolist(),'final_yaw':yaw,'provenance':c['provenance'],
                                   'reason':'physical_body_envelopes_overlap_at_proposal'});continue
            q={'waypoints':[p.tolist()],'final_yaw':yaw,'provenance':[copy.deepcopy(c['provenance'])],
               'decision_timestamp':now,'static_traversability':'known_free_reachable',
               'query_semantics':'ordered_waypoints_plus_terminal_yaw','executor_version':EXECUTOR_VERSION}
            result=compile_query(partial_graph,q,ego_state)
            if not result['ok']:
                rejections.append({'position':p.tolist(),'final_yaw':yaw,'provenance':c['provenance'],
                                   'reason':result['reason']});continue
            signature=(tuple(np.round(p,4)),round(yaw,3))
            if signature in seen:continue
            seen.add(signature);q['compiled']=result;q['route_distance']=result['route_distance']
            q['query_id']=digest(q);queries.append(q)
    # Round-robin source/band stratification, deterministic and fully recorded.
    groups={}
    for q in queries:
        p=q['provenance'][0];k=p['source']+':'+p.get('intended_band','')
        groups.setdefault(k,[]).append(q)
    selected=[];base_limit=max(1,limit-2) if len(queries)>8 else limit
    while len(selected)<base_limit and any(groups.values()):
        for key in sorted(groups):
            quota=2 if key.startswith('human:') else 1
            for _ in range(quota):
                if groups[key] and len(selected)<base_limit:selected.append(groups[key].pop(0))
    # A small number of composed queries use the same two-point semantics.
    if len(selected)<limit:
        anchors=[q for q in queries if q['provenance'][0]['source'] in ('topology','goal')]
        for a,b in zip(anchors,anchors[1:]):
            if np.linalg.norm(np.asarray(a['waypoints'][0])-b['waypoints'][0])<.5:continue
            q={k:copy.deepcopy(v) for k,v in b.items() if k not in ('query_id','compiled','route_distance')}
            q['waypoints']=[a['waypoints'][0],b['waypoints'][0]]
            q['provenance']=[a['provenance'][0],b['provenance'][0]]
            compiled=compile_query(partial_graph,q,ego_state)
            if compiled['ok']:
                q['compiled']=compiled;q['route_distance']=compiled['route_distance'];q['query_id']=digest(q);selected.append(q)
            if len(selected)>=limit:break
    while len(selected)<limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected)<limit:selected.append(groups[key].pop(0))
    return selected,{'spatial_candidates':spatial,'rejected':rejections,'orientation_proposals':proposal_count,
        'geometrically_accepted_queries':len(queries),'selected_queries':len(selected),
        'selection':'deterministic_round_robin_human_quota2_other_quota1_with_two_point_reserve_v1',
        'accepted_by_category':{k:sum(q['provenance'][0]['source']+':'+q['provenance'][0].get('intended_band','')==k for q in queries) for k in groups},
        'policy_uses_future':False}
