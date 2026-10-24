"""Large-scale, collection-only partial-graph roots and native counterfactuals."""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from ..config import REPO,digest,read_json,file_digest
from ..stage1.runtime import NativeEpisode,episode_record
from .map import PartialGraph
from .observer import GeometryObserver
from .query import generate_queries,HUMAN_SPEED_BOUND,wrap
from .initialization import make_episode
from .metrics import step_labels
from .execution import GraphPrefixFuture
from .storage import seal_root,implementation


def verify_gate(path,stage='stage0'):
    gate=read_json(path)
    required={'stage0':{'observation_causality','graph_correctness','query_correctness','determinism','real_Habitat_demonstrations'},
              'stage1':{'native_replay','official_psc','legal_reconstruction','branch_reproducibility','storage_integrity','independent_review'}}[stage]
    if (gate.get('stage')!=stage or gate.get('status')!='PASS' or not required.issubset(gate.get('checks',{}))
            or not all(value is True for value in gate['checks'].values())):
        raise ValueError(stage+'_acceptance_gate_closed')
    sources=implementation()
    core={'map.py','observer.py','query.py','initialization.py'}
    if stage=='stage1':core|={'metrics.py','execution.py','storage.py','collect.py'}
    if not core.issubset(gate.get('implementation',{})):raise ValueError('gate_missing_source_evidence')
    for name,sha in gate.get('implementation',{}).items():
        actual=sources.get(name)
        if actual is None and '/' in name:
            path=Path(__file__).resolve().parents[2]/name
            if not path.exists():path=REPO/name
            if path.exists():actual=file_digest(path)
        if actual is not None and actual!=sha:raise ValueError(stage+'_validated_source_changed:'+name)
    return gate


def check_motion_bound(previous,current,dt):
    if not previous:return
    p=np.asarray([h['position'] for h in previous]);q=np.asarray([h['position'] for h in current])
    speed=np.linalg.norm((q-p)[:,[0,2]],axis=1)/dt
    if speed.max(initial=0.)>HUMAN_SPEED_BOUND+1e-6:raise ValueError('declared_native_human_motion_bound_exceeded')


def collect_root(h,output,seed,stage0_gate,queries_per_root=24,max_query_steps=120,
                 node_index=None,human_count=None,scenario=None,private_cache=None,pilot=False,
                 safe_margin=.2,intrusion_radius=.96):
    verify_gate(stage0_gate)
    began=time.perf_counter();perf={}
    prior_overrides={'node_index':node_index,'human_count':human_count,'scenario':scenario}
    rng=np.random.default_rng(int(digest({'seed':int(seed),'domain':'partial_root_prior_v1'})[:16],16))
    starts=h.valid_starts()
    node=starts[int(rng.integers(len(starts))) if node_index is None else node_index%len(starts)]['node_id']
    count=int(rng.choice([0,1,2,3,4,5,6],p=[.08,.24,.22,.18,.12,.09,.07])) if human_count is None else int(human_count)
    if scenario is None:scenario=str(rng.choice(['natural','approach','crossing'],p=[.35,.35,.30]))
    episode,sampling=make_episode(h,node,seed,count,scenario)
    init=time.perf_counter()
    with NativeEpisode(h,episode,seed) as rt:
        perf['scene_initialization_seconds']=time.perf_counter()-init
        sensor=GeometryObserver(rt,h.robot_config.collision_reference/6,private_cache)
        rt.observation=sensor.observe
        row=sensor.observe();g=PartialGraph(sensor.resolution,row['robot_pose'][1]);g.update(row)
        if abs(wrap(row['robot_pose'][3]-sampling['root_heading']))>1e-5:raise ValueError('initial_heading_contract_mismatch')
        history=[row];labels=[];prefix=[];truth=rt.truth();last_time=row['timestamp']
        # Prefix is actual primitive execution; all maps start empty. It is
        # recorded and separately replayable. No private route directs actions.
        prefix_plan=['forward']*15+['left']*19+['forward']*5+['right']*38+['forward']*4
        t=time.perf_counter();proposals=[];diagnostics={}
        for i,action in enumerate(prefix_plan,1):
            if rt.env.episode_over:raise ValueError('prefix_human_collision')
            row=rt.step(action);prefix.append(action);g.update(row);history.append(row)
            now_truth=rt.truth();check_motion_bound(truth,now_truth,row['timestamp']-last_time)
            truth,last_time=now_truth,row['timestamp'];labels.append(step_labels(rt,row['timestamp']))
            if rt.env.episode_over:raise ValueError('prefix_human_collision')
            if i in (11,12,13,14,15,16,17,18,24,34,39,58,len(prefix_plan)):
                generation=time.perf_counter()
                proposals,diagnostics=generate_queries(g,row['tracks'],row,episode.goals[0].position,queries_per_root,
                    safe_margin=safe_margin,intrusion_radius=intrusion_radius)
                perf['query_generation_seconds']=perf.get('query_generation_seconds',0.)+time.perf_counter()-generation
                # Legal-history stopping rule, independent of future outcomes.
                spatial=sum(q['route_distance']>.25 for q in proposals)
                social=sum(q['provenance'][0]['source']=='human' for q in proposals)
                diverse=len(proposals)>=6 and spatial>=3 and (social>=2 or scenario=='natural' or i>=39)
                compact=i>=13 and len(proposals)>=3 and spatial>=1
                if diverse or compact:break
        perf['legal_prefix_seconds']=time.perf_counter()-t
        if len(proposals)<3 or not any(q['route_distance']>.25 for q in proposals):
            raise ValueError('insufficient_legal_compilable_queries')
        root_snapshot=g.snapshot()
        root_input={'history':history,'history_actions':prefix,'partial_graph':root_snapshot,'pointgoal':list(episode.goals[0].position),
                    'sensor':asdict(rt.observation_config)}
        tape=GraphPrefixFuture(rt,sensor,g,episode.goals[0].position,max_query_steps,
                              sum(not l['psc_violation'] for l in labels))
        t=time.perf_counter();horizon=min(max_query_steps,max(len(q['compiled']['actions']) for q in proposals))
        tape.generate(horizon)
        for previous,current,ta,tb in zip(tape.truth,tape.truth[1:],tape.times,tape.times[1:]):
            check_motion_bound(previous,current,tb-ta)
        perf['shared_human_future_seconds']=time.perf_counter()-t
        t=time.perf_counter();outcomes=tape.evaluate_many(proposals)
        perf['query_execution_seconds']=time.perf_counter()-t;perf.update(tape.performance)
        # Native motion is authoritative. Reject unsupported numerical roots,
        # preserving diagnostics in the caller, rather than relabeling plans.
        maximum_error=0.;maximum_planar=0.;maximum_height=0.
        for q,o in zip(proposals,outcomes):
            expected=np.asarray(q['compiled']['planned_poses'])[1:len(o['node_ids'])+1,:3]
            actual=np.asarray([tape.nodes[i]['observation']['robot_pose'][:3] for i in o['node_ids']])
            if len(actual):
                difference=actual-expected
                maximum_error=max(maximum_error,float(np.linalg.norm(difference,axis=1).max()))
                maximum_planar=max(maximum_planar,float(np.linalg.norm(difference[:,[0,2]],axis=1).max()))
                maximum_height=max(maximum_height,float(np.abs(difference[:,1]).max()))
            if o['summary']['query_complete'] and np.linalg.norm(np.asarray(o['successor']['robot_pose'][:3])-q['waypoints'][-1])>q['compiled']['position_tolerance']+.001:
                raise ValueError('native_terminal_position_tolerance_exceeded')
        if maximum_planar>.001:raise ValueError('native_compiled_planar_disagreement_above_1mm')
        if maximum_height>.080001:raise ValueError('native_compiled_height_disagreement_above_8cm')
        perf['maximum_native_position_error_m']=maximum_error
        perf['maximum_native_planar_error_m']=maximum_planar
        perf['maximum_native_height_error_m']=maximum_height
        perf['sensor']=dict(sensor.performance);perf['prestorage_seconds']=time.perf_counter()-began
        config={'schema':'partial_collection_config_1','queries_per_root':queries_per_root,
                'max_query_steps':max_query_steps,'human_horizontal_motion_bound_m_s':HUMAN_SPEED_BOUND,
                'safe_margin_m':safe_margin,'intrusion_radius_m':intrusion_radius,
                'pilot':bool(pilot),'automatic_training':False}
        record={**root_input,'scene_id':h.metadata['scene_id'],'seed':int(seed),
                'private_initialization':episode_record(episode),'source_map_manifest':str(h.manifest_path),
                'prefix_labels':labels,'sampling':{**sampling,'prefix_actions':prefix,
                    'prior_overrides':prior_overrides,'start_law':'uniform certified valid-start index before legal-history admission',
                    'crowd_prior_probabilities':[.08,.24,.22,.18,.12,.09,.07],
                    'scenario_prior_probabilities':{'natural':.35,'approach':.35,'crossing':.30},
                    'root_stop_rule':'first legal checkpoint: >=6 queries/3 spatial and (2 social or natural or step>=39), or step>=13 with >=3 queries/1 spatial',
                    'query_diagnostics':diagnostics,'future_draws':1,
                    'candidate_selection':'deterministic round robin by source/intended band',
                    'prefix_rejections_recorded':True,'offline_future_selection':False},
                'performance':perf,'collection_config':config}
        result=seal_root(output,record,tape,proposals,outcomes)
        result['wall_seconds']=time.perf_counter()-began
        return result
