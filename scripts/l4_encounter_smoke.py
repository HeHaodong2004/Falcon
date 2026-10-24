import argparse,json,time
from pathlib import Path
import habitat_sim
from habitat.tasks.nav.nav import NavigationEpisode,NavigationGoal
from eventwm_graph.benchmark.map import load_certified_map
from eventwm_graph.stage1.runtime import NativeEpisode
from eventwm_graph.partial.observer import GeometryObserver
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.partial.metrics import contact_labels
from eventwm_graph.l2_intermittent.model import IntermittentWorldModel
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output;out.mkdir(parents=True,exist_ok=False)
p=Path('docs/l4/social_executor_v1_20261022/native_scenarios.json')
c=next(c for c in json.loads(p.read_text())['cases'] if c['case_id']=='scenario_head_on')
h=load_certified_map(c['manifest']); e=c['original_episode']
ep=NavigationEpisode(episode_id=e['episode_id'],scene_id=e['scene_id'],start_position=e['start_position'],start_rotation=e['start_rotation'],goals=[NavigationGoal(**v) for v in e['goals']],info=e['info'])
history=[]; transitions=[]
with NativeEpisode(h,ep,c['seed']) as rt:
 s=GeometryObserver(rt,h.robot_config.collision_reference/6); ob=s.observe(); history.append(ob)
 g=PartialGraph(s.resolution,ob['robot_pose'][1]);g.update(ob)
 q=h.model.compile([ob['robot_pose'][:3]],start_yaw=ob['robot_pose'][3],target_yaw=ob['robot_pose'][3]+.25)
 for a in q['actions']+['right']*3+['forward']*2:
  before=ob;rt.step(a);ob=s.observe();g.update(ob);history.append(ob)
  transitions.append(dict(action=a,before=before['robot_pose'],after=ob['robot_pose'],dt=ob['timestamp']-before['timestamp'],contacts=contact_labels(rt)))
 ck='/research-user/workspace/projects/eventwm-falcon-l2-intermittent-500k-20261022/output/l2_intermittent_500k_20261022/training/checkpoint.pt'
 m=IntermittentWorldModel.load(ck);b=m.initialize(history,g.snapshot());start=time.perf_counter();f=m.predict_shared(b,1.92,16,221);latency=time.perf_counter()-start
 r=dict(case=c,query=q,history=history,graph=g.snapshot(),transitions=transitions,model_version=m.model_version,belief_time=b.timestamp,future_times=f.times.tolist(),future_shape=list(f.positions.shape),future_positions=f.positions.tolist(),forecast_seconds=latency)
 rt.step('stop');r['stop_terminated']=bool(rt.env.episode_over)
 r['visible_measurements']=sum(t['visible_now'] for row in history for t in row['tracks'])
 assert r['visible_measurements']>0 and f.positions.shape[2]>0
 (out/'encounter_smoke.json').write_text(json.dumps(r,indent=2))
 print({k:v for k,v in r.items() if k not in ('case','history','graph','transitions','future_positions','query','future_times')})
