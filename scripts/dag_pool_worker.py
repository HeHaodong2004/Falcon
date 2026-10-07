"""IPC execution worker; sampling and durable completion belong to the coordinator."""
import argparse,gc,hashlib,json,os,sys,time,traceback,copy
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
T=time.perf_counter()
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--mode',choices=['OLD','NO_VISUAL','NO_RENDERER','COMBO','COMBO_NO_RENDERER'],required=True);p.add_argument('--max-jobs',type=int,default=40);p.add_argument('--checkpoint',action='store_true');p.add_argument('--exact-root-gate',action='store_true');p.add_argument('--collection-output',type=Path);a=p.parse_args()
B=Path('/home/haodong/workspace/projects/eventwm-falcon');F=B/'output/overnight_collection_checkout';O=B/'output/execution_dag_exploration'
sys.path[:0]=[a.source,str(Path(a.source)/'scripts')]
from collector import runtime as rt
import controlled.run_case as rc
import controlled_matrix as cm
rt.REPO=rc.REPO=cm.REPO=Path(a.source).resolve() if a.collection_output else F
if a.collection_output:
 a.collection_output=a.collection_output.resolve()
 if not (a.mode=='COMBO_NO_RENDERER' and a.checkpoint and a.exact_root_gate):
  raise RuntimeError('Production workers require the validated COMBINED exact-gate configuration')
from collector import root_sampling,v3
import random,numpy as np,torch
IMPORTS=time.perf_counter()-T
metrics=defaultdict(float);counts=defaultdict(int);current={};phase=['prefix']
if a.mode in ['NO_VISUAL','NO_RENDERER','COMBO_NO_RENDERER']:
 from nonvisual_runtime import install
 install(rt,a.mode,metrics)
def wrap(obj,name,key):
 fn=getattr(obj,name)
 def call(*args,**kw):
  t=time.perf_counter()
  try:return fn(*args,**kw)
  finally:metrics[key]+=time.perf_counter()-t;counts[key]+=1
 setattr(obj,name,call)
wrap(rt,'build_config','config');wrap(rt.habitat.Env,'__init__','env_construct');wrap(rt.habitat.Env,'reset','reset');wrap(rt.habitat_sim.Simulator,'__init__','native_construct_subset');wrap(v3,'generate_queries_v3','query_generation');wrap(rt.habitat_sim.Simulator,'get_sensor_observations','sensor_render_subset');wrap(rt,'write','runtime_write_subset')
from collector import root_preflight
original_exact_gate=root_preflight.run_root_preflight
def exact_gate(*args,**kwargs):
 t=time.perf_counter()
 metrics['actual_root_available_seconds']=t-current['job_begin']
 try:return original_exact_gate(*args,**kwargs)
 finally:metrics['exact_gate_including_io']+=time.perf_counter()-t;counts['exact_gate_calls']+=1
root_preflight.run_root_preflight=exact_gate
from falcon.rearrange_sim_v2 import RearrangeSim_v2
wrap(RearrangeSim_v2,'reconfigure','scene_robot_reconfigure_subset')
original_prefix=root_sampling.prefix_commands
def prefix(*args,**kw):
 t=time.perf_counter()
 try:yield from original_prefix(*args,**kw)
 finally:metrics['prefix_inclusive']+=time.perf_counter()-t
root_sampling.prefix_commands=prefix
original_step=rt.habitat.Env.step
def step(*args,**kw):
 t=time.perf_counter()
 try:return original_step(*args,**kw)
 finally:metrics[phase[0]+'_env_step_subset']+=time.perf_counter()-t
rt.habitat.Env.step=step
original_rec=rt.Recorder.__init__;original_capture=rt.Recorder.capture
def rec_init(self,env,count):
 current['sim']=env.sim;current['step_world']=env.sim.step_world;current['rec']=self
 # Passive native-configuration evidence, outside all scientific hashes.
 try:
  stage=env.sim.get_stage_initialization_template()
  current['native_collision_config']=dict(stage_margin_m=float(stage.margin),stage_collision_asset=str(stage.collision_asset_handle),articulated_margin_exposed=hasattr(env.sim.get_agent_data(0).articulated_agent.sim_obj.creation_attributes,'margin'))
 except AttributeError as exc:current['native_collision_config']=dict(not_exposed=str(exc))
 original_rec(self,env,count)
rt.Recorder.__init__=rec_init
def capture(self,kind,*args,**kw):
 if kind=='root':
  current['root_captures']+=1
  if current['root_captures']==2:phase[0]='rollout'
 return original_capture(self,kind,*args,**kw)
rt.Recorder.capture=capture
original_legal=rt.legal_observation
def legal(*args,**kw):
 result=original_legal(*args,**kw)
 current['legal'].append(result[0])
 return result
rt.legal_observation=legal
cache=None
if a.mode in ['COMBO','COMBO_NO_RENDERER']:
 from fresh_world_cache import FreshWorldCache,install_config_cache
 cache=FreshWorldCache(rt,metrics,max_jobs=a.max_jobs,max_scenes=3)
 install_config_cache(metrics)
if a.checkpoint:
 from root_checkpoint import install_checkpoint
 install_checkpoint(rt,current,metrics,counts,phase)
original_close=rt.habitat.Env.close
def close(env):
 t=time.perf_counter()
 # Lightweight exact terminal state for failed and accepted jobs; no native stepping.
 current['terminal']=dict(robot=rt.state(env.sim.get_agent_data(0).articulated_agent),humans=[rt.state(env.sim.get_agent_data(i).articulated_agent) for i in range(1,7)],controllers=[rt.controller_state(env.task.actions[f'agent_{i}_oracle_nav_randcoord_action_obstacle']) for i in range(1,7)],sim_time=float(env.sim.get_world_time()),task_step=env.task._cur_episode_step,elapsed=env._elapsed_steps,episode_over=env.episode_over,should_end=env.task.should_end,rng=hashlib.sha256(repr((random.getstate(),np.random.get_state(),torch.get_rng_state().tolist())).encode()).hexdigest())
 if current.get('sim') is env.sim:env.sim.step_world=current['step_world']
 metrics['diagnostic_terminal_capture']=time.perf_counter()-t
 if cache is None:
  t=time.perf_counter();original_close(env);metrics['teardown']+=time.perf_counter()-t
rt.habitat.Env.close=close
def h(path):return hashlib.sha256(path.read_bytes()).hexdigest()
print('__READY__'+json.dumps(dict(imports=IMPORTS,pid=os.getpid())),flush=True)
for index,line in enumerate(sys.stdin):
 if not line.strip():continue
 if not a.collection_output:assert index<2048,'bounded worker job cap'
 job=json.loads(line);out=Path(job['output']).resolve()
 assert (a.collection_output if a.collection_output else O) in out.parents
 current.clear();current.update(root_captures=0,legal=[],job=job,collection_output=str(a.collection_output) if a.collection_output else None);metrics.clear();counts.clear();phase[0]='prefix'
 args=SimpleNamespace(output=out,worker_id=job['worker_id'],query_id=job.get('query_id'),limit_steps=job.get('limit_steps'),protocol='v3',storage_profile='production',exact_root_preflight=a.exact_root_gate)
 start=time.perf_counter();cpu=time.process_time();error=None
 current['job_begin']=start
 profiler=None
 if job.get('profile'):
  import cProfile
  profiler=cProfile.Profile();profiler.enable()
 try:rt.worker(args)
 except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc));traceback.print_exc()
 finally:
  if profiler:
   profiler.disable();profiler.dump_stats(str(out/(job['worker_id']+'.pstats')))
 elapsed=time.perf_counter()-current.get('branch_start',start);cpu=time.process_time()-current.get('branch_cpu_start',cpu)
 out=Path(job['output']).resolve()
 terminal=dict(error=error,state=current.get('terminal'),legal=current['legal'])
 dest=out/'diagnostic_jobs'/job['worker_id'];dest.mkdir(parents=True,exist_ok=True)
 rt.write(dest/'terminal.json',terminal)
 artifact_paths=['roots/replay.json','roots/legal_root.json','roots/privileged_root.json','legal_observations/history.json','queries/queries.json','audits/generation_v2.json','audits/root_sampling.json','audits/natural_prefix.json',f'audits/workers/{job["worker_id"]}/root_fingerprint.json',f'audits/workers/{job["worker_id"]}/branch.json',f'audits/workers/{job["worker_id"]}/observation_diagnostics.json']
 hashes={r:h(out/r) for r in artifact_paths if (out/r).exists()}
 hashes['diagnostic_terminal']=h(dest/'terminal.json')
 label=None
 if args.query_id and (out/f'audits/workers/{job["worker_id"]}/branch.json').exists():
  branch=rt.read(out/f'audits/workers/{job["worker_id"]}/branch.json');q=next(x['query'] for x in rt.read(out/'queries/queries.json') if x['query']['query_id']==args.query_id)
  label=v3.social_outcome_labels(branch,q)
 rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))
 result=dict(job=job,error=error,wall_seconds=elapsed,cpu_seconds=cpu,metrics=dict(metrics),counts=dict(counts),hashes=hashes,label=label,rss_kib=rss,pid=os.getpid())
 result['native_collision_config']=current.get('native_collision_config')
 if current.get('checkpoint'):result['checkpoint']=current['checkpoint']
 if current.get('branch_fork_seconds') is not None:result['branch_fork_seconds']=current['branch_fork_seconds']
 from scientific_comparison import scientific_hashes
 result['scientific_hashes']=scientific_hashes(out,job['worker_id'])
 if current.get('rec') is not None:
  from nonvisual_runtime import digest,scientific_trace
  result['full_native_trace_sha256']=digest(current['rec'].rows)
  result['full_native_trace_scientific_sha256']=digest(scientific_trace(current['rec'].rows,current['legal']))
 rt.write(dest/'result.json',result)
 # Durable staged receipt before acknowledging this immutable logical job.
 # Does not modify a scientific artifact or perform a simulator operation.
 seal_start=time.perf_counter()
 attempt=out/'audits/workers'/job['worker_id']
 stable=[*dest.glob('*.json'),*attempt.glob('*')]
 if job['worker_id']=='root':
  for folder in ['roots','queries','legal_observations']:
   stable.extend((out/folder).glob('*'))
  stable.extend((out/'audits').glob('*.json'))
 for file in stable:
  if file.is_file():
   with file.open('rb') as stream:os.fsync(stream.fileno())
 for directory in {file.parent for file in stable if file.is_file()}:
  fd=os.open(directory,os.O_RDONLY);os.fsync(fd);os.close(fd)
 result['staging_seal_seconds']=time.perf_counter()-seal_start
 child_connection=current.get('branch_connection');is_child=current.get('is_branch_child',False)
 current.clear();gc.collect()
 if is_child:
  child_connection.send(dict(result=result));child_connection.close();os._exit(0 if error is None else 2)
 print('__RESULT__'+json.dumps(result),flush=True)
 if a.mode in ['OLD','NO_VISUAL','NO_RENDERER']:break
if cache:cache.close()
