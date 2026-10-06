"""Fresh-process deterministic replay, native FALCON humans, passive recorder."""
import copy
import hashlib
import json
import math
import random
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'scripts'), str(REPO/'scripts/controlled')]
import habitat_sim  # Must precede magnum/habitat imports on the validated build.
import habitat
from habitat.config import read_write
from omegaconf import OmegaConf, open_dict
from controlled.run_case import build_config
from collector.core import LegalRoot, Query, digest, generate_queries, legal_observation, read, write
from collector.observation import HabitatHumanObserver, ObservationConfig
from dataclasses import asdict


class StaticPlanner:
    def __init__(self, navmesh):
        # Separate immutable static mesh, never the live simulator/pathfinder.
        self.pf = habitat_sim.PathFinder()
        if not self.pf.load_nav_mesh(str(navmesh)):
            raise ValueError('Cannot load static navmesh')

    def path(self, start, target, cfg):
        end = np.array(self.pf.snap_point(target))
        if not np.isfinite(end).all() or np.linalg.norm(end-target) > cfg['max_projection']:
            return None
        p = habitat_sim.ShortestPath()
        p.requested_start, p.requested_end = start, end
        if not self.pf.find_path(p) or not .15 < p.geodesic_distance < cfg['max_path_length']:
            return None
        return [np.asarray(x).tolist() for x in p.points]


def array(x):
    return np.asarray(x).astype(float).tolist()


def state(agent):
    obj = agent.sim_obj
    forward = np.asarray(agent.base_transformation.transform_vector([1.,0.,0.]))
    return dict(position=array(agent.base_pos), yaw=math.atan2(-forward[2], forward[0]),
                transform=array(obj.transformation), joints=array(obj.joint_positions),
                joint_velocities=array(obj.joint_velocities),
                linear_velocity=array(obj.root_linear_velocity), angular_velocity=array(obj.root_angular_velocity),
                object_id=obj.object_id)


def controller_state(action):
    result = {}
    for k,v in vars(action).items():
        if k in ('goals','current_goal_idx','coord_nav','old_human_pos_list','rand_human_speed_scale','skill_done'):
            result[k] = clean(v)
    controller = action.humanoid_controller
    # Numeric controller internals include gait frame, transforms, timing.
    result['controller'] = {k: clean(v) for k,v in vars(controller).items()
                            if isinstance(v,(int,float,bool,np.ndarray)) or type(v).__module__=='magnum'}
    return result


def clean(v):
    if v is None or isinstance(v,(str,int,float,bool)):
        return v
    if isinstance(v,dict):
        return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):
        return [clean(x) for x in v]
    return np.asarray(v).tolist()


class Recorder:
    def __init__(self, env, count):
        self.env, self.sim, self.count = env, env.sim, count
        self.rows, self.command, self.index = [], [0.,0.], -1
        self.previous = None
        self.original = self.sim.step_world
        self.sim.step_world = self.step_world

    def step_world(self, dt):
        self.original(dt)
        self.capture('physics', float(dt))

    def capture(self, phase, requested_dt=0.):
        sim = self.sim
        sim.perform_discrete_collision_detection()
        robot = state(sim.get_agent_data(0).articulated_agent)
        humans = [state(sim.get_agent_data(i).articulated_agent) for i in range(1,self.count+1)]
        human_ids = {h['object_id'] for h in humans}
        contacts, nh, ns = [], False, False
        for cp in sim.get_physics_contact_points():
            row = {k: clean(getattr(cp,k)) for k in ('object_id_a','object_id_b','link_id_a','link_id_b',
                   'position_on_a_in_ws','position_on_b_in_ws','contact_normal_on_b_in_ws',
                   'contact_distance','normal_force','is_active')}
            contacts.append(row)
            ids = {cp.object_id_a, cp.object_id_b}
            if robot['object_id'] in ids and len(ids)>1:
                if ids & human_ids:
                    nh = True
                # Ground support is retained in contacts but excluded from obstacle
                # contact: nearly vertical normal at/below base foot level.
                elif not (abs(cp.contact_normal_on_b_in_ws[1]) > .8 and
                          max(cp.position_on_a_in_ws[1],cp.position_on_b_in_ws[1]) < robot['position'][1]+.12):
                    ns = True
        # Bullet contact enumeration order is not physical state. Preserve every
        # exact value and duplicate, but serialize this multiset deterministically.
        contacts.sort(key=lambda contact: json.dumps(contact, sort_keys=True, allow_nan=False))
        row = dict(timestamp=float(sim.get_world_time()), phase=phase, requested_physics_dt=requested_dt,
                   control_index=self.index, issued_command=self.command[:], robot=robot, humans=humans,
                   contacts=contacts, native_human_contact=nh, native_static_contact=ns)
        if self.previous is not None and row['timestamp'] > self.previous['timestamp']:
            dt = row['timestamp']-self.previous['timestamp']
            for obj,prev in zip([robot]+humans,[self.previous['robot']]+self.previous['humans']):
                obj['finite_difference_velocity'] = ((np.asarray(obj['position'])-prev['position'])/dt).tolist()
        self.rows.append(row)
        self.previous = row
        return row


def save_images(path, entries):
    if entries:
        np.savez_compressed(path, timestamps=np.array([t for t,_ in entries]),
            rgb=np.stack([im['agent_0_articulated_agent_jaw_rgb'] for _,im in entries]),
            depth=np.stack([im['agent_0_articulated_agent_jaw_depth'] for _,im in entries]))


def worker(args):
    from collector.v2 import guard_output
    if getattr(args, 'protocol', 'v2') == 'v3':
        from collector.v3 import guard_output
    out = guard_output(args.output)
    attempt = out/'audits'/'workers'/args.worker_id
    attempt.mkdir(parents=True,exist_ok=True)
    recipe = read(out/'roots/replay.json')
    from collector.v2 import PROTOCOL, guard_output
    from collector.v3 import PROTOCOL as V3_PROTOCOL
    v3 = recipe.get('protocol_version') == V3_PROTOCOL
    if v3:
        from collector.v3 import PROTOCOL, guard_output
    v2 = recipe.get('protocol_version') == PROTOCOL
    if v2: guard_output(out)
    source = REPO/'evaluation/controlled50'
    case = next(c for c in read(source/'cases.json') if c['tag']==recipe['case'])
    case = dict(case, seed=recipe.get('seed', case['seed']))
    seed = case['seed']
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed); torch.set_num_threads(1)
    cfg = build_config(source,case,attempt,2000,REPO/'pretrained_model/falcon_noaux_25.pth')
    with read_write(cfg), open_dict(cfg.habitat.task.actions):
        # Continuous native BaseVelAction permits exact precompiled slow/wait
        # controls. This is an intervention adapter, not the FALCON policy.
        action = copy.deepcopy(cfg.habitat.task.actions.agent_0_discrete_move_forward)
        action.type = 'BaseVelAction'
        action.ang_speed = 10.0
        cfg.habitat.task.actions.agent_0_collector_velocity = action
        cfg.habitat.task.measurements = {k:v for k,v in cfg.habitat.task.measurements.items()
            if k in ('did_multi_agents_collide','human_collision','num_steps')}
        cfg.habitat.task.lab_sensors = {}
        cfg.habitat.simulator.concur_render = False
    OmegaConf.save(cfg,attempt/'resolved_config.yaml',resolve=True)
    observation_config = ObservationConfig.from_depth_sensor(cfg.habitat.simulator.agents.agent_0.sim_sensors.jaw_depth_sensor)
    write(attempt/'observation_config.json',asdict(observation_config))
    with habitat.Env(config=cfg.habitat) as env:
        env.seed(seed)
        obs = env.reset()
        sim, task = env.sim, env.task
        observer = HabitatHumanObserver(sim,observation_config)
        if v3:
            from collector.social_habitat import make_social_recorder
            rec = make_social_recorder(env,case['humans'])
        else:
            rec = Recorder(env,case['humans'])
        robot_action = task.actions['agent_0_collector_velocity']
        humans = [f'agent_{i}_oracle_nav_randcoord_action_obstacle' for i in range(1,case['humans']+1)]
        action_names = ('agent_0_collector_velocity', *humans)
        def step(command, index):
            rec.command, rec.index = list(command), index
            vals = {'agent_0_base_vel': np.array([command[0]/robot_action._lin_speed,
                                                        command[1]/robot_action._ang_speed])}
            vals.update({k:np.array([1.]) for k in humans})
            return env.step({'action':action_names,'action_args':vals})
        def observe(obs, command):
            r = state(sim.get_agent_data(0).articulated_agent)
            tracks = observer.observe(sim.get_world_time(),case['humans'])
            return legal_observation(obs,r['position'],r['yaw'],sim.get_world_time(),command,tracks)
        history, images = [], []
        meta, im = observe(obs,[0.,0.]); history.append(meta); images.append((meta['timestamp'],im))
        rec.capture('initial')
        control_times = [float(sim.get_world_time())]
        # Optional bounded legal-history root selection; the resolved controls are
        # frozen into replay.json before any branch is launched.
        from collector.root_sampling import prefix_commands
        def natural_motion(selection):
            # No history/human/PSC inputs: known static mesh + current robot only.
            from collector.v3 import generate_queries_v3
            from collector.v2 import QueryConfig
            from collector.static_feasibility import StaticConfig
            from collector.static_habitat import HabitatStaticWorld
            scene=next(s for s in read(source/'scenes.json') if s['name']==case['scene'])
            planner=StaticPlanner(source/scene['navmesh'])
            robot=sim.get_agent_data(0).articulated_agent
            pose=state(robot)
            prefix_root=LegalRoot(recipe['root_id']+'/prefix',tuple(pose['position']),pose['yaw'],
                1/task._physics_target_sps+sim.ac_freq_ratio*sim.get_physics_time_step(),1/sim.ctrl_freq)
            world=HabitatStaticWorld(planner,sim,robot)
            choices,generation=generate_queries_v3(prefix_root,world,world.geometry,
                QueryConfig(movement_queries=selection['prefix_movement_queries']),StaticConfig(**recipe.get('static_config',{})))
            chosen=choices[4*selection['selected_prefix_path']].data
            if args.query_id is None:
                write(out/'audits/natural_prefix.json',dict(selection=selection,query=chosen,generation=generation))
            return chosen['executable_controls']
        natural=recipe.get('sampling',{}).get('profile')=='natural'
        if natural and args.query_id is None:
            from collector.natural_sampling import natural_selection
            write(out/'audits/sampling_selection.json',natural_selection(seed,recipe['sampling']))
        executed_prefix = []
        for i,command in enumerate(prefix_commands(recipe, history, natural_motion=natural_motion)):
            prefix_row_start=len(rec.rows)
            executed_prefix.append(list(command))
            obs = step(command,i)
            meta, im = observe(obs,command); history.append(meta); images.append((meta['timestamp'],im))
            control_times.append(float(sim.get_world_time()))
            prefix_static=(natural and i>=len(recipe['sampling'].get('prelude_controls',[])) and
                           any(r['native_static_contact'] for r in rec.rows[prefix_row_start:]))
            if prefix_static:
                write(out/'audits/STATIC_FEASIBILITY_VIOLATION.json',dict(static_feasibility_violation=True,
                    phase='accepted_natural_prefix',step=i,termination_reason='static_collision'))
                raise RuntimeError('STATIC_FEASIBILITY_VIOLATION in accepted natural prefix')
            if env.episode_over:
                if recipe.get('sampling_audit'):
                    write(out/'audits/prefix_failure.json',dict(phase='before_preselected_root',steps_executed=i+1,
                        timestamp=float(sim.get_world_time()),native_human_contact=any(r['native_human_contact'] for r in rec.rows),
                        native_static_contact=any(r['native_static_contact'] for r in rec.rows),
                        sampling=recipe.get('sampling'),selection_not_replaced=True))
                raise RuntimeError('Prefix terminated before root')
        physics_dt = sim.get_physics_time_step()
        dt = 1/task._physics_target_sps + sim.ac_freq_ratio*physics_dt
        write(attempt/'clock.json',dict(prefix_timestamps=control_times,
              nominal_control_dt=dt,physics_dt=physics_dt,
              integration_dt=1/sim.ctrl_freq,physics_target_sps=task._physics_target_sps))
        root_state = rec.capture('root')
        root_state.pop('requested_physics_dt')
        root_state['controllers'] = [controller_state(task.actions[k]) for k in humans]
        root_state['task_step'] = task._cur_episode_step
        root_state['rng_hash'] = hashlib.sha256(repr((random.getstate(),np.random.get_state(),
                                                        torch.get_rng_state().tolist())).encode()).hexdigest()
        root_state['history_hash'] = digest(history)
        root_state['image_hash'] = hashlib.sha256(b''.join(im[k].tobytes() for _,im in images for k in sorted(im))).hexdigest()
        # Store controller artifacts for audit, not as a claim of full snapshot API.
        fingerprint = digest(root_state)
        write(attempt/'root_fingerprint.json',dict(sha256=fingerprint,state=root_state))
        write(attempt/'observation_diagnostics.json',observer.diagnostics)
        if args.query_id is None:
            if recipe.get('sampling_audit'):
                from collector.sampling_audit import root_diagnostics
                write(out/'audits/root_sampling.json',root_diagnostics(recipe,root_state,history,len(executed_prefix)))
            if 'sampling' in recipe and 'prefix_controls' not in recipe:
                recipe['prefix_controls'] = executed_prefix
                write(out/'roots/replay.json', recipe)
            root_id = recipe.get('root_id', f"root_{case['scene']}_h{case['humans']}_" + fingerprint[:10])
            legal = dict(root_id=root_id,robot_position=root_state['robot']['position'],
                         robot_yaw=root_state['robot']['yaw'],control_dt=dt,integration_dt=1/sim.ctrl_freq,
                         timestamp=root_state['timestamp'])
            write(out/'roots/legal_root.json',legal)
            write(out/'legal_observations/observation_config.json',asdict(observation_config))
            write(out/'roots/privileged_root.json',dict(sha256=fingerprint,state=root_state))
            write(out/'legal_observations/history.json',history)
            if not v3:
                save_images(out/'legal_observations/history_images.npz',images)
            else:
                write(out/'legal_observations/image_storage.json',dict(retained=False,
                    reason='structured input uses no pixels; image hash still verified in replay'))
            write(out/'roots/prefix_truth.json',rec.rows)
            scene = next(s for s in read(source/'scenes.json') if s['name']==case['scene'])
            planner = StaticPlanner(source/scene['navmesh'])
            if v2:
                from collector.v2 import generate_queries_v2, QueryConfig, InsufficientStaticQueries
                if v3:
                    from collector.v3 import generate_queries_v3 as generate_queries_v2
                from collector.static_feasibility import StaticConfig
                from collector.static_habitat import HabitatStaticWorld
                static_world = HabitatStaticWorld(planner,sim,sim.get_agent_data(0).articulated_agent)
                try:
                    queries,generation = generate_queries_v2(LegalRoot.load(legal),static_world,static_world.geometry,
                        QueryConfig(**recipe.get('query_config',{})),StaticConfig(**recipe.get('static_config',{})))
                except InsufficientStaticQueries as exc:
                    write(out/'audits/generation_v2.json',exc.diagnostics)
                    raise
                generation['static_world']=dict(stage_id=int(habitat_sim.stage_id),static_object_ids=sorted(static_world.static_ids))
                write(out/'audits/generation_v2.json',generation)
                if v3 and recipe.get('debug_static_reference',False):
                    from collector.debug_static import capture_static_reference
                    capture_static_reference(static_world,legal['robot_position'],out/'audits/debug_static_reference.json')
                rejected=generation['candidate_rejections']
            else:
                queries,rejected = generate_queries(LegalRoot.load(legal),planner)
            write(out/'queries/queries.json',[dict(sha256=q.sha256,query=q.data) for q in queries])
            write(out/'audits/generation.json',dict(rejected=rejected,generated=len(queries),
                  static_navmesh_sha256=hashlib.sha256((source/scene['navmesh']).read_bytes()).hexdigest()))
            print('ROOT',root_id,'QUERIES',len(queries),'DT',dt,flush=True)
            return
        expected = read(out/'roots/privileged_root.json')['sha256']
        if fingerprint != expected:
            raise RuntimeError(f'Root replay mismatch: {fingerprint} != {expected}')
        saved = next(q for q in read(out/'queries/queries.json') if q['query']['query_id']==args.query_id)
        query = Query.make(saved['query']); before = query.sha256
        assert before == saved['sha256']
        q = query.data
        if v2:
            from collector.v2 import PROTOCOL
            if v3:
                from collector.v3 import PROTOCOL
            from collector.static_feasibility import CHECKER_VERSION
            if q.get('protocol_version') != PROTOCOL or not q.get('static_precheck',{}).get('static_feasible') or q['static_precheck']['checker_version'] != CHECKER_VERSION:
                raise RuntimeError('Missing/incompatible static feasibility certificate')
        rec.rows = []; rec.previous = None; rec.index = -1; rec.command=[0.,0.]
        rec.capture('root')
        legal_rows, branch_images, controls = [], [], []
        meta, im = observe(obs,[0.,0.]); legal_rows.append(meta); branch_images.append((meta['timestamp'],im))
        reason, completed = 'query_completed', True
        for i,command in enumerate(q['executable_controls']):
            if args.limit_steps is not None and i >= args.limit_steps:
                reason, completed = 'timeout', False
                break
            t0 = float(sim.get_world_time()); n0 = len(rec.rows)
            obs = step(command,i)
            t1 = float(sim.get_world_time())
            controls.append(dict(index=i,t_start=t0,t_end=t1,nominal_command=command,issued_command=rec.command[:]))
            meta, im = observe(obs,command); legal_rows.append(meta); branch_images.append((meta['timestamp'],im))
            if any(r['native_human_contact'] for r in rec.rows[n0:]):
                reason,completed = 'robot_human_collision', False; break
            if any(r['native_static_contact'] for r in rec.rows[n0:]):
                reason,completed = 'static_collision', False; break
            if env.episode_over or task.should_end:
                reason,completed = 'environment_termination', False; break
        endpoint_error = float(np.linalg.norm((np.array(rec.rows[-1]['robot']['position'])-
                                               q['nominal_timed_trajectory'][-1][1:4])[[0,2]]))
        if completed and endpoint_error > .3:
            reason, completed = 'execution_failure', False
        assert query.sha256 == before == digest(q)
        result = dict(branch_id=args.worker_id,query_id=q['query_id'],root_id=q['root_id'],
                      query_hash_before=before,query_hash_after=query.sha256,root_fingerprint=fingerprint,
                      planned_duration=q['planned_duration'],observed_duration=rec.rows[-1]['timestamp']-rec.rows[0]['timestamp'],
                      completed=completed,termination_reason=reason,endpoint_error=endpoint_error,
                      truth=rec.rows,controls=controls,legal_observations=legal_rows)
        write(attempt/'branch.json',result)
        write(attempt/'observation_diagnostics.json',observer.diagnostics)
        if not v3 and getattr(args,'storage_profile','validated')!='production':
            save_images(attempt/'branch_images.npz',branch_images)
        print('BRANCH',args.worker_id,reason,'seconds',result['observed_duration'],'error',endpoint_error,flush=True)
