import json,sys,os,collections
from pathlib import Path
from types import SimpleNamespace
code=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2]).resolve()
sys.path[:0]=[str(code),str(code/'scripts')]
from collector import runtime as rt
import controlled.run_case as rc
from eventwm_pipeline_io import install_indexed_reads
from nonvisual_runtime import install
install(rt,'COMBO_NO_RENDERER',collections.defaultdict(float))
install_indexed_reads(rt,rc,{},out)
from collector.static_habitat import HabitatStaticWorld
from collector.static_feasibility import is_static_feasible,StaticConfig
from collector.core import LegalRoot
import magnum as mn
import numpy as np
b=Path('/tmp/eventwm_v4_300k/roots/v3m_e7c6987aa64211d2f63c/attempt_01')
q=rt.read(b/'queries/queries.json')[27]['query']
branch=rt.read(b/'audits/workers/b027/branch.json')
failed=next(s for s in branch['truth'] if s['native_static_contact'])
scene=next(s for s in json.load(open(out/'catalogue/scenes.json')) if s['name']=='hm3d_KJxdMPgweZG')
def hook(args,attempt,env,root_state,history,times):
    pid=os.fork()
    if pid: os.waitpid(pid,0); return attempt
    sim=env.sim; robot=sim.get_agent_data(0).articulated_agent
    world=HabitatStaticWorld(rt.StaticPlanner(out/'catalogue'/scene['navmesh']),sim,robot)
    root=LegalRoot.load(rt.read(b/'roots/legal_root.json'))
    result={'certificate':is_static_feasible(root,q,world,world.geometry,StaticConfig())}
    def contacts():
        return [{k:rt.clean(getattr(c,k)) for k in ['link_id_a','link_id_b','position_on_a_in_ws','position_on_b_in_ws','contact_normal_on_b_in_ws','contact_distance']} for c in sim.get_physics_contact_points() if robot.sim_obj.object_id in [c.object_id_a,c.object_id_b] and habitat_stage(c)]
    def habitat_stage(c): return rt.habitat_sim.stage_id in [c.object_id_a,c.object_id_b]
    result['probes']=[]
    for name,position,yaw in [('nominal',q['nominal_timed_trajectory'][51][1:4],q['nominal_timed_trajectory'][51][4]),('actual',failed['robot']['position'],failed['robot']['yaw'])]:
        passed=world.pose_free(position,yaw)
        robot.base_rot=yaw; robot.base_pos=mn.Vector3(*position);sim.perform_discrete_collision_detection()
        result['probes'].append(dict(name=name,passed=passed,contacts=contacts(),transform=rt.array(robot.sim_obj.transformation)))
    robot.sim_obj.transformation=mn.Matrix4(np.array(failed['robot']['transform']).T.tolist())
    sim.perform_discrete_collision_detection()
    result['exact_transform_contacts']=contacts()
    result['repeat_probes']=[]
    for name,position,yaw in [('nominal',q['nominal_timed_trajectory'][51][1:4],q['nominal_timed_trajectory'][51][4]),('actual',failed['robot']['position'],failed['robot']['yaw'])]:
        robot.base_rot=yaw;robot.base_pos=mn.Vector3(*position)
        for n in range(4):
            sim.perform_discrete_collision_detection()
            result['repeat_probes'].append(dict(name=name,n=n,contacts=contacts()))
    robot.sim_obj.transformation=world.original
    action=env.task.actions['agent_0_collector_velocity']
    result['dry_controls']=[]
    for i,command in enumerate(q['executable_controls']):
        action.step(agent_0_base_vel=np.array([command[0]/action._lin_speed,command[1]/action._ang_speed]))
        sim.perform_discrete_collision_detection()
        cs=contacts();bad=[c for c in cs if not(abs(c['contact_normal_on_b_in_ws'][1])>.8 and max(c['position_on_a_in_ws'][1],c['position_on_b_in_ws'][1])<robot.base_pos[1]+.12)]
        if bad:result['dry_controls'].append(dict(i=i,robot=rt.state(robot),contacts=bad))
    robot.sim_obj.transformation=world.original
    vc=rt.habitat_sim.physics.VelocityControl()
    vc.controlling_lin_vel=vc.lin_vel_is_local=vc.controlling_ang_vel=vc.ang_vel_is_local=True
    result['static_controls']=[]
    for i,(v,w) in enumerate(q['executable_controls']):
        if v or w:
            trans=robot.sim_obj.transformation
            rigid=rt.habitat_sim.RigidState(mn.Quaternion.from_matrix(trans.rotation()),trans.translation)
            vc.linear_velocity=mn.Vector3(v,0,0);vc.angular_velocity=mn.Vector3(0,w,0)
            target=vc.integrate_transform(q['integration_dt'],rigid)
            end=world.pf.try_step(rigid.translation,target.translation)
            if end!=rigid.translation:end-=robot.params.base_offset
            robot.sim_obj.transformation=mn.Matrix4.from_(target.rotation.to_matrix(),end)
        sim.perform_discrete_collision_detection()
        cs=contacts();bad=[c for c in cs if not(abs(c['contact_normal_on_b_in_ws'][1])>.8 and max(c['position_on_a_in_ws'][1],c['position_on_b_in_ws'][1])<robot.base_pos[1]+.12)]
        if i in [31,50] or bad:result['static_controls'].append(dict(i=i,robot=rt.state(robot),contacts=bad))
    (out/'probe.json').write_text(json.dumps(result,indent=2))
    os._exit(0)
rt._exact_root_checkpoint_hook=hook
rt.worker(SimpleNamespace(output=out/'case',worker_id='probe',query_id=None,limit_steps=None,protocol='v3',storage_profile='production',exact_root_preflight=True))
