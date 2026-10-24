"""Shared exogenous native futures + action-prefix DAG, with causal graph state.

Reuses the production collector's pinned SharedFuture restoration and native
human generation. This separate implementation adds dynamic visibility, partial
map deltas, native contacts and trajectory-wide PSC; no rendering or training.
"""
import copy
import time
import numpy as np
from ..stage1.runtime import SharedFuture
from .map import PartialGraph
from .metrics import step_labels,summarize
from .query import FORWARD_STEP,TURN_STEP,static_body_clearance,turn_clearance


class GraphPrefixFuture(SharedFuture):
    def __init__(self,rt,observer,graph,pointgoal,max_steps=120,prefix_compliant=0):
        super().__init__(rt)
        self.sensor=observer;self.sensor_root=observer.state();self.graph_root=graph.snapshot()
        self.pointgoal=list(pointgoal);self.max_steps=int(max_steps);self.prefix_compliant=prefix_compliant
        self.nodes=[];self.performance={}

    def evaluate_many(self,queries):
        import magnum as mn
        import falcon.additional_action as native_actions
        from ..robotmap.execution import signed_yaw
        robot=self.sim.get_agent_data(0).articulated_agent
        collision=self.task.measurements.measures['human_collision']
        trie={'children':{},'queries':[],'descendants':[]}
        for j,q in enumerate(queries):
            node=trie;node['descendants'].append(j)
            for action in q['compiled']['actions']:
                if action not in ('forward','left','right'):raise ValueError('STOP/unknown action in query')
                node=node['children'].setdefault(action,{'children':{},'queries':[],'descendants':[]})
                node['descendants'].append(j)
            node['queries'].append(j)
        def robot_state():
            return {'transform':mn.Matrix4(robot.sim_obj.transformation),
                    'joints':np.asarray(robot.sim_obj.joint_positions).copy(),
                    'velocities':np.asarray(robot.sim_obj.joint_velocities).copy(),'gait':native_actions.play_i}
        root_state={'transform':self.robot_transform,'joints':self.robot_joints,
                    'velocities':self.robot_velocities,'gait':self.gait_frame}
        results=[None]*len(queries);unique_steps=0
        def finish(j,ids,reason,graph):
            selected=[self.nodes[i] for i in ids]
            observations=[n['observation'] for n in selected]
            labels=[n['labels'] for n in selected]
            poses=[self.root_observation['robot_pose'],*[r['robot_pose'] for r in observations]]
            q=queries[j]
            summary=summarize(labels,poses,self.times[0],q['compiled']['actions'],reason,q,self.pointgoal,
                              self.prefix_compliant,self.prefix_steps)
            results[j]={'query_id':q['query_id'],'node_ids':list(ids),'summary':summary,
                        'successor':{'timestamp':observations[-1]['timestamp'] if observations else self.times[0],
                            'robot_pose':poses[-1],'tracks':copy.deepcopy(observations[-1]['tracks'] if observations else self.root_observation['tracks']),
                            'graph_revision':graph.revision},
                        'executor_version':q['executor_version'],
                        'execution_engine':'native_exogenous_humans_partial_graph_prefix_dag_1'}
        def visit(node,state,sensor_state,graph_snapshot,ids):
            nonlocal unique_steps
            graph=PartialGraph.restore(graph_snapshot)
            for j in node['queries']:finish(j,ids,'query_complete',graph)
            if len(ids)>=self.max_steps:
                for child in node['children'].values():
                    for j in child['descendants']:finish(j,ids,'query_timeout',graph)
                return
            for action,child in sorted(node['children'].items()):
                i=len(ids)+1
                if i>=len(self.times):raise ValueError('Shared future shorter than query horizon')
                restore_started=time.perf_counter()
                robot.sim_obj.transformation=state['transform'];robot.sim_obj.joint_positions=state['joints']
                robot.sim_obj.joint_velocities=state['velocities'];native_actions.play_i=state['gait']
                self.sensor.restore(sensor_state);graph=PartialGraph.restore(graph_snapshot)
                self.task.should_end=False;self.task.is_stop_called=False;collision._ever_collide=False
                self._restore_humans(i-1);robot.update()
                self.performance['branch_restore_seconds']=self.performance.get('branch_restore_seconds',0.)+time.perf_counter()-restore_started
                p=np.asarray(robot.base_pos,dtype=float);yaw=signed_yaw(robot)
                if action=='forward':
                    nxt=p+[FORWARD_STEP*np.cos(yaw),0.,-FORWARD_STEP*np.sin(yaw)]
                    samples=max(1,int(np.ceil(FORWARD_STEP/.03)))
                    if (not graph.segment_free(p,nxt) or
                            not all(static_body_clearance(graph,p+(nxt-p)*t,yaw) for t in np.linspace(0.,1.,samples+1))):
                        for j in child['descendants']:finish(j,ids,'route_invalidated',graph)
                        continue
                elif not turn_clearance(graph,p,yaw,yaw+(TURN_STEP if action=='left' else -TURN_STEP)):
                    for j in child['descendants']:finish(j,ids,'route_invalidated',graph)
                    continue
                start=time.perf_counter()
                self.task.actions[self.rt.action_names[action]].step(task=self.task)
                self._restore_humans(i);robot.update()
                self.task.measurements.measures['did_multi_agents_collide'].update_metric(task=self.task)
                collision.update_metric(episode=self.rt.episode,task=self.task,observations={})
                self.performance['native_branch_execution_seconds']=self.performance.get('native_branch_execution_seconds',0.)+time.perf_counter()-start
                start=time.perf_counter();row=self.sensor.observe(self.times[i])
                self.performance['visibility_observation_seconds']=self.performance.get('visibility_observation_seconds',0.)+time.perf_counter()-start
                start=time.perf_counter();delta=graph.update(row)
                self.performance['partial_map_update_seconds']=self.performance.get('partial_map_update_seconds',0.)+time.perf_counter()-start
                start=time.perf_counter();labels=step_labels(self.rt,self.times[i])
                self.performance['social_and_contact_metrics_seconds']=self.performance.get('social_and_contact_metrics_seconds',0.)+time.perf_counter()-start
                hit=bool(collision.get_metric())
                ended=bool(self.task.should_end or self.prefix_steps+i>=self.rt.config.habitat.environment.max_episode_steps)
                labels.update(episode_over=ended,official_human_collision=hit)
                nid=len(self.nodes)
                self.nodes.append({'node_id':nid,'parent':ids[-1] if ids else None,'action':action,
                    'observation':row,'graph_delta':delta,'labels':labels,'future_index':i})
                unique_steps+=1;new_ids=ids+[nid]
                if ended:
                    reason='human_collision' if hit else 'episode_action_limit'
                    for j in child['descendants']:finish(j,new_ids,reason,graph)
                else:visit(child,robot_state(),self.sensor.state(),graph.snapshot(),new_ids)
        visit(trie,root_state,self.sensor_root,self.graph_root,[])
        if any(r is None for r in results):raise RuntimeError('Incomplete query DAG')
        self.performance.update(unique_action_frames=unique_steps,
            proposed_action_frames=sum(len(q['compiled']['actions']) for q in queries),
            recorded_action_frames=sum(len(o['node_ids']) for o in results))
        return results
