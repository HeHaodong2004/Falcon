"""Keep live causal state on unary edges of the accepted action-prefix trie.

The native actions, per-endpoint human restoration, contacts, PSC, observation
order and sorted depth-first traversal are unchanged. Graph/sensor snapshots
are necessary at forks; encoding and restoring them on unary edges is not.
"""
import copy
import time

import numpy as np

from .execution import GraphPrefixFuture
from .map import PartialGraph
from .metrics import step_labels, summarize
from .query import FORWARD_STEP, TURN_STEP, static_body_clearance, turn_clearance


class LiveGraphPrefixFuture(GraphPrefixFuture):
    def evaluate_many(self, queries):
        import magnum as mn
        import falcon.additional_action as native_actions
        from ..robotmap.execution import signed_yaw
        robot = self.sim.get_agent_data(0).articulated_agent
        collision = self.task.measurements.measures['human_collision']
        trie = {'children': {}, 'queries': [], 'descendants': []}
        for j, q in enumerate(queries):
            node = trie; node['descendants'].append(j)
            for action in q['compiled']['actions']:
                if action not in ('forward', 'left', 'right'):
                    raise ValueError('STOP/unknown action in query')
                node = node['children'].setdefault(action, {'children': {}, 'queries': [], 'descendants': []})
                node['descendants'].append(j)
            node['queries'].append(j)

        def robot_state():
            return {'transform': mn.Matrix4(robot.sim_obj.transformation),
                    'joints': np.asarray(robot.sim_obj.joint_positions).copy(),
                    'velocities': np.asarray(robot.sim_obj.joint_velocities).copy(), 'gait': native_actions.play_i}

        root_state = dict(transform=self.robot_transform, joints=self.robot_joints,
                          velocities=self.robot_velocities, gait=self.gait_frame)
        results = [None]*len(queries)
        unique_steps = 0
        self.performance['fork_snapshots'] = 0
        self.performance['avoided_unary_snapshots'] = 0

        def finish(j, ids, reason, graph):
            selected = [self.nodes[i] for i in ids]
            observations = [n['observation'] for n in selected]
            labels = [n['labels'] for n in selected]
            poses = [self.root_observation['robot_pose'], *[r['robot_pose'] for r in observations]]
            q = queries[j]
            summary = summarize(labels, poses, self.times[0], q['compiled']['actions'], reason, q,
                                self.pointgoal, self.prefix_compliant, self.prefix_steps)
            results[j] = dict(query_id=q['query_id'], node_ids=list(ids), summary=summary,
                successor=dict(timestamp=observations[-1]['timestamp'] if observations else self.times[0],
                               robot_pose=poses[-1],
                               tracks=copy.deepcopy(observations[-1]['tracks'] if observations else self.root_observation['tracks']),
                               graph_revision=graph.revision), executor_version=q['executor_version'],
                execution_engine='native_exogenous_humans_partial_graph_prefix_dag_1')

        def visit(node, state, graph, ids):
            nonlocal unique_steps
            for j in node['queries']:
                finish(j, ids, 'query_complete', graph)
            if len(ids) >= self.max_steps:
                for child in node['children'].values():
                    for j in child['descendants']:
                        finish(j, ids, 'query_timeout', graph)
                return
            children = sorted(node['children'].items())
            if not children:
                return
            # Snapshot only when a sibling will need the parent's causal state.
            if len(children) > 1:
                graph_snapshot, sensor_state = graph.snapshot(), self.sensor.state()
                self.performance['fork_snapshots'] += 1
            else:
                self.performance['avoided_unary_snapshots'] += 1
            for branch, (action, child) in enumerate(children):
                i = len(ids)+1
                if i >= len(self.times):
                    raise ValueError('Shared future shorter than query horizon')
                began = time.perf_counter()
                robot.sim_obj.transformation = state['transform']; robot.sim_obj.joint_positions = state['joints']
                robot.sim_obj.joint_velocities = state['velocities']; native_actions.play_i = state['gait']
                if branch:
                    self.sensor.restore(sensor_state); graph = PartialGraph.restore(graph_snapshot)
                self.task.should_end = False; self.task.is_stop_called = False; collision._ever_collide = False
                # Retain even redundant native assignments/update calls: these
                # keep Bullet and the articulated camera on the accepted path.
                self._restore_humans(i-1); robot.update()
                self.performance['branch_restore_seconds'] = self.performance.get('branch_restore_seconds', 0.)+time.perf_counter()-began
                p = np.asarray(robot.base_pos, dtype=float); yaw = signed_yaw(robot)
                if action == 'forward':
                    nxt = p+[FORWARD_STEP*np.cos(yaw), 0., -FORWARD_STEP*np.sin(yaw)]
                    samples = max(1, int(np.ceil(FORWARD_STEP/.03)))
                    if (not graph.segment_free(p, nxt) or not all(static_body_clearance(graph, p+(nxt-p)*t, yaw)
                                                                for t in np.linspace(0., 1., samples+1))):
                        for j in child['descendants']:
                            finish(j, ids, 'route_invalidated', graph)
                        continue
                elif not turn_clearance(graph, p, yaw, yaw+(TURN_STEP if action == 'left' else -TURN_STEP)):
                    for j in child['descendants']:
                        finish(j, ids, 'route_invalidated', graph)
                    continue
                began = time.perf_counter()
                self.task.actions[self.rt.action_names[action]].step(task=self.task)
                self._restore_humans(i); robot.update()
                self.task.measurements.measures['did_multi_agents_collide'].update_metric(task=self.task)
                collision.update_metric(episode=self.rt.episode, task=self.task, observations={})
                self.performance['native_branch_execution_seconds'] = self.performance.get('native_branch_execution_seconds', 0.)+time.perf_counter()-began
                began = time.perf_counter(); row = self.sensor.observe(self.times[i])
                self.performance['visibility_observation_seconds'] = self.performance.get('visibility_observation_seconds', 0.)+time.perf_counter()-began
                began = time.perf_counter(); delta = graph.update(row)
                self.performance['partial_map_update_seconds'] = self.performance.get('partial_map_update_seconds', 0.)+time.perf_counter()-began
                began = time.perf_counter(); labels = step_labels(self.rt, self.times[i])
                self.performance['social_and_contact_metrics_seconds'] = self.performance.get('social_and_contact_metrics_seconds', 0.)+time.perf_counter()-began
                hit = bool(collision.get_metric())
                ended = bool(self.task.should_end or self.prefix_steps+i >= self.rt.config.habitat.environment.max_episode_steps)
                labels.update(episode_over=ended, official_human_collision=hit)
                nid = len(self.nodes)
                self.nodes.append(dict(node_id=nid, parent=ids[-1] if ids else None, action=action,
                                       observation=row, graph_delta=delta, labels=labels, future_index=i))
                unique_steps += 1; new_ids = ids+[nid]
                if ended:
                    reason = 'human_collision' if hit else 'episode_action_limit'
                    for j in child['descendants']:
                        finish(j, new_ids, reason, graph)
                else:
                    visit(child, robot_state(), graph, new_ids)
        # generate() has advanced the private simulator, so root state is always
        # explicitly restored. No observation from that latent future is reused.
        self.sensor.restore(self.sensor_root)
        visit(trie, root_state, PartialGraph.restore(self.graph_root), [])
        if any(r is None for r in results):
            raise RuntimeError('Incomplete query DAG')
        self.performance.update(unique_action_frames=unique_steps,
            proposed_action_frames=sum(len(q['compiled']['actions']) for q in queries),
            recorded_action_frames=sum(len(o['node_ids']) for o in results))
        return results
