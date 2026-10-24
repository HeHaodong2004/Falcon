"""Read-only asset/native smoke test. Run before developing the L4 controller."""
import hashlib
import json
from pathlib import Path
import time

import habitat_sim  # Required before habitat on the pinned installation.
import numpy as np

from eventwm_graph.benchmark.map import load_certified_map
from eventwm_graph.partial.demo import scene_manifest, prefix_actions
from eventwm_graph.partial.initialization import make_episode
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.partial.observer import GeometryObserver
from eventwm_graph.partial.query import compile_query, generate_queries
from eventwm_graph.partial.metrics import contact_labels
from eventwm_graph.stage1.runtime import NativeEpisode
from eventwm_graph.l2_intermittent.model import IntermittentWorldModel


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    checkpoint = Path('/research-user/workspace/projects/eventwm-falcon-l2-intermittent-500k-20261022/output/l2_intermittent_500k_20261022/training/checkpoint.pt')
    h = load_certified_map(scene_manifest('hm3d_mt5PhMmTk5m'))
    ep, initialization = make_episode(h, h.valid_starts()[4]['node_id'], 310201001, 2)
    report = dict(source_commit='cf1971228c37517202cc9e5787f785a3876a0514',
                  map_id=h.map_id, manifest=str(h.manifest_path), contract=h.contract,
                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    history = []; transitions = []
    with NativeEpisode(h, ep, 310201001) as rt:
        sensor = GeometryObserver(rt, h.robot_config.collision_reference / 6)
        ob = sensor.observe(); history.append(ob)
        graph = PartialGraph(sensor.resolution, ob['robot_pose'][1]); graph.update(ob)
        report['legal_native_actions'] = list(rt.action_names)
        report['configured_robot_actions'] = [k for k in rt.task.actions if k.startswith('agent_0')]
        for action in prefix_actions(0):
            before = history[-1]
            rt.step(action); ob = sensor.observe(); graph.update(ob); history.append(ob)
            transitions.append(dict(action=action, start=before['timestamp'], end=ob['timestamp'],
                                    before=before['robot_pose'], after=ob['robot_pose'],
                                    contacts=contact_labels(rt), episode_over=rt.env.episode_over))
            if rt.env.episode_over: break
        queries, diagnostics = generate_queries(graph, ob['tracks'], ob, ep.goals[0].position, limit=8)
        report['query_count'] = len(queries)
        report['visible_observation_count'] = sum(t.get('visible_now', False) for row in history for t in row['tracks'])
        report['observation_keys'] = list(ob)
        report['track_example'] = next((t for row in history for t in row['tracks'] if t.get('visible_now')), None)
        report['graph_summary'] = dict(resolution=graph.resolution, cells=len(graph.cells), revision=graph.revision)
        if queries:
            query = queries[0]
        else:
            # The partial graph may not yet certify the full body at this root.
            # Independently exercise an existing native orientation compiler.
            result=h.model.compile([ob['robot_pose'][:3]], start_yaw=ob['robot_pose'][3],
                                   target_yaw=ob['robot_pose'][3]+.25)
            if not result['reached']: raise RuntimeError('Native orientation compilation failed')
            query=dict(query_id='smoke_native_orientation', waypoints=[ob['robot_pose'][:3]],
                       final_yaw=ob['robot_pose'][3]+.25, compiled=result)
            report['partial_query_rejections']=diagnostics
        report['query'] = query
        for action in query['compiled']['actions']:
            before = ob; rt.step(action); ob = sensor.observe(); graph.update(ob); history.append(ob)
            transitions.append(dict(action=action, start=before['timestamp'], end=ob['timestamp'],
                                    before=before['robot_pose'], after=ob['robot_pose'],
                                    contacts=contact_labels(rt), episode_over=rt.env.episode_over))
            if rt.env.episode_over: break
        began=time.perf_counter(); model=IntermittentWorldModel.load(checkpoint)
        report['model_load_seconds']=time.perf_counter()-began
        began=time.perf_counter(); belief=model.initialize(history, graph.snapshot())
        report['initialize_seconds']=time.perf_counter()-began
        began=time.perf_counter(); future=model.predict_shared(belief, horizon=1.92, num_worlds=16, seed=221)
        report['first_forecast_seconds']=time.perf_counter()-began
        report['forecast']=dict(model_version=model.model_version, source_time=belief.timestamp,
                                times=future.times.tolist(), positions_shape=list(future.positions.shape),
                                weights_sum=float(future.weights.sum()), coordinates='world X,Z',
                                actual_role=belief.role, source_fingerprint=future.source_fingerprint)
        if not rt.env.episode_over:
            before=ob; rt.step('stop'); ob=sensor.observe()
            report['stop']=dict(dt=ob['timestamp']-before['timestamp'], episode_over=rt.env.episode_over,
                                 is_stop_called=rt.task.is_stop_called, metrics=rt.env.get_metrics())
    report['transitions']=transitions
    report['dt_range']=[min(t['end']-t['start'] for t in transitions),max(t['end']-t['start'] for t in transitions)]
    (out/'baseline_smoke.json').write_text(json.dumps(report, indent=2))
    (out/'baseline_legal_history.json').write_text(json.dumps(dict(history=history, graph=graph.snapshot()), indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('transitions','contract','query')}, indent=2))


if __name__ == '__main__': main()
