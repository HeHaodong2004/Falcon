"""Runnable actual belief → packet → small graph consumer → finite continuation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from l2_partial_structural_api_example import build_model,legal_fixture,observation,queries
from eventwm_graph.l2_partial.planning import (
    online_planning_call,packet_from_batch,graph_decision_features,pessimistic_scores,
)
from eventwm_graph.l2_partial.structural_budget import UnsupportedConditioning
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.partial.query import compile_query,EXECUTOR_VERSION


def run(frozen=False):
    model,assets=build_model(frozen)
    history,graph=legal_fixture();actual=model.initialize(history,graph)
    qs=queries(actual.graph.version)
    result=online_planning_call(model,actual,observation(.048,2.01),qs,
        pointgoal=[3.,0.,0.],horizon=.288,num_worlds=16,seed=7,budget_ms=5000.)
    assert result.status=='COMPLETE'
    actual=result.actual_belief;packet=result.packet
    features=graph_decision_features(packet);scores=pessimistic_scores(packet)
    assert packet.graph_reference is actual.graph
    c=packet.candidates[0]
    branch=next(b for b in c.observation_branches if b.kind=='survived_no_detection')
    imagined=branch.successor.resolve()
    assert branch.successor.can_select_new_query
    # Recompile a new short reference on exactly the same observed geometry.
    g=PartialGraph(actual.graph.resolution,actual.graph.height)
    g.cells={k:{'state':s,'height':h,'source':'retained_arrived_certificate','timestamp':actual.timestamp}
        for k,(s,h) in actual.graph.materialize().items()}
    g.timestamp=imagined.timestamp;g.revision=actual.graph.version
    pose=imagined.robot_pose;target=pose[:3]+[.3*np.cos(pose[3]),0.,-.3*np.sin(pose[3])]
    query={'query_id':'next','executor_version':EXECUTOR_VERSION,'waypoints':[target.tolist()],'final_yaw':float(pose[3])}
    query['compiled']=compile_query(g,query,{'robot_pose':pose})
    assert query['compiled']['ok']
    shared=model.predict_shared(imagined,.288,16,7)
    batch=model.evaluate_queries(imagined,[query],shared,budget_ms=5000.,refined_query_indices=(0,))
    continuation=packet_from_batch(imagined,[query],batch,pointgoal=[3.,0.,0.])
    refused=False
    try:model.observe(imagined,observation(imagined.timestamp,2.037,imagined.robot_pose))
    except UnsupportedConditioning:refused=True
    assert refused
    # This is an explicitly synthetic arrived row. A live integration waits for
    # real execution; it must never synthesize this row from an imagined branch.
    actual_after=model.observe(actual,observation(imagined.timestamp,2.037,imagined.robot_pose))
    assert actual_after.carrier is None
    restored=StructuralBeliefState.from_dict(json.loads(json.dumps(actual_after.to_dict(),allow_nan=False)))
    assert restored.fingerprint()==actual_after.fingerprint()
    timeout=online_planning_call(model,actual,observation(actual.timestamp+.048,2.02),qs,budget_ms=0.)
    assert timeout.pending_observation and timeout.packet is None
    return {'packet_version':packet.version,'model_version':packet.model_version,'executor_version':packet.executor_version,
        'feature_shape':features.candidate_features.shape,'feature_names':features.feature_names,
        'pessimistic_scores':scores.tolist(),'shared_future_reference':packet.shared_future.reference,
        'contact_intervals':[c.collision_risk_interval for c in packet.candidates],
        'signed_goal_progress':[c.goal_progress.mean_interval for c in packet.candidates],
        'finite_continuation_status':continuation.computation_status,'exact_imagined_refinement_refused':refused,
        'actual_belief_retained_separately':actual_after.carrier is None,'actual_json_roundtrip':True,
        'atomic_timeout_status':timeout.status,'assets':assets,'fixture':'synthetic legal centers/observed cspace; no native evaluation'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frozen-checkpoints',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.frozen_checkpoints),allow_nan=False),flush=True)
