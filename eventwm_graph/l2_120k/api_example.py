"""Runnable Layer 2-only demonstration with synthetic legally arrived inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from eventwm_graph.l2_partial.state import EXECUTOR_VERSION
from eventwm_graph.l2_partial.structural_budget import UnsupportedConditioning
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState
from .model import load_model
from .observations import predict_observations,condition_observation_event
from .data import OUT


def example(trained=False):
    motion=json.loads((OUT/'motion.json').read_text()) if trained else {'maneuver_rate':2.,'reset_std':.3}
    population=json.loads((OUT/'population_prior.json').read_text()) if trained else [.01,.8,.15,.04]
    wm=load_model(OUT/'training/checkpoint.pt' if trained else None,population,
        'docs/l2/partial_graph_wm_structural_20261021/artifacts/checkpoints/observation_fit.json' if trained else None,motion)
    cells=[dict(cell=[x,z],state=1,height=0.) for x in range(-10,30) for z in range(-10,11)]
    def row(t,x,static=()):
        return dict(timestamp=t,robot_pose=[0.,0.,0.,0.],static_updates=list(static),
            tracks=[dict(position=[x,0.,1.],visible_now=True,valid=True,last_seen_time=t)])
    initial=row(0.,2.,[{**c,'source':'synthetic_arrived_static_support','timestamp':0.} for c in cells])
    history=[initial,row(.048,2.02),row(.096,2.04)]
    graph=dict(timestamp=.096,resolution=.2,height=0.,revision=1,cells=cells)
    belief=wm.initialize(history,graph)
    belief=wm.observe(belief,row(.144,2.06))
    unconditioned=wm.propagate(belief,belief.timestamp+.24)
    poses=[[i*.12,0.,0.,0.] for i in range(25)]
    query=dict(query_id='fixed_demonstration_route',executor_version=EXECUTOR_VERSION,
        waypoints=[poses[-1][:3]],final_yaw=0.,compiled=dict(planned_poses=poses,actions=['forward']*24,graph_revision=1))
    future=wm.predict_shared(belief,.48,16,seed=73)
    optical=predict_observations(wm,belief,query,future)
    successor=condition_observation_event(wm,future,optical,'none')
    continued=wm.continue_rollout(successor,query,budget_ms=1e6,horizon=.48,seed=73)
    readout=wm.evaluate_queries(belief,[query],future,budget_ms=1e6,refined_queries=0)
    restored=StructuralBeliefState.from_dict(json.loads(json.dumps(belief.to_dict())))
    assert restored.fingerprint()==belief.fingerprint()
    refused=False
    try:wm.observe(successor,row(successor.timestamp,2.3))
    except UnsupportedConditioning:refused=True
    return dict(model_version=wm.model_version,initialized=True,observed=True,
        propagated_timestamp=unconditioned.timestamp,forecast_worlds=len(future.worlds),
        future_weight_sum=float(future.weights.sum()),any_detection=optical['any_probability'],
        no_detection=optical['none_probability'],successor_timestamp=successor.timestamp,
        continuation_cursor=continued.remaining_query_cursor,
        exact_imaginary_refinement_refused=refused,serialization_fingerprint_preserved=True,
        query_readout_status=readout.predictions[0].diagnostics['risk_semantics'],
        probability_status='NORMALIZED_APPROXIMATE; finite hypothetical successors; no rigorous-bound claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trained',action='store_true');a=p.parse_args()
    print(json.dumps(example(a.trained),indent=2))
