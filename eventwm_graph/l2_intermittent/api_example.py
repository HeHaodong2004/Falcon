"""Runnable synthetic legal arrivals; no ground truth is used by inference."""
import argparse
import json
import numpy as np
from eventwm_graph.l2_partial.state import EXECUTOR_VERSION
from .common import OUT,DOC
from .model import IntermittentWorldModel
from .observations import predict_observations,condition_observation_event


def inputs():
    cells=[dict(cell=[x,z],state=1,height=0.) for x in range(-10,30) for z in range(-10,11)]
    def row(t,x,static=()):
        return dict(timestamp=t,robot_pose=[0.,0.,0.,0.],static_updates=list(static),tracks=[dict(position=[x,0.,1.],visible_now=True,valid=True,last_seen_time=t)])
    initial=row(0.,2.,[{**c,'source':'synthetic_arrived_static_support','timestamp':0.} for c in cells])
    history=[initial,row(.048,2.02),row(.096,2.04)]
    return history,dict(timestamp=.096,resolution=.2,height=0.,revision=1,cells=cells),row


def example(trained=False,checkpoint=None):
    path=OUT/'training/checkpoint.pt'
    if not path.exists():path=DOC/'artifacts/checkpoints/checkpoint.pt'
    wm=IntermittentWorldModel.load(checkpoint or path) if trained or checkpoint else IntermittentWorldModel()
    history,graph,row=inputs();belief=wm.initialize(history,graph)
    actual=wm.observe(belief,row(.144,2.06));future=wm.predict_shared(actual,3.,16,73)
    tracks=wm.get_previously_observed_tracks(actual)
    missing=wm.observe(actual,dict(timestamp=actual.timestamp+1.,robot_pose=[0.,0.,0.,0.],tracks=[],static_updates=[],human_sensor_available=False))
    aged_future=wm.predict_shared(missing,3.,16,73)
    # A genuine anonymous new center, independently supplied by the sensor.
    posterior=wm.observe(missing,row(missing.timestamp,2.35));corrected=wm.predict_shared(posterior,3.,16,73)
    query=lambda key,poses,actions:dict(query_id=key,executor_version=EXECUTOR_VERSION,waypoints=[poses[-1][:3]],final_yaw=poses[-1][3],compiled=dict(planned_poses=poses,actions=actions,graph_revision=1))
    q1=query('supplied_forward',[[i*.12,0.,0.,0.] for i in range(25)],['forward']*24)
    q2=query('supplied_turn',[[0.,0.,0.,i*.083333333333] for i in range(25)],['left']*24)
    original_identity=future.identity()
    events=[predict_observations(wm,actual,q,future) for q in (q1,q2)]
    child=condition_observation_event(wm,future,events[0],'none')
    tail=wm.predict_shared(child,horizon=3.,num_worlds=16,seed=73,times=np.unique(np.r_[0.,np.arange(1,63)*.048,3.]))
    field=wm.project_future_distribution(aged_future,[[-1.,-1.,4.,3.],[4.,-1.,8.,3.]],[aged_future.times[0],aged_future.times[-1]])
    restored=wm.deserialize_actual_belief(json.loads(json.dumps(wm.serialize_actual_belief(posterior))))
    assert restored.fingerprint()==posterior.fingerprint()
    assert future.identity()==original_identity
    exact_error=max(np.linalg.norm(c.mean[np.flatnonzero(np.isclose(c.last_seen,posterior.timestamp)),:2]-[2.35,1.]) for c in posterior.components)
    return dict(example='SYNTHETIC LEGAL INPUTS; no offline truth',model_version=wm.model_version,
        persistent_tracks_before=len(tracks),tracks_after_1s=len(wm.get_previously_observed_tracks(missing)),
        last_seen_age=wm.get_previously_observed_tracks(missing)[0]['age'],posterior_exact_position_error=float(exact_error),
        shared_worlds=len(future.weights),weight_sum=float(future.weights.sum()),supplied_query_detection_probabilities=[e['any_probability'] for e in events],
        shared_future_reused=True,conditional_successor_time=child.timestamp,continued_to=tail.times[-1],
        probability_field_expected_count=field['expected_count'].tolist(),serialization_roundtrip=True,
        future_shapes=dict(visible=list(future.positions.shape),missing=list(aged_future.positions.shape),corrected=list(corrected.positions.shape)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trained',action='store_true');p.add_argument('--checkpoint');a=p.parse_args();print(json.dumps(example(a.trained,a.checkpoint),indent=2))
