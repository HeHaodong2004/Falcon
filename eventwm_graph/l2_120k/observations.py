"""Fixed-query human observation law, with no utility or query selection."""
from dataclasses import replace
import hashlib
import numpy as np

from eventwm_graph.l2_partial.state import EXECUTOR_VERSION
from eventwm_graph.l2_partial.queries import _support
from eventwm_graph.l2_partial.structural_model import _serialized


def _future_identity(future):
    digest=hashlib.sha256((future.input_fingerprint+future.source.model_version).encode())
    for x in (future.times,future.positions,future.velocities,future.weights):digest.update(x.tobytes())
    # Optical regimes can differ even when deterministic physical tapes agree.
    digest.update(repr(tuple((h.visibility_mode,tuple(h.slot_ids),tuple(h.hidden_pmf)) for h in future.worlds)).encode())
    return digest.hexdigest()


@_serialized
def predict_observations(model,belief,query,future,region=None,sensor_available=None):
    """None/any center detections and first-detection time on legal fixed motion.

    Optional region is an EXTERNALLY declared {bounds:[xmin,zmin,xmax,zmax],
    start_timestamp,end_timestamp}; it is a coarsening of exact center reports,
    not an imaginary exact measurement or an empty-space certificate.
    """
    model._check_belief(belief);model._validate_future_source(future)
    if future.input_fingerprint!=belief.fingerprint():raise ValueError('foreign shared future')
    if query.get('executor_version')!=EXECUTOR_VERSION:raise ValueError('foreign executor')
    poses=np.asarray(query['compiled']['planned_poses'],float)
    actions=query['compiled']['actions']
    if (poses.ndim!=2 or poses.shape[1]!=4 or len(poses)!=len(actions)+1 or not np.isfinite(poses).all()
            or any(a not in ('forward','left','right') for a in actions)):
        raise ValueError('invalid immutable query')
    total_actions=len(actions)
    cursor=belief.query_cursor or {}
    start_cursor=int(cursor.get('action_index',0)) if cursor.get('query_id')==query['query_id'] else 0
    if not 0<=start_cursor<=min(total_actions,120):raise ValueError('invalid fixed-query cursor')
    if not np.allclose(poses[start_cursor],belief.robot_pose,atol=1e-5):raise ValueError('query must start at legal belief pose')
    if belief.terminal or not _support(model,belief.graph,query,start_cursor):
        return dict(status='ABSORBED_OR_UNSUPPORTED_ROUTE',any_probability=0.,none_probability=1.,successor_supported=False)
    poses=poses[start_cursor:];actions=actions[start_cursor:]
    n=min(len(actions),120-start_cursor,int(np.floor((future.times[-1]-belief.timestamp+1e-8)/.048)))
    times=belief.timestamp+np.arange(n+1)*.048;indices=np.searchsorted(future.times,times-1e-9)
    if np.max(np.abs(future.times[indices]-times))>1e-7:raise ValueError('forecast must contain every query endpoint')
    positions=future.positions[:,indices[1:]]
    regimes=np.array([h.visibility_mode for h in future.worlds])
    pd=model.observation_law.detection(positions,poses[1:n+1],regimes)
    if sensor_available is not None:
        available=np.asarray(sensor_available,bool)
        if available.shape!=(n,):raise ValueError('one availability flag per endpoint')
        pd=pd*available[None,:,None]
    if region is not None:
        box=np.asarray(region['bounds'],float)
        if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[:2]>box[2:]):raise ValueError('invalid externally declared region')
        active=((times[1:]>=region['start_timestamp'])&(times[1:]<=region['end_timestamp']))
        inside=np.all((positions>=box[:2])&(positions<=box[2:]),axis=-1)
        pd=pd*inside*active[None,:,None]
    endpoint_none=np.prod(1-pd,axis=-1)
    survival=np.cumprod(endpoint_none,axis=1)
    before=np.c_[np.ones(len(future.worlds)),survival[:,:-1]] if n else np.empty((len(future.worlds),0))
    first=before*(1-endpoint_none)
    none=survival[:,-1] if n else np.ones(len(future.worlds))
    return dict(status='NORMALIZED_APPROXIMATE',any_probability=float(future.weights@(1-none)),
        none_probability=float(future.weights@none),first_detection_mass=(future.weights@first).tolist(),
        endpoint_any_probability=(future.weights@(1-endpoint_none)).tolist(),
        endpoint_times=times[1:].tolist(),index=int(indices[-1]),robot_pose=poses[n].copy(),
        query_cursor={'query_id':query['query_id'],'action_index':start_cursor+n,'total_actions':total_actions},
        per_world_none=none,per_world_last_none=endpoint_none[:,-1] if n else np.ones(len(none)),
        per_world_before_none=np.prod(endpoint_none[:,:-1],axis=1) if n else np.ones(len(none)),
        successor_supported=True,region=region,sensor_available=sensor_available,
        origin_future_sha256=_future_identity(future),
        semantics='fixed-query optical approximation; no contact/survival conditioning; root emission excluded')


@_serialized
def condition_observation_event(model,future,prediction,event):
    if event not in ('none','any') or not prediction['successor_supported']:
        raise ValueError('unsupported coarse event')
    if prediction.get('origin_future_sha256')!=_future_identity(future):
        raise ValueError('observation event belongs to a different shared world tape')
    none=prediction['per_world_none'];lik=none if event=='none' else 1-none
    successor=model.successor_at(future,prediction['index'],future.weights*lik,
        prediction['robot_pose'],cursor=prediction['query_cursor'],refinement=('declared-coarse-event',event))
    # Preserve exact empty refinement semantics for the ordinary unmasked event.
    if prediction['region'] is None and prediction['sensor_available'] is None:
        name='survived_no_detection' if event=='none' else 'survived_detection'
        empty=(1-prediction['per_world_before_none'])*prediction['per_world_last_none']/np.maximum(1-none,1e-300)
        ledger=(name,successor.timestamp,tuple(prediction['robot_pose']),tuple(empty) if event=='any' else ())
    else:
        ledger=('external_detection_event',successor.timestamp,event)
    successor=successor.fork(carrier=replace(successor.carrier,
        observation_ledger=successor.carrier.observation_ledger+(ledger,)),
        evidence={**successor.evidence,'declared_observation_event':event,
            'external_event_exact_refinement_unsupported':prediction['region'] is not None or prediction['sensor_available'] is not None,
            'external_event_timestamp':successor.timestamp,
            'observation_region':prediction['region'],'probability_status':'NORMALIZED_FINITE_APPROXIMATION'})
    return successor
