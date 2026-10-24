"""Generic externally supplied fixed-query optical events and finite successors."""
import numpy as np
from eventwm_graph.l2_partial.state import EXECUTOR_VERSION
from eventwm_graph.l2_partial.queries import _support


def predict_observations(model,belief,query,future,region=None,sensor_available=None):
    model._check(belief);model._check_future(future)
    if future.source_fingerprint!=belief.fingerprint():raise ValueError('foreign query future')
    if query.get('executor_version')!=EXECUTOR_VERSION:raise ValueError('foreign fixed follower')
    poses=np.asarray(query['compiled']['planned_poses'],float);actions=query['compiled']['actions']
    if poses.shape!=(len(actions)+1,4) or not np.isfinite(poses).all() or any(a not in ('forward','left','right') for a in actions):raise ValueError('invalid fixed query')
    if not np.allclose(poses[0],belief.robot_pose,atol=1e-5):raise ValueError('fixed query must start at arrived pose')
    if not hasattr(model,'_route_validator'):
        from eventwm_graph.l2_120k.model import load_model
        model._route_validator=load_model(kind='memory_cv')
    if not _support(model._route_validator,belief.graph,query,0):
        return dict(status='UNSUPPORTED_KNOWN_SPACE_QUERY',successor_supported=False)
    n=min(len(actions),120,int(np.floor((future.times[-1]-belief.timestamp+1e-8)/.048)))
    times=belief.timestamp+np.arange(n+1)*.048;indices=np.searchsorted(future.times,times-1e-9)
    if np.any(indices>=len(future.times)) or np.max(np.abs(future.times[indices]-times))>1e-7:raise ValueError('shared future lacks query endpoints')
    poses=poses[1:n+1];x=future.positions[:,indices[1:]];hidden=future.hidden_positions[:,indices[1:]]
    regimes=np.array([c.visibility_mode for c in future.endpoints[0]])
    pd=model.observation_law.detection(x,poses,regimes)
    # The inherited unlocalized channel uses coarse regime-specific detection,
    # before localization. Its geometry is deliberately not claimed calibrated.
    hd=np.broadcast_to(np.asarray(model.observation_law.hidden_detection)[regimes,None,None],hidden.shape[:-1]).copy()
    hd*=np.isfinite(hidden[...,0]);allx=np.concatenate([x,hidden],axis=2);pd=np.concatenate([pd,hd],axis=2)
    if sensor_available is not None:
        available=np.asarray(sensor_available)
        if available.dtype!=bool or available.shape!=(n,):raise ValueError('one Boolean human-availability flag per endpoint')
        pd*=available[None,:,None]
    if region is not None:
        box=np.asarray(region['bounds'],float);start=float(region['start_timestamp']);end=float(region['end_timestamp'])
        if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[:2]>box[2:]) or not np.isfinite([start,end]).all() or start>end:raise ValueError('invalid external region')
        inside=np.all((allx>=box[:2])&(allx<=box[2:]),axis=-1);active=(times[1:]>=start)&(times[1:]<=end)
        pd*=inside*active[None,:,None]
    endpoint_none=np.prod(1-pd,axis=2);survival=np.cumprod(endpoint_none,axis=1)
    before=np.c_[np.ones(len(future.weights)),survival[:,:-1]] if n else np.empty((len(future.weights),0))
    first=before*(1-endpoint_none);none=survival[:,-1] if n else np.ones(len(future.weights))
    return dict(status='NORMALIZED_FINITE_OPTICAL_APPROXIMATION',successor_supported=True,
        any_probability=float(future.weights@(1-none)),none_probability=float(future.weights@none),
        first_detection_mass=(future.weights@first).tolist(),endpoint_any_probability=(future.weights@(1-endpoint_none)).tolist(),
        endpoint_times=times[1:].tolist(),per_world_none=none,index=int(indices[-1]),robot_pose=query['compiled']['planned_poses'][n],
        future_identity=future.identity(),region=region,
        semantics='fixed exogenous humans, approximate optics, root emission excluded; no utility or visibility certificate')


def condition_observation_event(model,future,event_prediction,event):
    model._check_future(future)
    if event not in ('none','any') or not event_prediction.get('successor_supported'):raise ValueError('unsupported observation event')
    if event_prediction['future_identity']!=future.identity():raise ValueError('foreign event tape')
    none=np.asarray(event_prediction['per_world_none'],float)
    if none.shape!=future.weights.shape or not np.isfinite(none).all() or np.any(none<0) or np.any(none>1):raise ValueError('invalid event likelihood')
    index=event_prediction['index'];pose=np.asarray(event_prediction['robot_pose'],float)
    if not isinstance(index,(int,np.integer)) or not 0<=index<len(future.times) or pose.shape!=(4,) or not np.isfinite(pose).all():raise ValueError('invalid event endpoint/pose')
    lik=none if event=='none' else 1-none
    b=model.successor_at(future,event_prediction['index'],future.weights*lik,event_prediction['robot_pose'])
    return b.fork(diagnostics={**b.diagnostics,'conditioned_event':event,'event_probability':float(future.weights@lik),
        'region':event_prediction['region'],'observation_law':'same held joint worlds; exact imagined refinement unsupported'})
