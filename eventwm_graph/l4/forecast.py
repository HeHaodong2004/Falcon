"""Shared held L2 forecast adapter. Actual belief ownership stays with caller."""
from dataclasses import dataclass
import time
import numpy as np


@dataclass(frozen=True)
class BeliefSnapshot:
    belief_version: str
    model_version: str
    graph_version: str
    timestamp: float
    belief_reference: object
    support_metadata: dict


@dataclass(frozen=True)
class ForecastView:
    source_belief_version: str
    model_version: str
    prediction_start_time: float
    future_times: np.ndarray
    joint_future_reference: object
    valid_until: float
    support_metadata: dict

    def __post_init__(self):
        f=self.joint_future_reference
        if f.source.role!='actual' or f.source_fingerprint!=self.source_belief_version or f.model_version!=self.model_version:
            raise ValueError('Forecast lineage is not the declared actual belief')
        if len(self.future_times)<2 or not np.array_equal(self.future_times,f.times) or np.any(np.diff(f.times)<=0):
            raise ValueError('Invalid held forecast times')
        if abs(f.times[0]-self.prediction_start_time)>1e-7 or self.valid_until>f.times[-1]:
            raise ValueError('Forecast validity outside supported horizon')
        if not np.allclose(f.weights,1./len(f.weights)):
            raise ValueError('This ranking adapter requires uniform sampled joint worlds')

    def for_control(self,now):
        f=self.joint_future_reference
        if now<self.prediction_start_time-1e-7 or now>self.valid_until or now>self.future_times[-1]:
            return 'expired',np.empty((0,0,2)),np.empty(0),'observed_motion_fallback'
        # Every world remains represented. No hard safety assertion from probabilities.
        p=np.transpose(f.positions,(1,0,2,3)).reshape(len(f.times),-1,2)
        if f.hidden_positions.size:
            hidden=np.transpose(f.hidden_positions,(1,0,2,3)).reshape(len(f.times),-1,2)
            p=np.concatenate([p,hidden],axis=1)
        return 'valid',np.ascontiguousarray(p),f.times,'frozen_l2_shared'


class ForecastCache:
    def __init__(self,model,horizon=1.92,worlds=16,refresh_seconds=.24):
        self.model=model;self.horizon=horizon;self.worlds=worlds;self.refresh_seconds=refresh_seconds
        self.view=None;self.visibility=None;self.last_failure=None;self.refresh_ms=[]

    def get(self,actual_belief,observation,seed=221):
        if actual_belief.role!='actual': raise ValueError('Imagined belief cannot enter execution')
        now=float(observation['timestamp'])
        if abs(now-actual_belief.timestamp)>1e-7: raise ValueError('Actual update must precede forecast refresh')
        visible=tuple(sorted(t.get('track_id','') for t in observation.get('tracks',[]) if t.get('visible_now')))
        refresh=(self.view is None or visible!=self.visibility or now>=self.view.valid_until)
        if not refresh:
            # Material observed-vs-predicted discrepancy invalidates held forecasts.
            f=self.view.joint_future_reference;k=min(np.searchsorted(f.times,now),len(f.times)-1)
            p=f.positions[:,k].reshape(-1,2);p=p[np.isfinite(p).all(1)]
            for track in observation.get('tracks',[]):
                if track.get('visible_now') and (not len(p) or np.linalg.norm(p-np.asarray(track['position'])[[0,2]],axis=1).min()>.35):
                    refresh=True;break
        if refresh:
            began=time.perf_counter()
            try:
                f=self.model.predict_shared(actual_belief,self.horizon,self.worlds,seed)
                self.view=ForecastView(actual_belief.fingerprint(),self.model.model_version,now,f.times,f,
                    min(now+self.refresh_seconds,float(f.times[-1])),
                    dict(coordinates='world_XZ',horizon=self.horizon,joint_worlds=self.worlds,
                         current_observations_authoritative=True,conditional_risk_only=True))
                self.last_failure=None
            except (ValueError,RuntimeError,FloatingPointError) as exc:
                self.view=None;self.last_failure=type(exc).__name__+': '+str(exc)
            self.visibility=visible;self.refresh_ms.append((time.perf_counter()-began)*1000)
        return self.view
