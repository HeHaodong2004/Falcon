"""Adapters reuse canonical legal tensors, motion compiler and frozen models.

No Habitat import. The future static-map adapter supplies goal paths and swept
footprint certificates. It cannot see hidden humans or outcome labels.
"""
import hashlib
import json
import numpy as np
import torch
from collector.core import LegalRoot, compile_motion, digest
from ..inputs import tensorize, collate
from ..v2.history import arrived_tracks
from ..v2.model import sample_query, interaction_features
from .interface import Candidate, Risk
from .calibration import probability


class GoalCandidateGenerator:
    def __init__(self, control_dt=.05,integration_dt=1/120,max_paths=4,waits=(0.,.5,1.),
                 speed=.7,slow_scale=.5,wait_horizon=2.,max_path_length=4.):
        if control_dt<=0 or integration_dt<=0 or max_paths<1 or speed<=0 or not 0<slow_scale<1 or wait_horizon<=0:
            raise ValueError('Invalid generator configuration')
        self.control_dt,self.integration_dt,self.max_paths=control_dt,integration_dt,max_paths
        self.waits,self.slow_scale,self.wait_horizon=tuple(waits),slow_scale,wait_horizon
        self.config=dict(directions=max_paths,target_radius=1.5,speed=speed,yaw_speed=1.5,
                         max_projection=.25,max_path_length=max_path_length,human_blind=True)

    def generate(self,legal_history,static_map,navigation_goal):
        last=legal_history[-1]
        rid=hashlib.sha256(json.dumps([last['timestamp'],last['robot_position'],last['robot_yaw']],sort_keys=True).encode()).hexdigest()[:16]
        root=LegalRoot(rid,tuple(last['robot_position']),float(last['robot_yaw']),self.control_dt,self.integration_dt)
        paths=static_map.goal_paths(root.position,navigation_goal,self.max_paths,self.config['max_path_length'])
        specs=[]
        for j,path in enumerate(paths[:self.max_paths]):
            path=np.asarray(path,dtype=float)
            if path.ndim!=2 or path.shape[1]!=3 or len(path)<2 or not np.allclose(path[0],root.position):
                raise ValueError('Static map returned invalid path')
            if np.linalg.norm(np.diff(path,axis=0)[:,[0,2]],axis=1).sum()>self.config['max_path_length']+1e-5:
                raise ValueError('Path exceeds declared local horizon')
            specs.extend((path.tolist(),j,wait,1.) for wait in self.waits)
            specs.append((path.tolist(),j,0.,self.slow_scale))
        specs.append(([list(root.position)],'WAIT',self.wait_horizon,0.))
        result=[]
        for path,direction,wait,scale in specs:
            q=compile_motion(root,path,[0.,0.],direction,wait,scale,self.config).data
            # Provider checks actual swept footprint, including turn poses.
            certified=bool(static_map.certify(q))
            if certified:
                result.append(Candidate(q['query_id'],digest(path),'WAIT' if direction=='WAIT' else 'SLOW' if scale<1 else 'MOVE',
                    q['planned_duration'],q['static_path_length'],tuple(tuple(p) for p in q['nominal_timed_trajectory']),
                    True,wait,scale,tuple(tuple(p) for p in q['executable_controls']),self.control_dt))
        return result


def model_inputs(history,candidates):
    # Query metadata/labels never enter tensorization: only legal history and
    # the precommitted nominal trajectory. No future simulator state is read.
    return collate([tensorize(dict(history=history,history_images=None,
        query=dict(nominal_timed_trajectory=c.trajectory))) for c in candidates])


class EventWMEstimator:
    def __init__(self,model):
        self.model=model.cpu().eval()
        for p in self.model.parameters():p.requires_grad_(False)

    @torch.no_grad()
    def predict(self,history,candidates):
        if not candidates:return []
        value=model_inputs(history,candidates)
        if not hasattr(self.model,'backbone'):
            collision=1-self.model(value).sigmoid()
            return [Risk(collision=float(p)) for p in collision]
        result=self.model(value,with_occupancy=False)
        if 'outcome_probabilities' in result:
            return [Risk(collision=float(p[2]),social_success=float(p[1]),safe_success=float(p[0]))
                    for p in result['outcome_probabilities']]
        return [Risk(collision=float(p)) for p in result['p_human_collision']]


class GaussianCVEstimator:
    def __init__(self,dynamics,calibration=None,samples=64):
        if dynamics.kind!='cv':raise ValueError('Gaussian CV requires CV dynamics')
        self.dynamics=dynamics.cpu().eval();self.calibration=calibration;self.samples=samples
        for p in self.dynamics.parameters():p.requires_grad_(False)

    @torch.no_grad()
    def predict(self,history,candidates):
        if not candidates:return []
        value=model_inputs(history,candidates);tracks,valid,sequence,mask=arrived_tracks(value)
        q=sample_query(value.query,self.samples,None)
        future=self.dynamics.forecast(tracks,valid,q[...,2],sequence,mask)
        features=interaction_features(q,tracks,valid,future)
        exposure=torch.trapz(features[...,0],q[...,2],dim=1).numpy()
        p=probability(exposure,self.calibration) if self.calibration else [None]*len(exposure)
        return [Risk(collision=None if pc is None else float(pc),exposure=float(e)) for pc,e in zip(p,exposure)]


class ModelFreeEstimator:
    """Declared fastest static-feasible baseline; supplies no safety probability."""
    def predict(self,history,candidates):return [Risk() for _ in candidates]
