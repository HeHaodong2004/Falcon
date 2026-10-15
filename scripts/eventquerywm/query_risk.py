"""Read-only frozen physical forecasting and deterministic nominal-query exposure.

The exposure score is a linear space-time occupancy-mass heuristic. It never
multiplies cell or time survival probabilities or asserts temporal independence.
"""
from dataclasses import dataclass
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from .inputs import tensorize
from .track_uncertainty import TrackUncertainty,short_history

@dataclass(frozen=True)
class OccupancyForecast:
    probability:np.ndarray       # [5,z,x]; foreground center occupancy marginals
    horizons_s:np.ndarray        # Relative to decision time
    valid_horizons:np.ndarray    # Available at inference, never realized branch censoring
    robot_position_world:np.ndarray
    robot_yaw:float
    checkpoint_sha256:str
    cell_size_m:float=.5
    lower_bound_m:float=-6.

@dataclass(frozen=True)
class ExposureConfig:
    human_radius_m:float=.30
    clearance_margin_m:float=.10
    robot_radius_m:object=None   # None uses legal query research_radius
    max_horizon_s:float=1.       # Preferred short-horizon successor; fixed before scoring

class FrozenSuccessor:
    def __init__(self,checkpoint,device='cpu',expected_sha256=None):
        path=Path(checkpoint);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest!=expected_sha256:raise ValueError('Frozen checkpoint fingerprint mismatch')
        state=torch.load(path,map_location='cpu');kind=state['kind']
        self.model=TrackUncertainty(state['model']['background'],kind).to(device)
        self.model.load_state_dict(state['model'],strict=True);self.model.eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.device=device;self.checkpoint_sha256=digest;self.kind=kind

    @classmethod
    def from_manifest(cls,manifest,device='cpu'):
        path=Path(manifest);record=json.loads(path.read_text())
        return cls(path.parent/record['checkpoint_relative_path'],device,record['checkpoint_sha256'])

    @torch.no_grad()
    def predict(self,legal_arrived_history,horizon_mask=None):
        """Accept canonical legal history from collector.core.load_wm_input.

        Query is absent from this API. horizon_mask may describe only legally
        known forecast availability; never feed realized future termination masks.
        """
        history=legal_arrived_history
        if not history:raise ValueError('Empty legal history')
        last=history[-1];origin=np.asarray(last['robot_position'],dtype=float);yaw=float(last['robot_yaw'])
        # Reuse authoritative legal tensor semantics with a dummy stationary query.
        # No real candidate query can enter the physical forecaster.
        nominal=[[0.,*origin,yaw],[3.,*origin,yaw]]
        value=tensorize(dict(history=history,history_images=None,query={'nominal_timed_trajectory':nominal}))
        features=short_history([{'input':value}],self.device)
        p=self.model(*features)[0].cpu().numpy().copy();h=self.model.horizons.cpu().numpy().copy()
        valid=np.isfinite(p).all((1,2))&((p>=0)&(p<=1)).all((1,2))
        if horizon_mask is not None:
            requested=np.asarray(horizon_mask)
            if requested.dtype!=np.bool_ or requested.shape!=h.shape:raise ValueError('Expected boolean availability mask at supported horizons')
            valid &= requested
        for array in (p,h,valid,origin):array.setflags(write=False)
        return OccupancyForecast(p,h,valid,origin,yaw,self.checkpoint_sha256)

def query_exposure(forecast,nominal_query,config=ExposureConfig()):
    """Score timed ego-footprint overlap against one fixed physical forecast.

    Higher score means greater forecast occupancy exposure on supported samples.
    Every covered time interval joins adjacent, valid trained horizons. No
    extrapolation, invalid-mask bridging, future collision masks or outcomes.
    """
    q=nominal_query;trajectory=np.asarray(q['nominal_timed_trajectory'],dtype=float)
    if trajectory.ndim!=2 or trajectory.shape[1]!=5 or len(trajectory)<2 or not np.isfinite(trajectory).all():raise ValueError('Expected finite [t,x,y,z,yaw] nominal trajectory')
    clock=trajectory[:,0]
    if clock[0]!=0 or np.any(np.diff(clock)<=0):raise ValueError('Query clock must start at0 and strictly increase')
    duration=float(q['planned_duration'])
    if not math.isfinite(duration) or duration<=0 or abs(clock[-1]-duration)>1e-5:raise ValueError('Nominal query duration/trajectory mismatch')
    if np.linalg.norm(trajectory[0,1:4]-forecast.robot_position_world)>1e-4:raise ValueError('Query and forecast decision origins differ')
    delta_yaw=math.atan2(math.sin(trajectory[0,4]-forecast.robot_yaw),math.cos(trajectory[0,4]-forecast.robot_yaw))
    if abs(delta_yaw)>1e-4:raise ValueError('Query and forecast decision orientations differ')
    radius=float(q['robot_spec']['research_radius']) if config.robot_radius_m is None else float(config.robot_radius_m)
    if any(not math.isfinite(x) or x<0 for x in (radius,config.human_radius_m,config.clearance_margin_m)) or radius<=0:raise ValueError('Invalid footprint/clearance radii')
    if not math.isfinite(config.max_horizon_s) or config.max_horizon_s<=0:raise ValueError('Invalid scoring horizon limit')
    p=np.asarray(forecast.probability);h=np.asarray(forecast.horizons_s,dtype=float);mask=np.asarray(forecast.valid_horizons,dtype=bool)
    if p.shape!=(len(h),24,24) or mask.shape!=h.shape or not np.isfinite(h).all() or np.any(np.diff(h)<=0) or np.any(h<=0):raise ValueError('Unsupported occupancy forecast shape/clock')
    rotation=np.array([[math.cos(forecast.robot_yaw),-math.sin(forecast.robot_yaw)],[math.sin(forecast.robot_yaw),math.cos(forecast.robot_yaw)]])
    local=(trajectory[:,[1,3]]-forecast.robot_position_world[[0,2]])@rotation.T
    cell=forecast.cell_size_m;lower=forecast.lower_bound_m;upper=lower+24*cell
    centers=lower+(np.arange(24)+.5)*cell;x,z=np.meshgrid(centers,centers)
    clearance=radius+config.human_radius_m+config.clearance_margin_m
    active=np.zeros(len(h),bool);exposure=np.full(len(h),np.nan);details=[]
    for k,time_s in enumerate(h):
        reason=None;ego=None
        if not mask[k]:reason='forecast_unavailable'
        elif time_s>duration+1e-8:reason='after_nominal_query_completion'
        elif time_s>config.max_horizon_s+1e-8:reason='outside_requested_scoring_horizon'
        elif not np.isfinite(p[k]).all() or np.any((p[k]<0)|(p[k]>1)):reason='invalid_forecast_values'
        else:
            ego=np.array([np.interp(time_s,clock,local[:,axis]) for axis in range(2)])
            if np.any(ego-clearance<lower) or np.any(ego+clearance>upper):reason='clearance_footprint_outside_forecast_grid'
            else:
                # Include every grid cell intersecting the conservative clearance disk.
                dx=np.maximum(np.abs(x-ego[0])-cell/2,0);dz=np.maximum(np.abs(z-ego[1])-cell/2,0)
                footprint=dx*dx+dz*dz<=clearance**2
                exposure[k]=float(p[k][footprint].sum(dtype=float));active[k]=True
        details.append(dict(horizon_s=float(time_s),valid=bool(active[k]),reason=reason,
            nominal_ego_relative_xz=ego.tolist() if ego is not None else None,occupancy_mass=float(exposure[k]) if active[k] else None))
    intervals=[];integral=0.;covered=0.
    for k in range(len(h)-1):
        if active[k] and active[k+1]:
            dt=h[k+1]-h[k];value=.5*(exposure[k]+exposure[k+1])*dt
            integral+=value;covered+=dt;intervals.append([float(h[k]),float(h[k+1])])
    # Missing all intervals returns unknown, never a spurious zero-risk score.
    return dict(score=float(integral/covered) if covered else None,score_semantics='Mean marginal human-center occupancy mass in clearance footprint over supported time intervals; ranking heuristic, not collision probability',
        integrated_exposure_cell_mass_seconds=float(integral) if covered else None,peak_horizon_occupancy_mass=float(np.nanmax(exposure)) if active.any() else None,
        covered_seconds=float(covered),nominal_duration_s=duration,coverage_fraction=float(covered/duration),unscored_seconds=float(max(0,duration-covered)),
        covered_intervals_s=intervals,horizons=details,status='partial_horizon_coverage' if covered else 'no_supported_time_intervals',
        full_query_risk_available=False,robot_radius_m=radius,human_radius_m=config.human_radius_m,clearance_margin_m=config.clearance_margin_m,
        combined_clearance_radius_m=clearance,temporal_independence_assumed=False,physical_query_independent=True,
        limitations=['No forecast before first trained horizon or beyond selected horizons; sparse-time trapezoid is a numerical heuristic',
                    'Human-center grid and disc approximation are not native collision geometry','Unseen humans use training background prior; static obstacles are not modeled',
                    'Scores with different covered intervals are not directly comparable without checking coverage; missing exposure never means safe'])
