"""Public legal-input tracked-path baselines; no complete population claim.

Fitted JSON parameters come from evaluate.Kinematic on fitting-only labels.
The online classes import neither a target reader nor simulator geometry.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .context import OnlineContext
from .features import extract
from .types import SealedObservation,immutable_array


@dataclass(frozen=True)
class TrackedKinematicForecast:
    times: np.ndarray
    positions: np.ndarray
    supported: np.ndarray
    track_keys: tuple
    support: tuple = ('CONDITIONAL_TRACKED_PATH_BASELINE',
                      'UNATTRIBUTED_POPULATION_UNMODELED',
                      'UNKNOWN_GEOMETRY_UNCONSTRAINED_HUMAN_PATH_PRIOR')
    version: str = 'legal_tracked_kinematic_forecast_1'


class KinematicPredictor:
    def __init__(self,parameters):
        if isinstance(parameters,(str,Path)):parameters=json.loads(Path(parameters).read_text())
        if parameters['schema']!='legal_kinematic_baseline_1':raise ValueError('unknown baseline parameters')
        self.memory=bool(parameters['memory']);self.graph=bool(parameters['graph'])
        self.diffusion=float(parameters['diffusion'])
        self.coefficients=None if parameters['coefficients'] is None else np.array(parameters['coefficients'],float)
        if self.diffusion<=0 or not np.isfinite(self.diffusion):raise ValueError('invalid kinematic dispersion')
        if self.graph!=(self.coefficients is not None):raise ValueError('graph fit is missing')
        self.last_context=None;self.last_profile={}

    def predict(self,history,obs,partial_graph=None,horizon=3.,worlds=32,seed=None,actions=(),executed_action=None):
        if not 0<horizon<=3 or worlds<2:raise ValueError('unsupported kinematic request')
        started=time.perf_counter()
        if isinstance(history,OnlineContext):
            history.validate_certificate()
            if float(obs['timestamp'])>history.history[-1]['timestamp']:
                context=history.ingest(obs,executed_action,partial_graph)
            else:
                arrived=SealedObservation.from_raw(obs)
                if arrived.record!=history.history[-1].record:raise ValueError('conflicting current observation')
                context=history if partial_graph is None else OnlineContext.from_inputs(history.legal_history,obs,partial_graph,history.actions)
        else:
            if partial_graph is None:raise ValueError('raw history requires its legal partial graph')
            context=OnlineContext.from_inputs(history,obs,partial_graph,actions)
        if self.graph:
            f=extract(context);keys=f['keys'];age=f['age'];origin=f['origin'];velocity=f['velocity']
            drift=velocity.copy()
            drift+=np.c_[np.ones(len(keys)),f['x'],f['graph'].mean(1)]@self.coefficients
        else:
            memory=context.motion_summary.memory;keys=tuple(memory);now=float(obs['timestamp'])
            age=np.array([now-memory[k]['last'] for k in keys])
            velocity=np.array([memory[k]['velocity'] for k in keys],np.float32).reshape(-1,2)
            origin=np.array([memory[k]['pos']+memory[k]['velocity']*(now-memory[k]['last']) for k in keys],np.float32).reshape(-1,2)
            drift=velocity
        encoded=time.perf_counter()
        if seed is None:
            digest=hashlib.sha256(np.asarray(obs['timestamp'],np.float64).tobytes()+origin.tobytes()+repr(keys).encode()).digest()
            seed=int.from_bytes(digest[:8],'little')
        rng=np.random.default_rng(seed);times=np.linspace(0,horizon,64);n=len(keys)
        z=rng.normal(size=(worlds,n,2));initial=rng.normal(size=(worlds,n,2))*age[None,:,None]*self.diffusion
        positions=origin[None,None]+initial[:,None]+times[None,:,None,None]*(drift[None,None]+z[:,None]*self.diffusion)
        supported=np.ones(n,bool) if self.memory else age<1e-7
        result=TrackedKinematicForecast(immutable_array(times),immutable_array(positions),immutable_array(supported),keys)
        self.last_context=context;end=time.perf_counter()
        self.last_profile=dict(input_graph_history_ms=(encoded-started)*1000,
                               distribution_ms=(end-encoded)*1000,complete_ms=(end-started)*1000)
        return result
