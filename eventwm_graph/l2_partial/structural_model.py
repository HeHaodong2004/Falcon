"""Structural revision of the existing graph-native population world model.

No forecasting or event parameters are fitted here. Public operations serialize
access to this instance's caches. Beliefs are returned atomically. A hypothetical
coarse successor carries its finite path measure instead of inventing endpoint
Gaussian noise. Continuous refinement of that measure is explicitly unsupported.
"""
import hashlib
import copy
import json
import math
import threading
from collections import OrderedDict
import time
from dataclasses import replace
from functools import wraps
from contextvars import ContextVar

import numpy as np

from .repair_model import RepairedGraphBeliefWorldModel, NATIVE_DT
from .state import BeliefState, SharedFuture, EXECUTOR_VERSION
from .structural_state import StructuralBeliefState, WorldCarrier, seal_future, immutable_array, freeze_hypothesis
from .structural_budget import (ACTIVE_DEADLINE, Deadline, check_budget,
                                UnsupportedConditioning)

_VALIDATED_FUTURE=ContextVar('l2_structural_validated_future',default=None)


def _serialized(fn):
    @wraps(fn)
    def invoke(self, *args, **kwargs):
        token=None
        active=ACTIVE_DEADLINE.get()
        if active is None and fn.__name__ in ('online_call','evaluate_queries','continue_rollout'):
            offset={'online_call':6,'evaluate_queries':4,'continue_rollout':2}[fn.__name__]
            default=None if fn.__name__=='online_call' else 40.
            budget=kwargs.get('budget_ms',args[offset] if len(args)>offset else default)
            if budget is not None:
                active=Deadline.create(budget,pending_observation=fn.__name__=='online_call')
                token=ACTIVE_DEADLINE.set(active)
        acquired=False
        try:
            if active is None:
                self._instance_lock.acquire();acquired=True
            else:
                remaining=max(0.,active.budget_ms/1000.-(time.perf_counter()-active.start))
                acquired=self._instance_lock.acquire(timeout=remaining)
                active.check('instance_lock')
            return fn(self, *args, **kwargs)
        finally:
            if acquired:self._instance_lock.release()
            if token is not None:ACTIVE_DEADLINE.reset(token)
    return invoke


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        default=lambda x: np.asarray(x).tolist(), separators=(',', ':')).encode()).hexdigest()


def _finite_time(value, name='timestamp'):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return value


def _pose(value):
    value = np.asarray(value, float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError('robot_pose must contain four finite coordinates')
    return value


def _validate_static_arrival(rows, timestamp):
    """Same causal delta boundary as Stage 0; omitted fixture clocks inherit t."""
    for update in rows:
        if abs(_finite_time(update.get('timestamp', timestamp)) - timestamp) > 1e-8:
            raise ValueError('static delta must belong to the arrived observation clock')
        cell = update.get('cell')
        if (not isinstance(cell, (list, tuple)) or len(cell) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in cell)):
            raise ValueError('static cell must be an integer pair')
        if isinstance(update.get('state'), bool) or update.get('state') not in (1, 2):
            raise ValueError('arrived static state must be KNOWN_FREE or KNOWN_OCCUPIED')
        _finite_time(update.get('height'), 'static height')
        if not isinstance(update.get('source'), str) or not update['source']:
            raise ValueError('static certificate source required')


class StructuralGraphBeliefWorldModel(RepairedGraphBeliefWorldModel):
    """Versioned structural API, preserving the trained compact graph network."""
    structural_version = 'partial-graph-structural-0.2'

    def __init__(self, *args, residual_policy='bound', optimized=True, **kwargs):
        self._instance_lock = threading.RLock()
        super().__init__(*args, **kwargs)
        if not math.isfinite(self.maneuver_rate) or self.maneuver_rate<0 or not math.isfinite(self.reset_std) or self.reset_std<0 or not math.isfinite(self.maneuver_retention) or not 0<=self.maneuver_retention<=1:
            raise ValueError('finite nonnegative maneuver rate/std and retention in [0,1] required')
        if residual_policy not in ('bound', 'legacy_parity_only'):
            raise ValueError('residual_policy must be bound or legacy_parity_only')
        self.residual_policy = residual_policy
        self.optimized = bool(optimized)
        self.endpoint_template_cache=OrderedDict()
        if self.optimized:
            from .structural_rollout_kernel import warmup_rollout_kernel
            warmup_rollout_kernel()
        self.model_version = self.structural_version + '-' + _digest({
            'base': self.model_version, 'residual_policy': residual_policy,
            'observation': 'native-endpoint-law; immutable finite coarse carrier',
            'successor': 'analytic-unconditioned or atomic finite conditional; no smoothing',
            'sampler':'unique symmetric PSD Gaussian root; persistent carrier RNG stream',
            'evidence': 'separate bounded sensor/execution ledgers'})[:16]

    def history_filter(self):
        return StructuralGraphBeliefWorldModel(kind='structured',
            max_hypotheses=self.max_hypotheses, population_prior=self.population_prior,
            observation_law=self.observation_law, maneuver_rate=self.maneuver_rate,
            reset_std=self.reset_std, maneuver_retention=self.maneuver_retention,
            residual_policy=self.residual_policy, optimized=self.optimized)

    def prepare(self,belief):
        b=super().prepare(belief)
        positive=all(h.relaxation>0 for h in b.hypotheses if h.weight>0)
        evidence=dict(b.evidence)
        if b.unresolved_mass==0:
            evidence['residual_count_support']=sorted({h.n+int(k) for h in b.hypotheses if h.weight>0 for k in np.flatnonzero(h.hidden_pmf>0)})
        if not positive:
            evidence.pop('transition_noise_contract',None)
        elif b.unresolved_mass==0:
            # The complete incoming measure is available here. All subsequent
            # omitted components inherit this independent additive noise law.
            # An externally supplied incomplete measure cannot acquire this
            # certificate merely from its remaining marginal covariances.
            evidence['transition_noise_contract']={'kind':'independent_position_diffusion',
                'variance_rate':.0025,'source':'complete prepared measure before any pruning',
                'model_version':self.model_version}
        return b.fork(evidence=evidence)

    @_serialized
    def initialize(self, legal_history, observed_graph):
        if not legal_history:
            raise ValueError('empty legal history')
        for row in legal_history:
            _finite_time(row['timestamp']); _pose(row['robot_pose'])
        _finite_time(observed_graph['timestamp'])
        b = super().initialize(legal_history, observed_graph)
        contract=b.evidence.get('transition_noise_contract',{})
        if self.kind=='structured' and contract.get('kind')=='independent_position_diffusion' and all(h.relaxation>0 for h in b.hypotheses if h.weight>0):
            b=b.fork(evidence={**b.evidence,'transition_noise_contract':{
                **contract,'model_version':self.model_version,
                'source':'history-filter certificate transferred across the identical positive-relaxation position-noise kernel'}})
        return StructuralBeliefState.wrap(b).fork(evidence={**b.evidence,'belief_role':'actual_continuous'})

    @_serialized
    def propagate(self, belief, timestamp):
        t = _finite_time(timestamp)
        self._check_belief(belief)
        if t < belief.timestamp - 1e-9:
            raise ValueError('time reversal')
        check_budget('propagate:entry')
        if belief.terminal:
            return belief
        carrier = getattr(belief, 'carrier', None)
        if carrier is not None and t > belief.timestamp + 1e-10:
            i = int(np.searchsorted(carrier.times, t - 1e-9))
            if i < len(carrier.times) and abs(carrier.times[i] - t) < 1e-8:
                return self._carrier_endpoint(belief, carrier, i, belief.robot_pose)
            if t < carrier.times[-1] - 1e-8:
                raise UnsupportedConditioning('requested clock is absent from the finite path carrier; a stochastic bridge is required')
            # Beyond its stored interval, start the same analytic transition
            # kernel at the final finite conditional state. No smoothing floor.
            belief = self._carrier_endpoint(belief, carrier, len(carrier.times)-1, belief.robot_pose)
            belief = belief.fork(carrier=None)
        b = super().propagate(belief, t)
        check_budget('propagate:complete')
        return StructuralBeliefState.wrap(b)

    def _apply_execution(self, belief, trace, timestamp):
        if not trace:
            return belief
        trace = dict(trace)
        payload = {k: v for k,v in trace.items() if k != 'evidence_id'}
        signature = _digest(payload)
        identity = str(trace.get('evidence_id', _digest({'trace':payload,'endpoint':timestamp})))
        if len(identity)>256:raise ValueError('execution event ID exceeds bounded metadata contract')
        ledger = dict(belief.evidence.get('execution_evidence_ledger', {}))
        if identity in ledger:
            if ledger[identity] != signature:
                raise ValueError('conflicting execution evidence under the same evidence_id')
            return belief
        if timestamp<=float(belief.evidence.get('execution_ledger_sealed_through',-math.inf)):
            raise UnsupportedConditioning('execution evidence ledger capacity reached at this clock; cannot safely identify an old replay')
        if 'start_timestamp' in trace and _finite_time(trace['start_timestamp']) > timestamp:
            raise ValueError('execution begins after arrived observation')
        if 'end_timestamp' in trace and abs(_finite_time(trace['end_timestamp']) - timestamp) > 1e-8:
            raise ValueError('execution end and arrived observation clocks disagree')
        b = belief
        if 'world_likelihood' in trace:
            if trace.get('support_fingerprint') != belief.fingerprint():
                raise ValueError('execution likelihood is not bound to this prior support')
            likelihood = np.asarray(trace['world_likelihood'], float)
            if likelihood.shape != (len(belief.hypotheses),) or not np.isfinite(likelihood).all() or np.any((likelihood<0)|(likelihood>1)):
                raise ValueError('invalid component survival likelihood')
            residual_likelihood = float(trace.get('unresolved_likelihood', 1.))
            if not math.isfinite(residual_likelihood) or not 0 <= residual_likelihood <= 1:
                raise ValueError('invalid residual survival likelihood')
            from .structural_probability import bayes_submeasure, EvidenceBound, ObservationMeasure
            measure = ObservationMeasure()
            update = bayes_submeasure([h.weight for h in belief.hypotheses],
                np.log(np.maximum(likelihood, 1e-300)) if np.all(likelihood>0) else
                np.array([math.log(v) if v else -math.inf for v in likelihood]),
                belief.unresolved_mass, EvidenceBound.event(upper=residual_likelihood,
                    provenance='caller-supplied residual likelihood upper bound'),
                component_measure=measure,
                approximation_notes=('incoming Gaussian projections are not globally certified',))
            b = b.fork(hypotheses=[h.fork(weight=float(w)) for h,w in zip(b.hypotheses,update.retained_weights)],
                       unresolved_mass=update.unresolved_mass,
                       evidence={**b.evidence, 'execution_probability_accounting':update.metadata()})
            if getattr(b, 'carrier', None) is not None:
                c = b.carrier
                b = b.fork(carrier=replace(c, weights=immutable_array(update.retained_weights),
                                           unresolved_mass=update.unresolved_mass))
        # Reject replay outside the bounded history rather than silently repeat
        # old evidence. Explicit event IDs must be unique within this session.
        ledger[identity] = signature
        if len(ledger)>256:
            oldest = next(iter(ledger)); ledger.pop(oldest)
            # Replaying an evicted same-clock ID cannot be distinguished from
            # new evidence with bounded storage. Refuse further trace events
            # at this clock rather than silently make it independent again.
            sealed_through=timestamp
        else:sealed_through=b.evidence.get('execution_ledger_sealed_through')
        evidence={**b.evidence,'execution_evidence_ledger':ledger}
        if sealed_through is not None:evidence['execution_ledger_sealed_through']=sealed_through
        return b.fork(evidence=evidence)

    @_serialized
    def observe(self, belief, actual_observation, actual_execution_trace=None):
        started = time.perf_counter()
        self._check_belief(belief)
        row = dict(actual_observation)
        t = _finite_time(row['timestamp']); pose = _pose(row['robot_pose'])
        if t < belief.timestamp - 1e-8:
            raise ValueError('old observation')
        if belief.terminal:
            return belief
        if row.get('observation_valid', True):
            _validate_static_arrival(row.get('static_updates', []), t)
        check_budget('observe:entry')
        prior_time = belief.timestamp
        b = self._apply_execution(belief, actual_execution_trace, t)
        b = self.propagate(b, t)
        graph = b.graph.update(row.get('static_updates', []) if row.get('observation_valid',True) else [])
        b = StructuralBeliefState.wrap(b).fork(graph=graph, robot_pose=pose.copy())
        valid = bool(row.get('observation_valid', True) and row.get('human_sensor_available', True))
        if valid:
            positions=[]
            for r in row.get('tracks', []):
                if r.get('visible_now') and r.get('valid',True):
                    if abs(_finite_time(r['last_seen_time'])-t)>1e-6:
                        raise ValueError('visible detection clock')
                    z=np.asarray(r['position'],float)
                    if z.shape!=(3,) or not np.isfinite(z).all():
                        raise ValueError('detection must have three finite coordinates')
                    positions.append(tuple(z))
            boxes = sorted(set(tuple(np.asarray(box,float).ravel()) for box in row.get('certified_visible_empty_boxes',[])))
            for box in boxes:
                a=np.asarray(box)
                if a.shape!=(4,) or not np.isfinite(a).all() or np.any(a[:2]>a[2:]):
                    raise ValueError('invalid certified-empty rectangle')
            payload={'timestamp':t,'robot_pose':pose.tolist(),'positions':sorted(positions),'empty_boxes':boxes}
            signature=_digest(payload)
            identity=str(row.get('human_observation_id', 'endpoint:'+repr(t)))
            if len(identity)>256:raise ValueError('human event ID exceeds bounded metadata contract')
            ledger = dict(b.evidence.get('human_evidence_ledger', {})) if b.evidence.get('human_evidence_timestamp')==t else {}
            if identity in ledger and ledger[identity]!=signature:
                raise ValueError('conflicting human evidence at one endpoint; provide a distinct physical observation event ID')
            if identity not in ledger:
                if len(ledger)>=32:
                    raise UnsupportedConditioning('too many human exposure IDs at one clock',status='UNSUPPORTED_EVIDENCE_CAPACITY')
                if ledger and signature not in ledger.values():
                    raise ValueError('one endpoint has conflicting exact human frames')
                if signature not in ledger.values():
                    if b.carrier is not None:
                        if positions or boxes:
                            raise UnsupportedConditioning('finite coarse successor cannot be refined to a continuous exact set/certificate without a joint-law bridge')
                        b = self._observe_carrier_empty(b, pose, identity)
                    else:
                        from .structural_filter import observe_sensor
                        b = observe_sensor(self, b, row, elapsed=t-prior_time)
                ledger[identity]=signature
                b=b.fork(evidence={**b.evidence,'human_evidence_ledger':ledger,'human_evidence_timestamp':t})
        else:
            b=b.fork(evidence={**b.evidence,'negative_evidence_available':False,
                               'human_sensor_status':'invalid_or_unavailable; no negative evidence'})
        if actual_execution_trace and actual_execution_trace.get('episode_terminated'):
            b=b.fork(terminal=True,query_cursor=None,
                terminal_event={'timestamp':t,'pose':pose.tolist(),'semantics':'actual arrived terminal endpoint'})
        b.validate()
        check_budget('observe:complete')
        self.timings['observe']=(time.perf_counter()-started)*1000.
        return b

    def _observe_carrier_empty(self, b, pose, identity):
        c=b.carrier
        event_key=('actual_no_detection',b.timestamp,identity)
        if event_key in c.observation_ledger:
            return b
        likelihood=None
        for event in reversed(c.observation_ledger):
            if event[0] not in ('survived_no_detection','survived_detection'):
                continue
            if abs(event[1]-b.timestamp)<1e-8 and np.array_equal(np.asarray(event[2]),pose):
                # These refer to one physical endpoint emission. An exact
                # empty endpoint refines its coarsened prefix observation.
                likelihood=np.ones(len(c.weights)) if event[0]=='survived_no_detection' else np.asarray(event[3],float)
                break
        if likelihood is None:
            likelihood=np.array([np.prod(1-self.observation_law.detection(
                c.positions[w,c.index,:h.n+int(round(h.expected_hidden))],pose,h.visibility_mode))
                for w,h in enumerate(c.worlds)])
        ws=c.weights*likelihood;den=float(ws.sum()+c.unresolved_mass)
        if den<=0:
            raise UnsupportedConditioning('zero-probability empty observation under finite carrier')
        c=replace(c,weights=immutable_array(ws/den),unresolved_mass=c.unresolved_mass/den,
                  observation_ledger=c.observation_ledger+(event_key,))
        return self._carrier_endpoint(b,c,c.index,pose)

    @_serialized
    def predict_shared(self, belief, horizon, num_worlds=16, seed=0, times=None):
        horizon=_finite_time(horizon,'horizon')
        if horizon<0 or horizon>30 or isinstance(num_worlds,bool) or int(num_worlds)!=num_worlds or not 1<=int(num_worlds)<=256:
            raise ValueError('finite horizon [0,30] and integer world count [1,256] required')
        if times is not None:
            rel=np.asarray(times,float)
            if rel.ndim!=1 or not len(rel) or not np.isfinite(rel).all() or rel[0]!=0 or np.any(np.diff(rel)<=0) or rel[-1]>horizon+1e-9 or rel[-1]<horizon-1e-9:
                raise ValueError('explicit clock must be finite, strictly increasing, start at zero and end at horizon')
        self._check_belief(belief);check_budget('forecast:entry')
        carrier=getattr(belief,'carrier',None)
        if carrier is not None and not belief.terminal:
            f=self._predict_carrier(belief,horizon,int(num_worlds),seed,times)
        else:
            f=super().predict_shared(belief,horizon,int(num_worlds),seed,times)
            # Sum the genuinely omitted components. Subtracting two differently
            # ordered floating-point sums can manufacture residual mass even
            # when every count component was retained.
            hs=f.source.hypotheses
            positive=[i for i,h in enumerate(hs) if h.weight>0]
            selected=set(sorted(positive,key=lambda i:-hs[i].weight)[:int(num_worlds)])
            f.unresolved_mass=min(1.,f.source.unresolved_mass+sum(h.weight for i,h in enumerate(hs) if i not in selected))
            f.continuous_source=belief
            f.rng_state=copy.deepcopy(getattr(self,'_last_path_rng_state',None))
            # Copy only the small immutable world templates; do not make the
            # caller's hypothesis buffers read-only as a side effect.
            f.worlds=[freeze_hypothesis(h) for h in f.worlds]
        f.generation_source_fingerprint=f.source.fingerprint()
        check_budget('forecast:complete')
        return seal_future(f)

    def sample_maneuver_paths(self,worlds,rel,start,rng,n):
        from .structural_dynamics import sample_maneuver_paths
        result=sample_maneuver_paths(self,worlds,rel,start,rng,n)
        self._last_path_rng_state=copy.deepcopy(rng.bit_generator.state)
        return result

    def sample_initial(self,h,rng):
        from .structural_dynamics import symmetric_gaussian_sample
        if h.maneuver_mean is not None and len(h.maneuver_mean)==h.n:
            k=(np.cumsum(h.maneuver_probability,axis=1)<rng.random((h.n,1))).sum(1)
            ii=np.arange(h.n);mean=h.maneuver_mean[ii,k];cov=h.maneuver_covariance[ii,k]
        else:mean=h.mean;cov=h.covariance
        return symmetric_gaussian_sample(mean,cov,rng)

    def _predict_carrier(self,b,horizon,num_worlds,seed,times):
        started=time.perf_counter();c=b.carrier
        if num_worlds!=len(c.weights):
            raise ValueError('changing the finite carrier world count requires an explicit resampling approximation')
        rel=np.asarray(times if times is not None else np.unique(np.r_[0.,np.arange(1,int(math.floor(horizon/NATIVE_DT))+1)*NATIVE_DT,horizon]),float)
        requested=b.timestamp+rel
        old_end=float(c.times[-1]);extension=requested[requested>old_end+1e-8]
        ts=c.times;pos=c.positions;vel=c.velocities;worlds=c.worlds
        if len(extension):
            last=self._carrier_endpoint(b,c,len(ts)-1,b.robot_pose)
            extrel=np.r_[0.,extension-old_end]
            rng=np.random.default_rng()
            if c.rng_state is None:
                raise UnsupportedConditioning('carrier lacks a continuation RNG state; replay from its originating legal belief')
            rng.bit_generator.state=json.loads(c.rng_state) if isinstance(c.rng_state,str) else copy.deepcopy(c.rng_state)
            p,v,ww=self.sample_maneuver_paths(last.hypotheses,extrel,old_end,rng,max(h.n for h in worlds))
            # Existing tape is never regenerated when another route is tried.
            ts=np.r_[ts,extension];pos=np.concatenate([pos,p[:,1:]],axis=1);vel=np.concatenate([vel,v[:,1:]],axis=1);worlds=tuple(ww)
        ids=np.searchsorted(ts,requested-1e-9)
        if np.any(ids>=len(ts)) or np.max(np.abs(ts[ids]-requested))>1e-8:
            raise UnsupportedConditioning('requested endpoint absent from finite carrier; conditional interpolation requires a bridge')
        f=SharedFuture(requested,pos[:,ids],vel[:,ids],c.weights,list(worlds),b,c.unresolved_mass,seed,b.fingerprint())
        f.carrier_origin=c.origin_fingerprint
        f.carrier_ledger=c.observation_ledger
        f.rng_state=copy.deepcopy(self._last_path_rng_state if len(extension) else c.rng_state)
        self.timings['forecast']=(time.perf_counter()-started)*1000.
        return f

    def _carrier_endpoint(self,source,c,index,pose,terminal=False,cursor=None):
        # Templates carry count/mode/ancestry; positions are exact quadrature
        # support points. Zero covariance is intentional, not an exact sensor.
        key=(c.tape_digest,int(index))
        templates=self.endpoint_template_cache.get(key)
        if templates is None:
            templates=[]
            until=(math.floor((c.times[index]-1e-9)/NATIVE_DT)+1)*NATIVE_DT
            states=np.concatenate([c.positions[:,index],c.velocities[:,index]],axis=-1)
            for w,h in enumerate(c.worlds):
                k=int(round(h.expected_hidden))
                changes=dict(mean=immutable_array(states[w,:h.n]),covariance=immutable_array(np.zeros((h.n,4,4))),
                    maneuver_mean=None,maneuver_covariance=None,maneuver_probability=None,maneuver_until=until)
                if h.hidden_mean is not None:
                    changes.update(hidden_mean=immutable_array(states[w,h.n:h.n+k]),
                        hidden_covariance=immutable_array(np.zeros((k,4,4))),hidden_exchangeable=False,
                        hidden_maneuver_mean=None,hidden_maneuver_covariance=None,hidden_maneuver_probability=None)
                templates.append(freeze_hypothesis(h.fork(**changes)))
            templates=tuple(templates);self.endpoint_template_cache[key]=templates
            def cache_bytes():
                return sum(sum(v.nbytes for h in hs for v in vars(h).values() if isinstance(v,np.ndarray))
                           for hs in self.endpoint_template_cache.values())
            while len(self.endpoint_template_cache)>64 or cache_bytes()>8*1024*1024:
                self.endpoint_template_cache.popitem(last=False)
        else:self.endpoint_template_cache.move_to_end(key)
        hs=[h.fork(weight=float(c.weights[w])) for w,h in enumerate(templates)]
        ledger=tuple(event for event in c.observation_ledger if len(event)>1 and event[1]>=float(c.times[index])-1e-8)
        return StructuralBeliefState(source.graph,hs,float(c.times[index]),np.array(pose,copy=True),
            c.unresolved_mass,terminal,self.model_version,
            evidence={**copy.deepcopy(source.evidence),'continuous_bridge_supported':False,
                      'belief_role':'imagined_finite',
                      'successor_measure':'finite joint path quadrature; no endpoint smoothing'},
            query_cursor=copy.deepcopy(cursor if cursor is not None else source.query_cursor),
            carrier=replace(c,index=int(index),observation_ledger=ledger))

    @_serialized
    def successor_at(self,future,index,world_weights,robot_pose,terminal=False,cursor=None,refinement=None):
        self._check_belief(future.source)
        if _VALIDATED_FUTURE.get() is not future:self._validate_future_source(future)
        if future.source.model_version not in ('unbound',self.model_version):
            raise ValueError('foreign model future')
        if isinstance(index,bool) or int(index)!=index or not 0<=int(index)<len(future.times):
            raise ValueError('invalid successor index')
        index=int(index);pose=_pose(robot_pose);weights=np.asarray(world_weights,float)
        if weights.shape!=future.weights.shape or not np.isfinite(weights).all() or np.any(weights<0) or np.any(weights>future.weights+1e-12):
            raise ValueError('branch weights must be finite joint masses bounded by source weights')
        mass=float(weights.sum())
        if mass<=0:
            raise ValueError('zero-probability branch has no defined conditional belief')
        check_budget('successor:entry')
        continuous=getattr(future,'continuous_source',None)
        if continuous is not None and not terminal and refinement is None and np.array_equal(weights,future.weights):
            # An unobserved action boundary conveys no state information and
            # must not insert quadrature or endpoint regularization into T.
            b=self.propagate(continuous,float(future.times[index]))
            return b.fork(robot_pose=pose.copy(),query_cursor=cursor,
                          evidence={**b.evidence,'belief_role':'imagined_analytic'})
        residual=float(future.unresolved_mass)
        normalizer=mass+residual
        c=WorldCarrier(future.times,future.positions,future.velocities,tuple(future.worlds),
            immutable_array(weights/normalizer),residual/normalizer,future.seed,
            getattr(future,'carrier_origin',future.input_fingerprint),index,
            tuple(getattr(future,'carrier_ledger',())),getattr(future,'tape_digest',None),
            copy.deepcopy(getattr(future,'rng_state',None)),getattr(future,'_tape_validation',None))
        future.tape_digest=c.tape_digest
        future._tape_validation=c._tape_validation
        return self._carrier_endpoint(future.source,c,index,pose,terminal,cursor)

    def _validate_future_source(self,future):
        fingerprint=future.source.fingerprint()
        if getattr(future,'generation_source_fingerprint',fingerprint)!=fingerprint:
            raise ValueError('shared future source was mutated after generation')
        original=getattr(future,'continuous_source',None)
        if original is not None and original.fingerprint()!=future.input_fingerprint:
            raise ValueError('continuous source was mutated after generation')

    @_serialized
    def evaluate_queries(self,belief,queries,shared_future,executor=EXECUTOR_VERSION,budget_ms=40.,refined_queries=4,refined_query_indices=None):
        self._check_belief(belief)
        if shared_future.source.model_version!=self.model_version:
            raise ValueError('shared future generated by a different model version')
        if isinstance(refined_queries,bool) or int(refined_queries)!=refined_queries or not 0<=refined_queries<=4:
            raise ValueError('refined_queries must be an integer in [0,4]')
        if refined_query_indices is not None:
            chosen = tuple(refined_query_indices)
            if (len(chosen)>4 or len(set(chosen))!=len(chosen) or
                    any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<len(queries) for i in chosen)):
                raise ValueError('choose at most four distinct candidate row indices')
            refined_query_indices=chosen
        deadline=Deadline.create(budget_ms)
        token=None
        if ACTIVE_DEADLINE.get() is None:
            token=ACTIVE_DEADLINE.set(deadline)
        future_token=None
        try:
            self._validate_future_source(shared_future)
            future_token=_VALIDATED_FUTURE.set(shared_future)
            from .structural_queries import evaluate
            return evaluate(self,belief,queries,shared_future,executor,budget_ms,refined_queries,refined_query_indices)
        finally:
            if future_token is not None:_VALIDATED_FUTURE.reset(future_token)
            if token is not None:ACTIVE_DEADLINE.reset(token)

    @_serialized
    def continue_rollout(self,successor_belief,query,budget_ms=40.,horizon=3.,seed=0):
        carrier=getattr(successor_belief,'carrier',None)
        worlds=len(carrier.weights) if carrier is not None else 16
        check_budget('continue:entry')
        future=self.predict_shared(successor_belief,horizon,num_worlds=worlds,seed=seed)
        batch=self.evaluate_queries(successor_belief,[query],future,budget_ms=budget_ms)
        check_budget('continue:complete')
        return batch.predictions[0]

    @_serialized
    def online_call(self,belief,actual_observation,queries,horizon=3.,num_worlds=16,seed=0,budget_ms=None):
        start=time.perf_counter();token=None
        if budget_ms is not None and ACTIVE_DEADLINE.get() is None:
            token=ACTIVE_DEADLINE.set(Deadline.create(budget_ms,pending_observation=True))
        try:
            self.timings={}
            b=self.observe(belief,actual_observation)
            active=ACTIVE_DEADLINE.get()
            # Whole-call publication is atomic. Even after local assimilation,
            # a later timeout leaves the caller's observation pending.
            f=self.predict_shared(b,horizon,num_worlds,seed)
            # None means deliberately unbounded measurement, not a 40ms
            # inner deadline silently excluding the observe/forecast path.
            batch=self.evaluate_queries(b,queries,f,budget_ms=1e12 if budget_ms is None else budget_ms)
            check_budget('online:complete')
            batch.timings_ms={**self.timings,**batch.timings_ms,'complete':(time.perf_counter()-start)*1000.}
            return b,batch
        finally:
            if token is not None:ACTIVE_DEADLINE.reset(token)
