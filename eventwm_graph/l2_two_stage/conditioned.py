"""History-conditioned human population proposal and exact rejection sampler.

The graph/motion neural proposal is unchanged. This module conditions its
CURRENT physical population on every actual positive observation, then draws
coherent future marks. Tracker keys are proposal labels, never physical IDs.
No hypothetical observations, Queries or privileged labels are accepted.
"""
from dataclasses import dataclass
import hashlib
import time
import numpy as np
from .count import sample_conditioned,sample_total_count,certain_sources,count_log_probability
from .model import Stage1WorldModel,softmax
from .neural import KNOTS
from .speed import bound_numpy
from .types import JointHumanWorlds,immutable_array,SealedObservation
from .context import OnlineContext
from .positive_history import PositiveSetHistory,positive_xyz
from .association import AssociationGraph
from types import MappingProxyType


class PopulationSamplingFailure(RuntimeError):
    """Explicit unsuccessful prediction; never a partial or zero-risk forecast."""


def history_from_context(context):
    context.validate_certificate()
    frames=context.history;points=[positive_xyz(frame) for frame in frames]
    return PositiveSetHistory('','','',len(frames)-1,
        np.array([frame['timestamp'] for frame in frames],np.float64),
        np.r_[0,np.cumsum([len(p) for p in points])],np.concatenate(points,axis=0),
        np.array([frame.get('observation_valid',True) for frame in frames]),
        np.array([frame.get('human_sensor_available',True) for frame in frames]))


def source_coordinates(features,history):
    """Recover exact legal atoms, preserving co-located observation multiplicity.

    Prepared float32 centers select a recorded atom; they are never treated as
    exact same-clock observation coordinates. Public calls supply raw last
    positions directly. Neither physical identity nor future truth is consulted.
    """
    event_times=np.repeat(history.timestamps,np.diff(history.positive_offsets))
    points=history.positive_xyz[:,[0,2]]
    if 'source_last_event_indices' in features:
        indices=np.asarray(features['source_last_event_indices'],int)
        if indices.shape!=(len(features['keys']),) or np.any(indices<0) or np.any(indices>=len(points)):
            raise ValueError('invalid legal source-event references')
        if (np.max(abs(event_times[indices]-(features['timestamp']-features['age'])),initial=0)>1e-7
                or len(set(indices.tolist()))!=len(indices)):
            raise ValueError('source-event references differ from actual last observations')
        if not np.array_equal(event_times[indices]==features['timestamp'],np.asarray(features['age'])==0):
            raise ValueError('source current/positive-age status differs from exact observation clocks')
        return points[indices],indices
    last_times=np.asarray(features.get('last_times',features['timestamp']-features['age']),float)
    approximate=np.asarray(features.get('last_positions',features['origin']-features['velocity']*features['age'][:,None]),float)
    ages=np.asarray(features['age'])
    if (last_times.shape!=(len(features['keys']),) or approximate.shape!=(len(features['keys']),2)
            or not np.isfinite(last_times).all() or not np.isfinite(approximate).all()
            or np.any(last_times>features['timestamp'])
            or not np.array_equal(last_times==features['timestamp'],ages==0)):
        raise ValueError('source recovery clocks/coordinates differ from current observation status')
    indices=[];used=set()
    for timestamp,point,age in zip(last_times,approximate,ages):
        candidates=np.flatnonzero(abs(event_times-timestamp)<1e-7)
        candidates=np.array([i for i in candidates if i not in used and
            bool(event_times[i]==features['timestamp'])==bool(age==0)],int)
        if not len(candidates):raise ValueError('proposal episode has no actual positive event')
        # Clock matching precedes coordinate matching, including co-located
        # events separated by a genuinely positive, very short native gap.
        delta=abs(event_times[candidates]-timestamp)
        candidates=candidates[delta==delta.min()]
        distance=np.linalg.norm(points[candidates]-point,axis=-1);at=int(candidates[np.argmin(distance)])
        if float(distance.min())>2e-4:raise ValueError('proposal source differs from actual recorded measurement')
        indices.append(at);used.add(at)
    indices=np.asarray(indices,int)
    if not np.array_equal(event_times[indices]==features['timestamp'],np.asarray(features['age'])==0):
        raise ValueError('source current/positive-age status differs from exact observation clocks')
    return points[indices],indices


@dataclass(frozen=True)
class CurrentProposal:
    mode: int
    source_present: np.ndarray
    initial_coefficients: np.ndarray
    hidden_components: np.ndarray
    hidden_positions: np.ndarray
    source_positions: np.ndarray

    @property
    def count(self):return int(self.source_present.sum()+len(self.hidden_positions))


def draw_current(features,parameters,raw_last,minimum,worlds,rng,unique=True,total_count=None):
    n=len(features['keys']);mode=rng.choice(4,worlds,p=softmax(parameters['mode_logits']))
    certain=certain_sources(features['age'],features['minimum_count'],unique,False)
    from .mixed_count import is_mixed,numpy_components
    from .tilted_count import is_tilted
    if is_tilted(parameters):
        from .tilted_count_sampler import PreparedTiltedCountSampler
        sampler=PreparedTiltedCountSampler(parameters['presence_logits'][:n],features['age'],float(parameters['rate']),
            minimum,parameters['total_count_tilt_logits'],certain=certain,total=total_count)
        present,counts=sampler.sample(worlds,rng)
    elif is_mixed(parameters):
        from .mixture_count_sampler import PreparedMixtureCountSampler
        rates,weights=numpy_components(parameters)
        sampler=PreparedMixtureCountSampler(parameters['presence_logits'][:n],features['age'],rates,weights,
            minimum,certain=certain,total=total_count)
        present,counts=sampler.sample(worlds,rng)
    elif total_count is None:
        present,counts,_=sample_conditioned(parameters['presence_logits'][:n],features['age'],float(parameters['rate']),minimum,worlds,rng,certain)
    else:
        if int(total_count)<minimum:raise ValueError('total below the historical minimum count')
        present,counts,_=sample_total_count(parameters['presence_logits'][:n],features['age'],float(parameters['rate']),int(total_count),worlds,rng,certain)
    noise=rng.normal(size=(worlds,n,2))
    initial=parameters['means'][:n][np.arange(n)[None],mode[:,None],0]+parameters['scales'][:n][np.arange(n)[None],mode[:,None],0]*noise
    velocity=bound_numpy(features['velocity'][None]+initial,3.)
    positions=raw_last[None]+velocity*features['age'][None,:,None]
    rows=[];spatial_cdf=np.cumsum(softmax(parameters['spatial_logits']));spatial_cdf/=spatial_cdf[-1]
    for w in range(worlds):
        count=int(counts[w]);component=np.searchsorted(spatial_cdf,rng.random(count),side='right')
        hidden=features['robotxy']+parameters['spatial_mean'][component]+parameters['spatial_scale'][component]*rng.normal(size=(count,2))
        rows.append(CurrentProposal(int(mode[w]),present[w],initial[w],component,hidden,positions[w]))
    return rows


def pad_current(rows,n):
    hidden=max((len(row.hidden_positions) for row in rows),default=0)
    positions=np.zeros((len(rows),n+hidden,2),np.float64);present=np.zeros((len(rows),n+hidden),bool)
    for i,row in enumerate(rows):
        positions[i,:n]=row.source_positions;positions[i,n:n+len(row.hidden_positions)]=row.hidden_positions
        present[i,:n]=row.source_present;present[i,n:n+len(row.hidden_positions)]=True
    return positions,present


def draw_conditioned_current(features,parameters,history,graph,worlds,rng,unique=True,max_proposals=1000000):
    from .vector_sampler import PreparedCurrentSampler
    raw_last,source_events=source_coordinates(features,history);n=len(raw_last)
    minimum=max(int(features['minimum_count']),int(graph.minimum_current_count))
    sampler=PreparedCurrentSampler(features,parameters,raw_last,minimum,unique,graph=graph)
    selected=[];proposed=0;examined=0
    while len(selected)<worlds:
        # Adaptive blocks only use previous accept/reject evidence. Each new
        # proposal remains iid Q, and we retain exactly its first K successes.
        # Avoid drawing32 extra populations when only one or two remain.
        acceptance=(len(selected)+1)/(proposed+2)
        count=min(worlds if proposed==0 else 1024,int(np.ceil((worlds-len(selected))/acceptance)),max_proposals-proposed)
        if count<=0:raise PopulationSamplingFailure('exact population conditioning exceeded declared resource allowance; no worlds returned')
        batch=sampler.draw(count,rng)
        accepted=sampler.accept(batch,graph,rng)
        indices=np.flatnonzero(accepted)[:worlds-len(selected)]
        selected.extend(batch.objects(indices))
        if len(selected)==worlds:examined=proposed+int(indices[-1])+1
        proposed+=count
    multiplier=float(np.exp(sampler.log_normalizer_multiplier))
    beta_estimate=(worlds-1)/(examined-1) if examined>1 else 1.
    diagnostics=dict(proposals_generated=proposed,proposals_examined_through_last_accept=examined,
        worlds=worlds,minimum_current_count=minimum,
        algorithm='vectorized_adaptive_block_exact_rejection_4; optional_exact_Palm_proposal; unbounded_tail_retained',
        proposal_count_offset=sampler.count_offset,log_normalizer_multiplier=sampler.log_normalizer_multiplier,
        compulsory_residual_event=None if sampler.required_rectangle is None else sampler.required_rectangle.event,
        proposal_acceptance_unbiased_sampling_estimate=beta_estimate,
        normalizer_unbiased_sampling_estimate=beta_estimate*multiplier if multiplier>0 else None,
        log_of_unbiased_normalizer_probability_estimate=float(np.log(beta_estimate)+sampler.log_normalizer_multiplier) if beta_estimate>0 else None,
        normalizer_warning='Negative-binomial diagnostic estimate, not an exact likelihood normalizer')
    from .mixed_count import is_mixed
    if is_mixed(parameters):
        diagnostics['residual_count_family']='finite_mixed_Poisson'
        diagnostics['residual_count_components']=len(parameters['residual_count_rates'])
    from .tilted_count import is_tilted
    if is_tilted(parameters):
        diagnostics['population_count_family']='total_count_tilt_Bernoulli_PPP'
        diagnostics['total_count_explicit_bins']=len(parameters['total_count_tilt_logits'])-1
    return selected,source_events,diagnostics


def construct_future(features,parameters,rows,times,rng):
    n=len(features['keys']);k=len(rows);current,present=pad_current(rows,n)
    mode=np.array([row.mode for row in rows]);initial=np.array([row.initial_coefficients for row in rows]).reshape(k,n,2)
    noise=rng.normal(size=(k,n,4,2));common=rng.normal(size=(k,1,4,2))
    coefficient=parameters['means'][:n][np.arange(n)[None],mode[:,None],1:]+parameters['scales'][:n][np.arange(n)[None],mode[:,None],1:]*(noise*.979795897+common*.2)
    velocity=bound_numpy(features['velocity'][None,:,None]+initial[:,:,None]+coefficient,3.)
    starts=np.array(KNOTS[:-1]);ends=np.array(KNOTS[1:])
    basis=np.minimum(np.maximum(times[:,None]-starts,0),ends-starts);basis[:,-1]=np.maximum(times-starts[-1],0)
    positions=np.zeros((k,len(times),current.shape[1],2),np.float64)
    increments=(basis@velocity.transpose(2,0,1,3).reshape(4,-1)).reshape(len(times),k,n,2).transpose(1,0,2,3)
    positions[:,:,:n]=current[:,None,:n]+increments
    # Concatenate only actual PPP marks in the original world/slot order.
    # The flat normal draw consumes exactly the reference loop's stream,
    # including zero draws for empty worlds. Padded slots remain zero.
    counts=np.array([len(row.hidden_components) for row in rows],int)
    total=int(counts.sum())
    if total:
        components=np.concatenate([row.hidden_components for row in rows])
        locations=np.concatenate([row.hidden_positions for row in rows])
        world=np.repeat(np.arange(k),counts)
        slot=np.arange(total)-np.repeat(np.cumsum(counts)-counts,counts)
        raw_velocity=parameters['spatial_velocity'][components]+parameters['spatial_velocity_scale'][components,None]*rng.normal(size=(total,2))
        positions[world,:,n+slot]=locations[:,None]+times[None,:,None]*bound_numpy(raw_velocity,3.)[:,None]
    return positions,present


@dataclass(frozen=True)
class HistoryConditionedLaw:
    """Exact definition q(W)1[feasible(H,W0)]/Pr_q(feasible).

    The normalizer is an integral, not the old unconditioned count formula.
    This class deliberately refuses to mislabel that old formula as its NLL.
    """
    parameters: object
    features: object
    history: PositiveSetHistory
    graph: AssociationGraph
    unique: bool
    association_guidance: bool = False
    version: str = 'history_conditioned_joint_human_law_1'

    @property
    def residual_count_parameters(self):
        from .tilted_count import is_tilted,numpy_weights
        if is_tilted(self.parameters):
            return MappingProxyType(dict(version='total_cardinality_tilt_proposal_1',family='total_count_tilt_Bernoulli_PPP',
                base_PPP_rate=float(self.parameters['rate']),log_potentials=immutable_array(numpy_weights(self.parameters)),
                cutoff=len(self.parameters['total_count_tilt_logits'])-1,
                semantics='Potentials weight TOTAL source-plus-residual count. Last potential applies to every count >=cutoff, with unbounded base tail. These are not normalized count probabilities. Full-history conditioning changes the final law.'))
        from .mixed_count import numpy_components,is_mixed
        rates,log_weights=numpy_components(self.parameters)
        return MappingProxyType(dict(version='residual_cardinality_proposal_1',
            family='finite_mixed_Poisson' if is_mixed(self.parameters) else 'Poisson',
            rates=immutable_array(rates),weights=immutable_array(np.exp(log_weights)),
            semantics='Unconditioned residual proposal cardinality; legal floor/full-history conditioning changes the final count law'))

    def count_log_probability(self,count):
        raise NotImplementedError('History conditioning changes count probabilities; use an explicitly bounded integration estimate, not proposal count NLL')

    def estimate_count_log_probability(self,count,successes=32,seed=731,max_proposals=1000000):
        """Unbiased log-count score via independent inverse-binomial streams.

        For N~NegBin(k,p), E[psi(k)-psi(N)]=log(p) and variance is at
        most trigamma(k). Values are NEVER clipped, even if a finite noisy NLL
        happens to be negative. Exhaustion is explicit failure, never censoring.
        """
        from scipy.special import digamma,polygamma
        for value in (count,successes):
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0:
                raise ValueError('integer nonnegative count and fixed-success budget required')
        count=int(count);successes=int(successes)
        if successes<2:raise ValueError('at least2 fixed successes required')
        raw_last,_=source_coordinates(self.features,self.history)
        minimum=max(self.features['minimum_count'],self.graph.minimum_current_count)
        if count<minimum:return dict(log_probability=-np.inf,status='ZERO_PROPOSAL_COUNT_SUPPORT',successes=successes)
        certain=certain_sources(self.features['age'],self.features['minimum_count'],self.unique,False)
        from .vector_sampler import PreparedCurrentSampler
        proposal=PreparedCurrentSampler(self.features,self.parameters,raw_last,minimum,self.unique,graph=self.graph)
        from .mixed_count import is_mixed,numpy_count_log_probability
        from .tilted_count import is_tilted,numpy_count_log_probability as tilted_count_log_probability
        if is_tilted(self.parameters):
            qlog=tilted_count_log_probability(self.parameters,self.parameters['presence_logits'][:len(raw_last)],
                self.features['age'],count-proposal.count_offset,minimum-proposal.count_offset,certain,shift=proposal.count_offset)
        elif is_mixed(self.parameters):
            qlog=numpy_count_log_probability(self.parameters,self.parameters['presence_logits'][:len(raw_last)],
                self.features['age'],count-proposal.count_offset,minimum-proposal.count_offset,certain,palm=bool(proposal.count_offset))
        else:
            qlog=count_log_probability(self.parameters['presence_logits'][:len(raw_last)],self.features['age'],
                float(self.parameters['rate']),count-proposal.count_offset,minimum-proposal.count_offset,certain)
        if not np.isfinite(qlog):return dict(log_probability=-np.inf,status='ZERO_PROPOSAL_COUNT_SUPPORT',successes=successes)
        def integrate(total,stream):
            from .vector_sampler import PreparedCurrentSampler
            sampler=PreparedCurrentSampler(self.features,self.parameters,raw_last,minimum,self.unique,total,graph=self.graph)
            rng=np.random.default_rng(np.random.SeedSequence([seed,stream]));accepted=0;generated=0
            while accepted<successes:
                size=min(max(32,successes-accepted),max_proposals-generated)
                if size<=0:raise PopulationSamplingFailure('inverse-binomial integration exhausted; no metric returned')
                batch=sampler.draw(size,rng);valid=sampler.accept(batch,self.graph,rng)
                hits=np.flatnonzero(valid)
                if accepted+len(hits)>=successes:
                    waiting=generated+int(hits[successes-accepted-1])+1
                    return float(digamma(successes)-digamma(waiting)),waiting
                accepted+=len(hits);generated+=size
        logz,nz=integrate(None,0);loga,na=integrate(count,1)
        return dict(log_probability=float(qlog+loga-logz),status='UNBIASED_LOG_SCORE_ESTIMATE',
            proposal_log_count_probability=float(qlog),log_normalizer_estimate=logz+proposal.log_normalizer_multiplier,
            proposal_acceptance_log_estimate=logz,log_normalizer_multiplier=proposal.log_normalizer_multiplier,
            proposal_count_offset=proposal.count_offset,
            conditional_acceptance_log_estimate=loga,normalizer_waiting=nz,count_waiting=na,
            successes=successes,variance_upper_bound=float(2*polygamma(1,successes)),
            interpretation='Monte Carlo estimator of the true conditional count log probability; independent fixed-success streams, no clipping')

    def estimate_count_probabilities(self,counts,proposals=4096,seed=731,confidence=.99):
        """Offline binomial bounds on joint numerator / acceptance normalizer.

        Bonferroni Clopper-Pearson intervals give a simultaneous ratio bound.
        Zero Monte Carlo hits NEVER mean certified zero probability.
        """
        from scipy.stats import beta
        counts=tuple(int(c) for c in counts)
        if proposals<2 or not 0<confidence<1:raise ValueError('invalid integration request')
        rng=np.random.default_rng(seed);raw_last,_=source_coordinates(self.features,self.history)
        minimum=max(self.features['minimum_count'],self.graph.minimum_current_count)
        from .vector_sampler import PreparedCurrentSampler
        sampler=PreparedCurrentSampler(self.features,self.parameters,raw_last,minimum,self.unique,graph=self.graph)
        accepted=0;hits=np.zeros(len(counts),int);generated=0
        while generated<proposals:
            batch=sampler.draw(min(1024,proposals-generated),rng);valid=sampler.accept(batch,self.graph,rng)
            population=batch.present.sum(-1);accepted+=int(valid.sum())
            for i,count in enumerate(counts):hits[i]+=int((valid&(population==count)).sum())
            generated+=len(batch.present)
        alpha=(1-confidence)/(len(counts)+1)
        def interval(hit):
            return (0. if hit==0 else float(beta.ppf(alpha/2,hit,proposals-hit+1)),
                    1. if hit==proposals else float(beta.ppf(1-alpha/2,hit+1,proposals-hit)))
        zlo,zhi=interval(accepted);results={}
        for count,hit in zip(counts,hits):
            lo,hi=interval(int(hit));results[count]=dict(estimate=float(hit/accepted) if accepted else None,
                lower=max(0.,lo/zhi),upper=min(1.,hi/zlo) if zlo else 1.,numerator_hits=int(hit))
        multiplier=float(np.exp(sampler.log_normalizer_multiplier))
        return dict(schema='conditional_count_integration_2',proposals=proposals,accepted=accepted,
            confidence=confidence,normalizer_interval=[zlo*multiplier,zhi*multiplier] if multiplier>0 else None,
            log_normalizer_interval=[float(np.log(zlo)+sampler.log_normalizer_multiplier) if zlo else -np.inf,
                                     float(np.log(zhi)+sampler.log_normalizer_multiplier)],
            proposal_acceptance_interval=[zlo,zhi],log_normalizer_multiplier=sampler.log_normalizer_multiplier,
            proposal_count_offset=sampler.count_offset,probabilities=results,
            method='simultaneous binomial numerator/normalizer confidence bounds; not exact point probabilities')


@dataclass(frozen=True)
class ConditionedHumanWorlds(JointHumanWorlds):
    association: object = None
    source_event_indices: object = None
    previously_observed: object = None
    sampling: object = None
    unseen_semantics: str = 'proposal-residual-slot marker only; use world-specific previously_observed for physical ancestry'
    version: str = 'history_conditioned_joint_human_worlds_2'

    def positions_at(self,relative_times):
        from .time_grid import interpolate_positions
        return interpolate_positions(self,relative_times)

    @property
    def residual_count_parameters(self):return self.law.residual_count_parameters


class HistoryConditionedWorldModel(Stage1WorldModel):
    def __init__(self,checkpoint=None,network=None,max_proposals=1000000,history_fraction=None,association_guidance=None):
        super().__init__(checkpoint,network)
        saved=None
        if checkpoint is not None:
            import torch
            saved=torch.load(checkpoint,map_location='cpu')
            if 'conditioned_config' in saved:
                configured=saved['conditioned_config']['history_fraction']
                if history_fraction is not None and history_fraction!=configured:
                    raise ValueError('cannot silently reinterpret a trained conditioned checkpoint with another spatial mixture')
                history_fraction=configured
                configured_guidance=saved['conditioned_config'].get('association_guidance',False)
                if association_guidance is not None and bool(association_guidance)!=configured_guidance:
                    raise ValueError('cannot silently reinterpret a trained identity-association law')
                association_guidance=configured_guidance
        self.history_fraction=0. if history_fraction is None else float(history_fraction)
        self.association_guidance=bool(association_guidance)
        if self.history_fraction and saved is not None and 'conditioned_config' in saved:
            from pathlib import Path
            from .common import sha
            expected=saved['config']['source_hashes']['history_proposal.py']
            if sha(Path(__file__).with_name('history_proposal.py'))!=expected:
                raise ValueError('trained spatial proposal source differs; use the checkpoint frozen source, or retrain and revalidate the changed law')
        if self.association_guidance and saved is not None and 'conditioned_config' in saved:
            from pathlib import Path
            from .common import sha
            if sha(Path(__file__).with_name('guided_association.py'))!=saved['config']['source_hashes'].get('guided_association.py'):
                raise ValueError('trained association source differs; use its frozen source or revalidate the changed law')
        if not (self.network.max_speed==3. and self.network.independent_initial and self.network.persistent_offset_velocity and self.network.tail=='gaussian'):
            raise ValueError('history conditioning requires the validated bounded Gaussian joint-motion proposal')
        self.max_proposals=int(max_proposals)

    def parameters(self,features,association_graph=None):
        if getattr(self.network,'residual_count_components',1)==1 and getattr(self.network,'total_count_tilt_cutoff',None) is None:
            return super().parameters(features)
        from .mixed_count import legal_context
        history=features.get('positive_history')
        if not isinstance(history,PositiveSetHistory):
            raise ValueError('mixed cardinality requires the complete actual positive history')
        graph=association_graph or AssociationGraph.from_history(history)
        return super().parameters(features,count_context=legal_context(features,history,graph))

    def predict(self,history,obs,partial_graph=None,horizon=3.,worlds=32,seed=None,actions=(),executed_action=None):
        if not 0<horizon<=3. or worlds<2:raise ValueError('supported horizon (0,3] and >=2 joint worlds required')
        start=time.perf_counter()
        if isinstance(history,OnlineContext):
            history=history.with_association()
            if float(obs['timestamp'])>history.history[-1]['timestamp']:
                context=history.ingest(obs,executed_action,partial_graph)
            else:
                arrived=SealedObservation.from_raw(obs)
                if arrived.record!=history.history[-1].record:raise ValueError('conflicting current observation')
                context=history if partial_graph is None else OnlineContext.from_inputs(history.legal_history,obs,partial_graph,history.actions)
        else:
            if partial_graph is None:raise ValueError('raw histories require the current legal graph')
            context=OnlineContext.from_inputs(history,obs,partial_graph,actions)
        from .features import extract
        context=context.with_association()
        features=extract(context);features['positive_history']=context._positive_history
        processed=time.perf_counter();result=self.predict_features(features,horizon,worlds,seed,association_graph=context._association_graph)
        self.last_profile.update(input_graph_history_ms=(processed-start)*1000,complete_ms=(time.perf_counter()-start)*1000)
        self.last_context=context
        return result

    def predict_features(self,features,horizon=3.,worlds=32,seed=None,times=None,association_graph=None):
        started=time.perf_counter();history=features.get('positive_history')
        if not isinstance(history,PositiveSetHistory) or float(history.timestamps[-1])!=float(features['timestamp']):
            raise ValueError('complete causal actual-positive history is required')
        graph=association_graph or AssociationGraph.from_history(history)
        from .history_proposal import augment_numpy
        parameters={key:np.asarray(value,np.float64) for key,value in self.parameters(features,association_graph=graph).items()}
        parameters=augment_numpy(parameters,features,graph,self.history_fraction);neural_done=time.perf_counter()
        if seed is None:
            digest=hashlib.sha256(b'conditioned-human-worlds-1')
            digest.update(history.timestamps.tobytes());digest.update(history.positive_xyz.tobytes())
            for name in ('x','origin','velocity','age','robotxy'):digest.update(np.ascontiguousarray(features[name]).tobytes())
            seed=int.from_bytes(digest.digest()[:8],'little')
        from .time_grid import prediction_grid
        rng=np.random.default_rng(seed);times=prediction_grid(horizon,64) if times is None else np.asarray(times,np.float64)
        if times[0]!=0 or times[-1]>3.028001 or np.any(np.diff(times)<=0):raise ValueError('invalid output times')
        rows,events,diagnostics=draw_conditioned_current(features,parameters,history,graph,worlds,rng,
            self.network.compulsory_unique_episode,self.max_proposals)
        conditioned=time.perf_counter();positions,present=construct_future(features,parameters,rows,times,rng)
        if self.association_guidance:
            from .guided_association import lineage as guided_lineage
            lineage=guided_lineage(graph,positions[:,0],present,events,positions[:,0,:len(features['keys'])],rng,source_first=True)
        else:lineage=graph.lineage(positions[:,0],present,rng)
        n=len(features['keys']);hidden=positions.shape[2]-n
        minimal={key:features[key] for key in ('keys','age','velocity','origin','robotxy','timestamp','minimum_count')}
        # Preserve the same exact measured atoms for all probability integrals.
        minimal['source_last_event_indices']=events
        frozen_features=MappingProxyType({key:immutable_array(value) if isinstance(value,np.ndarray) else value for key,value in minimal.items()})
        law=HistoryConditionedLaw(MappingProxyType({key:immutable_array(value) for key,value in parameters.items()}),
            frozen_features,history,graph,self.network.compulsory_unique_episode,association_guidance=self.association_guidance)
        from .mixed_count import is_mixed
        mixed=is_mixed(parameters)
        from .tilted_count import is_tilted
        tilted=is_tilted(parameters)
        if mixed:object.__setattr__(law,'version','history_conditioned_joint_human_law_2')
        if tilted:object.__setattr__(law,'version','history_conditioned_joint_human_law_3')
        result=ConditionedHumanWorlds(float(features['timestamp']),immutable_array(times),immutable_array(np.full(worlds,1/worlds)),
            immutable_array(positions),immutable_array(present),tuple('proposal_episode:'+str(k) for k in features['keys'])+
            tuple('proposal_residual:%d'%i for i in range(hidden)),immutable_array(np.r_[np.zeros(n,bool),np.ones(hidden,bool)]),
            float(parameters['rate']),('EXACT_HISTORICAL_POSITIVE_SET_SUPPORT','CONDITIONED_COUNTS_REQUIRE_NORMALIZER',
                'ASSOCIATION_KERNEL_NONUNIFORM_NOT_ORACLE_IDENTITY','UNKNOWN_GEOMETRY_UNCONSTRAINED_HUMAN_PATH_PRIOR',
                'CSPACE_OBSTRUCTIONS_ARE_NOT_HUMAN_MATERIAL_WALLS','MONTE_CARLO_BANK_NOT_SAFETY_CERTIFICATE'),
            minimum_count=graph.minimum_current_count,law=law,association=lineage,source_event_indices=immutable_array(events),
            previously_observed=lineage.physical_seen,sampling=MappingProxyType(diagnostics),
            version='history_conditioned_joint_human_worlds_4' if tilted else 'history_conditioned_joint_human_worlds_3' if mixed else 'history_conditioned_joint_human_worlds_2').validate()
        ended=time.perf_counter();self.last_profile=dict(model_ms=(neural_done-started)*1000,
            history_conditioning_ms=(conditioned-neural_done)*1000,distribution_ms=(ended-conditioned)*1000)
        return result
