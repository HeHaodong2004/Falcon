"""CPU public Human World Predictor. No target reader or candidate Query input."""
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from .types import LegalContext, JointHumanWorlds,JointHumanLaw,immutable_array,SealedObservation
from types import MappingProxyType
from .features import extract
from .neural import HumanGraphPredictor, KNOTS
from .count import sample_conditioned,certain_sources
from .context import OnlineContext


def softmax(x):
    a=np.exp(x-np.max(x));return a/a.sum()


class Stage1WorldModel:
    def __init__(self,checkpoint=None,network=None):
        if checkpoint is not None:
            saved=torch.load(Path(checkpoint),map_location='cpu')
            network=HumanGraphPredictor(**saved['model_config']);network.load_state_dict(saved['model'])
        self.network=(network or HumanGraphPredictor()).cpu().eval()
        self.last_profile={}
        self.last_context=None

    def predict(self,history,obs,partial_graph=None,horizon=3.,worlds=32,seed=None,actions=(),executed_action=None):
        if not 0<horizon<=3. or worlds<2:raise ValueError('supported horizon (0,3] and >=2 joint worlds required')
        start=time.perf_counter()
        if isinstance(history,OnlineContext):
            history.validate_certificate()
            if float(obs['timestamp'])>history.history[-1]['timestamp']:
                context=history.ingest(obs,executed_action,partial_graph)
            else:
                arrived=SealedObservation.from_raw(obs)
                if arrived.record!=history.history[-1].record:raise ValueError('conflicting current observation')
                context=history if partial_graph is None else OnlineContext.from_inputs(history.legal_history,obs,partial_graph,history.actions)
        else:
            if partial_graph is None:raise ValueError('raw histories require the current legal graph')
            context=OnlineContext.from_inputs(history,obs,partial_graph,actions)
        features=extract(context);features_done=time.perf_counter()
        result=self.predict_features(features,horizon,worlds,seed)
        end=time.perf_counter()
        self.last_profile.update(input_graph_history_ms=(features_done-start)*1000,complete_ms=(end-start)*1000)
        self.last_context=context
        return result

    def parameters(self,features):
        forbidden={'target','target_mask','physical','unseen_positions','set_positions','ground_truth','oracle_id','query','queries',
                   'future_observations','outcomes','privileged_future','labels','supervision_human_index','graph_delta',
                   'population_current_positions','population_future_positions','population_future_valid','population_person_mask'}
        if forbidden&set(features):raise ValueError('offline target fields in predictor features')
        def tensor(x):return torch.as_tensor(np.asarray(x,np.float32))[None]
        n=len(features['keys']);dummy=max(n,1)
        x=features['x'] if n else np.zeros((1,28),np.float32)
        channels=self.network.graph_channels
        graph=features['graph'][...,:channels] if n else np.zeros((1,41,channels),np.float32)
        with torch.no_grad():
            p=self.network(tensor(x),tensor(graph),tensor(np.ones(n) if n else np.zeros(1)),
                           tensor(features['global_features']),tensor(features['global_graph'][...,:channels]))
        return {k:v[0].numpy() for k,v in p.items()}

    def predict_features(self,f,horizon=3.,worlds=32,seed=None,times=None):
        """Prepared-feature entry is for offline matched science, not latency."""
        begin=time.perf_counter();p=self.parameters(f);network_done=time.perf_counter()
        if seed is None:
            fingerprint=hashlib.sha256(b'legal-human-world-bank-1')
            fingerprint.update(np.asarray(f['timestamp'],np.float64).tobytes())
            for field in ('x','origin','velocity','age','robotxy'):fingerprint.update(np.ascontiguousarray(f[field]).tobytes())
            fingerprint.update(repr(f['keys']).encode());seed=int.from_bytes(fingerprint.digest()[:8],'little')
        rng=np.random.default_rng(seed);n=len(f['keys']);k=worlds
        times=np.linspace(0,horizon,64) if times is None else np.asarray(times,float)
        if times[0]!=0 or times[-1]>3.028001 or np.any(np.diff(times)<=0):raise ValueError('invalid output times')
        mode=rng.choice(4,size=k,p=softmax(p['mode_logits']))
        z=rng.normal(size=(k,n,5,2));common=rng.normal(size=(k,1,5,2))
        initial_noise=z[:,:,0].copy()
        if self.network.tail=='student':
            z/=np.sqrt(rng.chisquare(3,size=(k,n,1,1))).clip(1e-12)
            if self.network.independent_initial:z[:,:,0]=initial_noise
        coefficient=p['means'][:n][np.arange(n)[None],mode[:,None]]+p['scales'][:n][np.arange(n)[None],mode[:,None]]*(z*.979795897+common*.2)
        if self.network.independent_initial:
            coefficient[:,:,0]=p['means'][:n][np.arange(n)[None],mode[:,None],0]+p['scales'][:n][np.arange(n)[None],mode[:,None],0]*z[:,:,0]
        initial=coefficient[:,:,0]*f['age'][None,:,None]
        ends=np.asarray(KNOTS[1:]);starts=np.asarray(KNOTS[:-1])
        phi=np.minimum(np.maximum(times[:,None]-starts,0),ends-starts)
        phi[:,-1]=np.maximum(times-starts[-1],0)
        if self.network.max_speed is None:
            drift=f['velocity'][None]+(coefficient[:,:,0] if self.network.persistent_offset_velocity else 0.)
            trajectories=f['origin'][None,None]+drift[:,None]*times[None,:,None,None]+initial[:,None]+np.einsum('wnic,ti->wtnc',coefficient[:,:,1:],phi)
        else:
            from .speed import bound_numpy
            measured=f['velocity'][None]
            initial_velocity=bound_numpy(measured+coefficient[:,:,0],self.network.max_speed)
            future_velocity=bound_numpy(measured[:,:,None]+coefficient[:,:,1:]+
                (coefficient[:,:,0,None] if self.network.persistent_offset_velocity else 0.),self.network.max_speed)
            current=f['origin'][None]+(initial_velocity-measured)*f['age'][None,:,None]
            trajectories=current[:,None]+np.einsum('wnic,ti->wtnc',future_velocity,phi)
        rate=float(p['rate'])
        if not np.isfinite(rate) or rate>10000:raise RuntimeError('population law outside resource-supported count range')
        presence=1/(1+np.exp(-np.clip(p['presence_logits'][:n],-50,50)))
        certain=certain_sources(f['age'],f['minimum_count'],self.network.compulsory_unique_episode,self.network.compulsory_observed_clique)
        presence=np.where(certain,1.,presence)
        minimum=0
        if self.network.minimum_count_conditioned:
            minimum=int(f['minimum_count'])
            track_present,counts,log_normalizer=sample_conditioned(p['presence_logits'][:n],f['age'],rate,minimum,k,rng,certain)
        else:
            counts=rng.poisson(rate,size=k);track_present=rng.random((k,n))<presence[None]
            track_present[:,certain]=True
        hidden=int(counts.max());total=n+hidden
        positions=np.zeros((k,len(times),total,2),np.float32);positions[:,:,:n]=trajectories
        present=np.zeros((k,total),bool);present[:,:n]=track_present
        spatial_p=softmax(p['spatial_logits'])
        for w in range(k):
            c=int(counts[w]);present[w,n:n+c]=True
            if not c:continue
            component=rng.choice(4,size=c,p=spatial_p)
            initial_position=f['robotxy']+p['spatial_mean'][component]+p['spatial_scale'][component]*rng.normal(size=(c,2))
            velocity=p['spatial_velocity'][component]+p['spatial_velocity_scale'][component,None]*rng.normal(size=(c,2))
            if self.network.max_speed is not None:
                from .speed import bound_numpy
                velocity=bound_numpy(velocity,self.network.max_speed)
            positions[w,:,n:n+c]=initial_position[None]+times[:,None,None]*velocity[None]
        support=('MONTE_CARLO_APPROXIMATION_NOT_SAFETY_CERTIFICATE',
            'ANONYMOUS_EPISODE_OWNERSHIP_APPROXIMATION',
            'UNKNOWN_GEOMETRY_UNCONSTRAINED_HUMAN_PATH_PRIOR',
            'CSPACE_OBSTRUCTIONS_ARE_NOT_HUMAN_MATERIAL_WALLS',
            'RESIDUAL_POISSON_MOTION_IS_CONSTANT_VELOCITY_PRIOR')
        law=JointHumanLaw(MappingProxyType({key:immutable_array(value) for key,value in p.items()}),
            immutable_array(f['origin']),immutable_array(f['velocity']),immutable_array(f['age']),immutable_array(f['robotxy']),
            minimum,self.network.tail,self.network.independent_initial,persistent_offset_velocity=self.network.persistent_offset_velocity,
            compulsory_unique_episode=self.network.compulsory_unique_episode)
        if self.network.compulsory_observed_clique:
            from dataclasses import replace
            law=replace(law,compulsory_observed_clique=True)
        if self.network.max_speed is not None:
            from dataclasses import replace
            law=replace(law,max_speed=self.network.max_speed)
        result=JointHumanWorlds(f['timestamp'],immutable_array(times),immutable_array(np.full(k,1./k)),immutable_array(positions),immutable_array(present),
            tuple(f['keys'])+tuple('unobserved_sample_slot_%d'%i for i in range(hidden)),
            immutable_array(np.r_[np.zeros(n,bool),np.ones(hidden,bool)]),rate,support,minimum_count=minimum,law=law).validate()
        end=time.perf_counter();self.last_profile=dict(model_ms=(network_done-begin)*1000,distribution_ms=(end-network_done)*1000)
        return result
