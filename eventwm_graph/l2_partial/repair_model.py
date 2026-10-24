"""Controlled repair of the released graph belief; the original is reproducible.

The compact graph network, count hypotheses, shared worlds and native query
executor are retained. A canonical .048s clock prevents resets at arbitrary
API boundaries. Within-interval alternatives survive partial propagation.
"""
import math,time,json,pathlib,hashlib
import numpy as np
from .model import GraphBeliefWorldModel
from .maneuver import transition_components,advance_components,moment_match,condition_components,sample_resets
from .probability import transition_matrices,condition_position
from .observation import EndpointObservationLaw

REPAIR_VERSION='partial-graph-belief-repair-0.2'
NATIVE_DT=.048

class RepairedGraphBeliefWorldModel(GraphBeliefWorldModel):
    event_clock_version='native_elapsed_age_v2'
    def __init__(self,*args,observation_repair=True,maneuver_rate=2.,reset_std=2.,maneuver_retention=1.,observation_law=None,observation_checkpoint=None,successor_regularization=(.005,.02),**kwargs):
        super().__init__(*args,**kwargs)
        self.observation_repair=bool(observation_repair)
        self.maneuver_rate=float(maneuver_rate);self.reset_std=float(reset_std)
        self.maneuver_retention=float(maneuver_retention)
        parameters=json.loads(pathlib.Path(observation_checkpoint).read_text())['parameters'] if observation_checkpoint else None
        self.observation_law=observation_law or EndpointObservationLaw(parameters)
        self.successor_regularization=tuple(successor_regularization)
        config={'kind':self.kind,'hypotheses':self.max_hypotheses,'rate':self.maneuver_rate,
            'std':self.reset_std,'retention':self.maneuver_retention,'observation_repair':self.observation_repair,
            'in_view':self.observation_law.in_view,'out_view':self.observation_law.out_view,
            'regime_prior':self.observation_law.regime_prior,'successor_regularization':self.successor_regularization,
            'observation_law_class':type(self.observation_law).__module__+'.'+type(self.observation_law).__qualname__,
            'observation_law_version':self.observation_law.version,'observation_instance':vars(self.observation_law)}
        digest=hashlib.sha256(json.dumps(config,sort_keys=True,default=lambda x:np.asarray(x).tolist()).encode());digest.update(self.population_prior.tobytes())
        if self.dynamics is not None and hasattr(self.dynamics,'network'):
            digest.update(repr(self.dynamics.scale).encode())
            for name,p in self.dynamics.network.state_dict().items():digest.update(name.encode());digest.update(p.cpu().numpy().tobytes())
        elif self.dynamics is not None:digest.update((type(self.dynamics).__module__+'.'+type(self.dynamics).__qualname__).encode())
        if self.event_model is not None and hasattr(self.event_model,'coefficients'):digest.update(self.event_model.coefficients.tobytes())
        elif self.event_model is not None:digest.update((type(self.event_model).__module__+'.'+type(self.event_model).__qualname__).encode())
        self.model_version=REPAIR_VERSION+'-'+digest.hexdigest()[:12]

    def history_filter(self):
        return RepairedGraphBeliefWorldModel(kind='structured' if self.observation_repair else 'memory_cv',max_hypotheses=self.max_hypotheses,
            population_prior=self.population_prior,observation_repair=self.observation_repair,
            maneuver_rate=self.maneuver_rate,reset_std=self.reset_std,maneuver_retention=self.maneuver_retention,observation_law=self.observation_law)

    def initial_hypotheses(self,h):
        if not self.observation_repair:return [h]
        from .hidden import prepare_prior
        hm,hc=prepare_prior(self._initial_legal_pose)
        h=h.fork(hidden_mean=hm,hidden_covariance=hc)
        return [h.fork(weight=float(p),visibility_mode=i) for i,p in enumerate(self.observation_law.regime_prior)]

    def initialize(self,*args,**kwargs):
        history=args[0] if args else kwargs['legal_history']
        self._initial_legal_pose=np.asarray(history[0]['robot_pose'],float)
        b=super().initialize(*args,**kwargs).fork(model_version=self.model_version)
        # Initial latent setup belongs to initialization and remains persistent.
        # Subsequent observe/forecast calls reuse these same joint intentions.
        return self.propagate(b,b.timestamp).validate()

    def propagate(self,belief,timestamp):
        self._check_belief(belief)
        return super().propagate(belief,timestamp).fork(model_version=self.model_version)

    def _check_belief(self,belief):
        if belief.model_version not in ('unbound',self.model_version):
            raise ValueError('belief belongs to a different model version; reinitialize from legal history')

    def predict_shared(self,belief,*args,**kwargs):
        self._check_belief(belief)
        f=super().predict_shared(belief,*args,**kwargs)
        if f.source.model_version=='unbound':f.source=f.source.fork(model_version=self.model_version)
        return f

    def evaluate_queries(self,belief,queries,shared_future,*args,**kwargs):
        self._check_belief(belief)
        if shared_future.source.model_version!=self.model_version:raise ValueError('shared future generated by a different model version')
        return super().evaluate_queries(belief,queries,shared_future,*args,**kwargs)

    def prepare(self,belief):
        self._check_belief(belief)
        b=super().prepare(belief)
        if self.observation_repair and any(h.hidden_mean is None for h in b.hypotheses):
            from .hidden import prepare_prior
            hm,hc=prepare_prior(b.robot_pose)
            b=b.fork(hypotheses=[h if h.hidden_mean is not None else h.fork(hidden_mean=hm,hidden_covariance=hc) for h in b.hypotheses])
        return b

    def advance_hypothesis(self,h,dt,start):
        if dt<=1e-12:return h
        end=start+dt;t=float(start);hh=h
        while t<end-1e-10:
            boundary=(math.floor((t+1e-9)/NATIVE_DT)+1)*NATIVE_DT
            stop=min(end,boundary);step=stop-t
            if hh.maneuver_mean is None or hh.maneuver_until<t+1e-9:
                # If starting inside a native interval without a previous
                # mixture (e.g. exact first sight), reset begins NEXT interval.
                if abs(t/NATIVE_DT-round(t/NATIVE_DT))<1e-7:
                    mm,cc,pp=transition_components(hh,step,self.maneuver_rate,self.reset_std,retention=self.maneuver_retention)
                    pp=np.broadcast_to(pp,(hh.n,2)).copy()
                else:
                    mm=hh.mean[:,None,:];cc=hh.covariance[:,None,:,:];pp=np.ones((hh.n,1))
                    mm,cc=advance_components(mm,cc,step,hh.intent,hh.diffusion,hh.relaxation)
            else:
                mm,cc=advance_components(hh.maneuver_mean,hh.maneuver_covariance,step,hh.intent,hh.diffusion,hh.relaxation)
                pp=hh.maneuver_probability
            mean,cov=moment_match(mm,cc,pp)
            hh=hh.fork(mean=mean,covariance=cov,maneuver_mean=mm,maneuver_covariance=cc,
                       maneuver_probability=pp,maneuver_until=boundary)
            t=stop
        if h.hidden_mean is not None:
            # Use the SAME native-clock maneuver/OU family for never-localized
            # people. The broad hidden target is zero; sensing does not change
            # the transition of a first-localized component.
            from .state import Hypothesis
            m=len(h.hidden_mean)
            private=Hypothesis(1.,h.hidden_mean,h.hidden_covariance,np.zeros(m),np.arange(m),np.array([1.]),
                intent=np.zeros((m,2)),diffusion=np.full(m,.25),mode=np.zeros(m,int),relaxation=.7,
                maneuver_until=h.maneuver_until,maneuver_mean=h.hidden_maneuver_mean,
                maneuver_covariance=h.hidden_maneuver_covariance,maneuver_probability=h.hidden_maneuver_probability)
            hp=self.advance_hypothesis(private,dt,start)
            hh=hh.fork(hidden_mean=hp.mean,hidden_covariance=hp.covariance,hidden_maneuver_mean=hp.maneuver_mean,
                hidden_maneuver_covariance=hp.maneuver_covariance,hidden_maneuver_probability=hp.maneuver_probability)
        return hh

    def _advance_family(self,mean,cov,target,diff,rate,dt,start,mm,cc,pp,until):
        """Vectorized independent Gaussian families, with unchanged native law."""
        t=float(start);end=t+dt
        while t<end-1e-10:
            boundary=(math.floor((t+1e-9)/NATIVE_DT)+1)*NATIVE_DT;stop=min(end,boundary)
            if mm is None or until<t+1e-9:
                if abs(t/NATIVE_DT-round(t/NATIVE_DT))<1e-7:
                    mm=np.repeat(mean[...,None,:],2,axis=-2);cc=np.repeat(cov[...,None,:,:],2,axis=-3)
                    rho=self.maneuver_retention
                    mm[...,1,2:]*=rho;cc[...,1,:2,2:]*=rho;cc[...,1,2:,:2]*=rho
                    cc[...,1,2:,2:]=rho*rho*cc[...,1,2:,2:]+np.eye(2)*self.reset_std**2
                    p=-math.expm1(-self.maneuver_rate*NATIVE_DT)
                    pp=np.broadcast_to(np.array([1-p,p]),mean.shape[:-1]+(2,))
                else:mm=mean[...,None,:];cc=cov[...,None,:,:];pp=np.ones(mean.shape[:-1]+(1,))
            mm,cc=advance_components(mm,cc,stop-t,target,diff,rate)
            mean,cov=moment_match(mm,cc,pp);until=boundary;t=stop
        return mean,cov,mm,cc,pp,until

    def advance_hypotheses(self,hypotheses,dt,start):
        # Match the scalar loop's no-step tolerance, including roundoff at
        # large absolute timestamps. No Gaussian mixture has been constructed
        # if the scalar transition would execute zero iterations.
        if not float(start)<float(start+dt)-1e-10:return hypotheses
        changes=[{} for h in hypotheses]
        for hidden in (False,True):
            groups={}
            for i,h in enumerate(hypotheses):
                mean=h.hidden_mean if hidden else h.mean
                if mean is None:continue
                mm=h.hidden_maneuver_mean if hidden else h.maneuver_mean
                key=(len(mean),.7 if hidden else h.relaxation,0 if mm is None else mm.shape[-2],h.maneuver_until)
                groups.setdefault(key,[]).append(i)
            for (N,rate,K,until),indices in groups.items():
                hs=[hypotheses[i] for i in indices]
                mean=np.stack([h.hidden_mean if hidden else h.mean for h in hs])
                cov=np.stack([h.hidden_covariance if hidden else h.covariance for h in hs])
                target=np.zeros((*mean.shape[:-1],2)) if hidden else np.stack([h.intent for h in hs])
                diff=np.full(mean.shape[:-1],.25) if hidden else np.stack([h.diffusion for h in hs])
                if K:
                    mm=np.stack([h.hidden_maneuver_mean if hidden else h.maneuver_mean for h in hs])
                    cc=np.stack([h.hidden_maneuver_covariance if hidden else h.maneuver_covariance for h in hs])
                    pp=np.stack([h.hidden_maneuver_probability if hidden else h.maneuver_probability for h in hs])
                else:mm=cc=pp=None
                mean,cov,mm,cc,pp,until=self._advance_family(mean,cov,target,diff,rate,dt,start,mm,cc,pp,until)
                prefix='hidden_' if hidden else ''
                for j,i in enumerate(indices):
                    changes[i].update({prefix+'mean':mean[j],prefix+'covariance':cov[j],prefix+'maneuver_mean':mm[j],
                        prefix+'maneuver_covariance':cc[j],prefix+'maneuver_probability':pp[j],'maneuver_until':until})
        return [h.fork(**c) for h,c in zip(hypotheses,changes)]

    def condition_track(self,h,i,z):
        if h.maneuver_mean is not None and len(h.maneuver_mean)==h.n:
            return condition_components(h.maneuver_mean[i],h.maneuver_covariance[i],h.maneuver_probability[i],z)
        return condition_position(h.mean[i],h.covariance[i],z)

    def observe(self,belief,actual_observation,actual_execution_trace=None):
        self._check_belief(belief)
        if not self.observation_repair:
            b=super().observe(belief,actual_observation,actual_execution_trace)
            # The legacy sequential updater moment-matches the conditionals.
            return b.fork(hypotheses=[h.fork(maneuver_mean=None,maneuver_covariance=None,maneuver_probability=None) for h in b.hypotheses],model_version=self.model_version)
        from .repair_filter import observe
        return observe(self,belief,actual_observation,actual_execution_trace)

    def sample_initial(self,h,rng):
        if h.maneuver_mean is not None and len(h.maneuver_mean)==h.n:
            k=(np.cumsum(h.maneuver_probability,axis=1)<rng.random((h.n,1))).sum(1)
            ii=np.arange(h.n);mean=h.maneuver_mean[ii,k].copy();cov=h.maneuver_covariance[ii,k]
        else:mean=h.mean.copy();cov=h.covariance
        if h.n:
            eig,U=np.linalg.eigh(cov);mean+=np.einsum('nij,nj->ni',U,rng.normal(size=(h.n,4))*np.sqrt(np.maximum(eig,0)))
        return mean

    def sample_maneuver_paths(self,worlds,rel,start,rng,n):
        # One physical-state tensor for tracked AND private hidden people.
        # Their targets/rates remain distinct; every latent person is retained.
        from .hidden import sample_hidden_states
        W=len(worlds)
        counts=[int(round(h.expected_hidden)) if h.hidden_mean is not None else 0 for h in worlds]
        N=max((h.n+k for h,k in zip(worlds,counts)),default=0)
        means=np.zeros((W,N,4));targets=np.zeros((W,N,2));diff=np.zeros((W,N));rates=np.zeros((W,N));valid=np.zeros((W,N),bool)
        for w,(h,k) in enumerate(zip(worlds,counts)):
            means[w,:h.n]=h.mean;targets[w,:h.n]=h.intent;diff[w,:h.n]=h.diffusion;rates[w,:h.n]=h.relaxation
            if k:
                if h.hidden_maneuver_mean is not None:
                    if h.hidden_exchangeable:
                        gm=np.repeat(h.hidden_maneuver_mean,k,axis=0);gc=np.repeat(h.hidden_maneuver_covariance,k,axis=0)
                        gp=np.repeat(h.hidden_maneuver_probability,k,axis=0)
                    else:gm,gc,gp=h.hidden_maneuver_mean,h.hidden_maneuver_covariance,h.hidden_maneuver_probability
                    choice=(np.cumsum(gp,axis=1)<rng.random((k,1))).sum(1);ii=np.arange(k)
                    hidden=sample_hidden_states(gm[ii,choice],gc[ii,choice],k,rng,False)
                else:hidden=sample_hidden_states(h.hidden_mean,h.hidden_covariance,k,rng,h.hidden_exchangeable)
                means[w,h.n:h.n+k]=hidden;diff[w,h.n:h.n+k]=.25;rates[w,h.n:h.n+k]=.7
            valid[w,:h.n+k]=True
        positions=np.full((W,len(rel),N,2),np.nan);velocities=positions.copy()
        positions[:,0]=np.where(valid[...,None],means[:,:,:2],np.nan);velocities[:,0]=np.where(valid[...,None],means[:,:,2:],np.nan)
        flat_target=targets.reshape(-1,2);flat_diff=diff.ravel();groups=[]
        for rate in np.unique(rates[valid]):
            select=np.flatnonzero((valid&(rates==rate)).ravel())
            groups.append((float(rate),select,flat_target[select],np.sqrt(flat_diff[select,None])))
        factors={};current=float(start);reset_already=all(h.maneuver_until>start+1e-9 for h in worlds)
        for si,relative in enumerate(rel[1:],1):
            stop=float(start+relative)
            while current<stop-1e-10:
                at_grid=abs(current/NATIVE_DT-round(current/NATIVE_DT))<1e-7
                if at_grid and not reset_already:
                    means,_=sample_resets(means,rng,self.maneuver_rate,self.reset_std,retention=self.maneuver_retention)
                boundary=(math.floor((current+1e-9)/NATIVE_DT)+1)*NATIVE_DT
                nxt=min(stop,boundary);dt=nxt-current;flat=means.reshape(-1,4)
                for rate,select,target,sqrt_diff in groups:
                    key=(round(dt,12),rate)
                    if key not in factors:
                        F,Q,a,e=transition_matrices(dt,rate);factors[key]=(F,np.linalg.cholesky(Q+np.eye(4)*1e-16),a,e)
                    F,L,a,e=factors[key];x=flat[select]@F.T
                    x[:,:2]+=(dt-a)*target;x[:,2:]+=(1-e)*target
                    x+=(rng.normal(size=x.shape)@L.T)*sqrt_diff
                    if rate>0:x[:,:2]+=rng.normal(size=x[:,:2].shape)*np.sqrt(.0025*dt)
                    flat[select]=x
                current=nxt;reset_already=abs(current-boundary)>1e-9
            positions[:,si]=np.where(valid[...,None],means[:,:,:2],np.nan)
            velocities[:,si]=np.where(valid[...,None],means[:,:,2:],np.nan)
        until=(math.floor((current-1e-9)/NATIVE_DT)+1)*NATIVE_DT;updated=[]
        for w,(h,k) in enumerate(zip(worlds,counts)):
            changes={'mean':means[w,:h.n].copy(),'maneuver_mean':None,'maneuver_covariance':None,'maneuver_probability':None,'maneuver_until':until}
            if h.hidden_mean is not None:changes.update(hidden_mean=means[w,h.n:h.n+k].copy(),hidden_covariance=np.zeros((k,4,4)),hidden_exchangeable=False,
                hidden_maneuver_mean=None,hidden_maneuver_covariance=None,hidden_maneuver_probability=None)
            updated.append(h.fork(**changes))
        return positions,velocities,updated

    def _endpoint_hypothesis(self,future,index,w,h,cov):
        mean=np.empty((h.n,4));mean[:,:2]=future.positions[w,index,:h.n];mean[:,2:]=future.velocities[w,index,:h.n]
        until=(math.floor((future.times[index]-1e-9)/NATIVE_DT)+1)*NATIVE_DT
        changes={'mean':mean,'covariance':cov,'maneuver_mean':None,'maneuver_covariance':None,'maneuver_probability':None,'maneuver_until':until}
        if h.hidden_mean is not None:
            k=int(round(h.expected_hidden));rp,rv=self.successor_regularization
            hidden=np.empty((k,4));hidden[:,:2]=future.positions[w,index,h.n:h.n+k];hidden[:,2:]=future.velocities[w,index,h.n:h.n+k]
            changes.update(hidden_mean=hidden,hidden_covariance=np.tile(np.diag([rp**2,rp**2,rv**2,rv**2]),(k,1,1)),hidden_exchangeable=False,
                hidden_maneuver_mean=None,hidden_maneuver_covariance=None,hidden_maneuver_probability=None)
        return h.fork(**changes)
