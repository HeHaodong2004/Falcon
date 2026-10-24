"""Small graph-conditioned joint path mixture and residual spatial Poisson law."""
import torch
from torch import nn
from torch.nn import functional as F
from .features import ADJ, FEATURES, GRAPH_FEATURES

MODES=4
KNOTS=(0.,.5,1.,2.,3.)


class HumanGraphPredictor(nn.Module):
    def __init__(self,width=48,graph_enabled=True,tail='gaussian',independent_initial=False,graph_channels=10,unseen_motion_std=None,minimum_count_conditioned=False,persistent_offset_velocity=False,compulsory_unique_episode=False,max_speed=None,compulsory_observed_clique=False,residual_count_components=1,total_count_tilt_cutoff=None,empty_history_count_affine=False,positive_history_count_affine=None):
        super().__init__();self.width=width;self.graph_enabled=graph_enabled;self.tail=tail;self.independent_initial=independent_initial
        self.register_buffer('adj',torch.tensor(ADJ))
        self.graph_in=nn.Linear(graph_channels,width)
        self.graph_channels=graph_channels;self.unseen_motion_std=unseen_motion_std
        self.minimum_count_conditioned=minimum_count_conditioned
        self.persistent_offset_velocity=persistent_offset_velocity
        self.compulsory_unique_episode=compulsory_unique_episode
        self.max_speed=max_speed
        if max_speed is not None and max_speed!=3.:raise ValueError('this dataset certifies exactly the3m/s bound')
        self.compulsory_observed_clique=compulsory_observed_clique
        if compulsory_observed_clique:
            raise ValueError('Rejected research configuration: greedy tracker keys do not certify persistent physical identities; use the archived experiment only to reproduce its failure')
        self.graph_message=nn.Linear(width,width)
        self.agent=nn.Sequential(nn.Linear(FEATURES+2*width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.path=nn.Linear(2*width,MODES*20+1)
        self.population=nn.Sequential(nn.Linear(12+2*width,width),nn.SiLU(),nn.Linear(width,1+MODES+MODES*8))
        if type(residual_count_components) is not int or residual_count_components not in (1,2):
            raise ValueError('supported residual count components are1 or2')
        self.residual_count_components=residual_count_components
        if total_count_tilt_cutoff is not None and (type(total_count_tilt_cutoff) is not int or total_count_tilt_cutoff!=7 or residual_count_components!=1):
            raise ValueError('total-count tilt uses seven explicit counts plus an unbounded tail, independently of residual mixtures')
        self.total_count_tilt_cutoff=total_count_tilt_cutoff
        if type(empty_history_count_affine) is not bool or (empty_history_count_affine and total_count_tilt_cutoff!=7):
            raise ValueError('empty-history affine calibration requires the total-count tilt family')
        self.empty_history_count_affine=empty_history_count_affine
        if positive_history_count_affine not in (None,'pooled','intercept') or (positive_history_count_affine is not None and total_count_tilt_cutoff!=7):
            raise ValueError('positive-history count correction requires a declared tilt feature mode')
        self.positive_history_count_affine=positive_history_count_affine
        if residual_count_components==2:
            self.count_mixture=nn.Linear(width+3,3)
            nn.init.zeros_(self.count_mixture.weight)
            with torch.no_grad():
                self.count_mixture.bias.copy_(torch.tensor([-.432777,1.247505,0.]))
        if total_count_tilt_cutoff is not None:
            self.count_tilt=nn.Linear(width+3,total_count_tilt_cutoff+1)
            nn.init.zeros_(self.count_tilt.weight);nn.init.zeros_(self.count_tilt.bias)
        if empty_history_count_affine:
            # The eighth potential applies unchanged to ALL counts >=7. This
            # seven-output correction fixes its gauge, never its probability.
            self.empty_count_affine=nn.Linear(width,7)
            nn.init.zeros_(self.empty_count_affine.weight);nn.init.zeros_(self.empty_count_affine.bias)
        if positive_history_count_affine is not None:
            self.positive_count_affine=nn.Linear(width+3,7)
            nn.init.zeros_(self.positive_count_affine.weight);nn.init.zeros_(self.positive_count_affine.bias)
        nn.init.zeros_(self.path.weight);nn.init.zeros_(self.path.bias)
        with torch.no_grad():self.path.bias[-1]=3.

    def forward(self,x,graph,mask,global_features,global_graph,count_context=None):
        g=F.silu(self.graph_in(graph))
        # Messages retain UNKNOWN/FREE/CSPACE_OBSTRUCTED as distinct channels.
        g=F.silu(self.graph_message(torch.einsum('ij,bnjh->bnih',self.adj,g))+g)
        if not self.graph_enabled:g=g*0
        local=torch.cat((g[:,:,0],g.mean(2)),dim=-1)
        agent=self.agent(torch.cat((x,local),dim=-1))
        pooled=(agent*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[...,None]
        h=self.path(torch.cat((agent,pooled[:,None].expand_as(agent)),dim=-1))
        params=h[...,:-1].reshape(*h.shape[:2],MODES,20)
        means=params[...,:10].reshape(*params.shape[:3],5,2)
        scales=(F.softplus(params[...,10:]).reshape(*params.shape[:3],5,2)+.02)*.35
        gg=F.silu(self.graph_in(global_graph))
        gg=F.silu(self.graph_message(torch.einsum('ij,bjh->bih',self.adj,gg))+gg)
        if not self.graph_enabled:gg=gg*0
        population_input=torch.cat((global_features,gg[:,0],gg.mean(1)),dim=-1)
        if self.residual_count_components==1 and self.total_count_tilt_cutoff is None:
            pop=self.population(population_input)
        else:
            if count_context is None or count_context.shape!=(len(x),3) or not torch.isfinite(count_context).all():
                raise ValueError('mixed cardinality requires three certified causal count facts')
            count_hidden=self.population[1](self.population[0](population_input))
            pop=self.population[2](count_hidden)
        rate=F.softplus(pop[:,0])+1e-5
        logits=pop[:,1:5]
        spatial=pop[:,5:].reshape(-1,MODES,8)
        spatial_mean=spatial[...,:2]*5
        spatial_scale=F.softplus(spatial[...,2:4])*8+.2
        spatial_logits=spatial[...,4]
        spatial_velocity=spatial[...,5:7]
        spatial_velocity_scale=F.softplus(spatial[...,7])+.05
        if self.unseen_motion_std is not None:
            spatial_velocity=torch.zeros_like(spatial_velocity)
            spatial_velocity_scale=torch.full_like(spatial_velocity_scale,self.unseen_motion_std)
        presence=h[...,-1]
        result=dict(means=means,scales=scales,mode_logits=logits,presence_logits=presence,
                    rate=rate,spatial_mean=spatial_mean,spatial_scale=spatial_scale,spatial_logits=spatial_logits,
                    spatial_velocity=spatial_velocity,spatial_velocity_scale=spatial_velocity_scale)
        if self.residual_count_components==2:
            count_raw=self.count_mixture(torch.cat((count_hidden,count_context.to(count_hidden)),dim=-1))
            count_rates=rate[:,None]*(F.softplus(count_raw[:,:2])+1e-5)
            count_logits=torch.stack((torch.zeros_like(count_raw[:,2]),count_raw[:,2]),dim=-1)
            result.update(residual_count_rates=count_rates,residual_count_logits=count_logits,
                rate=(count_rates*F.softmax(count_logits,dim=-1)).sum(-1))
        if self.total_count_tilt_cutoff is not None:
            count_logits=self.count_tilt(torch.cat((count_hidden,count_context.to(count_hidden)),dim=-1))
            if self.empty_history_count_affine:
                # A nonempty finite causal DAG needs >=1 endpoint. Thus the
                # certified min0 fact is exactly no actual positive history;
                # current nondetection/empty proposal arrays do not imply it.
                empty=count_context[:,0]==0
                if (empty & (count_context[:,1:]!=0).any(-1)).any():
                    raise ValueError('empty actual history cannot contain current or proposal evidence')
                correction=F.pad(self.empty_count_affine(count_hidden),(0,1))
                count_logits=torch.where(empty[:,None],count_logits+correction,count_logits)
            if self.positive_history_count_affine is not None:
                # Existing masked source embeddings include legal positions,
                # ages, motion, view evidence and source-local graph features.
                count_input=torch.cat((pooled,count_context.to(pooled)),dim=-1)
                if self.positive_history_count_affine=='intercept':count_input=torch.zeros_like(count_input)
                correction=F.pad(self.positive_count_affine(count_input),(0,1))
                count_logits=torch.where((count_context[:,0]>0)[:,None],count_logits+correction,count_logits)
            result['total_count_tilt_logits']=count_logits
        return result


def basis(times):
    """Cumulative continuous piecewise-linear velocity basis, no time resampling."""
    ends=torch.tensor(KNOTS[1:],device=times.device,dtype=times.dtype)
    starts=torch.tensor(KNOTS[:-1],device=times.device,dtype=times.dtype)
    phi=(times[...,None]-starts).clamp_min(0).minimum(ends-starts)
    # Native endpoint alignment may place the around3s observation at3.024 or
    #3.028. Continue the final velocity basis through that measured endpoint.
    phi=torch.cat((phi[...,:-1],(times[...,None]-starts[-1]).clamp_min(0)),dim=-1)
    return phi


def draw_paths(p,origin,velocity,age,times,worlds,tail='gaussian',noise=None,independent_initial=False,persistent_offset_velocity=False,max_speed=None):
    """Differentiable complete paths. One persistent mode per joint world.

    Training integrates all four modes exactly; public inference samples them.
    Returns [batch,mode,world,person,time,xy].
    """
    b,n=origin.shape[:2];device=origin.device
    if noise is None:
        z=torch.randn(b,MODES,worlds,n,5,2,device=device)
        initial_noise=z[...,0,:].clone()
        common=torch.randn(b,MODES,worlds,1,5,2,device=device)
        if tail=='student':
            # t(3), scaled to unit variance; shared denominators per trajectory.
            denominator=torch.randn(b,MODES,worlds,n,1,3,device=device).square().sum(-1,keepdim=True).sqrt()
            z=z/denominator.clamp_min(1e-6)
            if independent_initial:z[...,0,:]=initial_noise
        noise=(z,common)
    z,common=noise
    mean=p['means'].permute(0,2,1,3,4)[:,:,None]
    scales=p['scales'].permute(0,2,1,3,4)[:,:,None]
    coefficient=mean+scales*(z*.979795897+common*.2)
    if independent_initial:
        first=mean[...,0,:]+scales[...,0,:]*z[...,0,:]
        coefficient=torch.cat((first[...,None,:],coefficient[...,1:,:]),dim=-2)
    offset=coefficient[...,0,:]*age[:,None,None,:,None]
    increments=coefficient[...,1:,:]
    phi=basis(times)
    if max_speed is not None:
        from .speed import bound_torch
        measured_velocity=velocity[:,None,None]
        initial_velocity=bound_torch(measured_velocity+coefficient[...,0,:],max_speed)
        future_velocity=bound_torch(measured_velocity[...,None,:]+increments+
            (coefficient[...,0,None,:] if persistent_offset_velocity else 0.),max_speed)
        current=origin[:,None,None]+(initial_velocity-measured_velocity)*age[:,None,None,:,None]
        return current[...,None,:]+torch.einsum('bmwnic,bti->bmwntc',future_velocity,phi)
    residual=torch.einsum('bmwnic,bti->bmwntc',increments,phi)
    drift=velocity[:,None,None]
    if persistent_offset_velocity:drift=drift+coefficient[...,0,:]
    return origin[:,None,None,:,None,:]+drift[...,None,:]*times[:,None,None,None,:,None]+offset[...,None,:]+residual


def poisson_loss(p,unseen_positions,unseen_mask,robotxy,unseen_velocity=None,velocity_valid=None):
    relative=unseen_positions-robotxy[:,None]
    delta=(relative[:,:,None]-p['spatial_mean'][:,None])/p['spatial_scale'][:,None]
    density=-.5*delta.square().sum(-1)-p['spatial_scale'].log().sum(-1)[:,None]-1.8378770664
    if unseen_velocity is not None:
        dv=(unseen_velocity[:,:,None]-p['spatial_velocity'][:,None])/p['spatial_velocity_scale'][:,None,:,None]
        vd=-.5*dv.square().sum(-1)-2*p['spatial_velocity_scale'].log()[:,None]-1.8378770664
        density=density+vd*velocity_valid[...,None]
    log_density=torch.logsumexp(density+F.log_softmax(p['spatial_logits'],dim=-1)[:,None],dim=-1)
    return p['rate']-(unseen_mask*(p['rate'].log()[:,None]+log_density)).sum(-1)


def native_marginal_nll(p,batch,persistent_offset_velocity=False):
    """Proper exact2D Gaussian-mixture score at certified native future times.

    This scores the marginal of the SAME coherent coefficient law. It does not
    assert a nonsingular density for the complete low-rank trajectory vector.
    Shared future innovations have unit marginal variance per coefficient.
    """
    phi=basis(batch['times'])
    age=batch['age'][:,:,None,None]
    offset_factor=age[...,None]+batch['times'][:,None,None,:,None] if persistent_offset_velocity else age[...,None]
    mean=batch['origin'][:,:,None,None,:]+batch['velocity'][:,:,None,None,:]*batch['times'][:,None,None,:,None]
    mean=mean+p['means'][:,:,:,0,None,:]*offset_factor
    mean=mean+torch.einsum('bnmic,bti->bnmtc',p['means'][:,:,:,1:],phi)
    variance=(p['scales'][:,:,:,0,None,:]*offset_factor).square()
    variance=variance+torch.einsum('bnmic,bti->bnmtc',p['scales'][:,:,:,1:].square(),phi.square())
    # Censoredtime0 entries are outside the score; the tiny numerical floor
    # prevents inactive0/0 and never activates at a valid native future time.
    variance=variance.clamp_min(1e-12)
    deviation=batch['target'][:,:,None]-mean
    log_density=-.5*(deviation.square()/variance+variance.log()).sum(-1)-1.8378770664093453
    mixture=torch.logsumexp(log_density+F.log_softmax(p['mode_logits'],dim=-1)[:,None,:,None],dim=2)
    return -(mixture*batch['target_mask']).sum((1,2))/batch['target_mask'].sum((1,2)).clamp_min(1)


def proper_loss(p,batch,worlds=4,tail='gaussian',population_objective='residual',independent_initial=False,condition_minimum=False,endpoint_weight=0.,persistent_offset_velocity=False,compulsory_unique_episode=False,max_speed=None,compulsory_observed_clique=False):
    paths=draw_paths(p,batch['origin'],batch['velocity'],batch['age'],batch['times'],worlds,tail,independent_initial=independent_initial,persistent_offset_velocity=persistent_offset_velocity,max_speed=max_speed)
    mask=batch['target_mask'][:,None,None,:,:,None]
    denom=batch['target_mask'].sum((1,2)).clamp_min(1).sqrt()
    deviation=((paths-batch['target'][:,None,None])*mask).flatten(3)
    first=deviation.norm(dim=-1).mean(2)
    # Exact mixture cross term estimated with independent draws from each mode.
    # Fixed pairing of disjoint independent worlds is unbiased, including m=m'.
    left=paths[:,:,:worlds//2];right=paths[:,:,worlds//2:]
    cross=[];marginal_cross=[]
    for i in range(MODES):
        row=[];marginal_row=[]
        for j in range(MODES):
            d=(left[:,i]-right[:,j])*mask[:,0]
            row.append(d.flatten(2).norm(dim=-1).mean(1))
            if max_speed is not None:marginal_row.append(d.norm(dim=-1).mean(1))
        cross.append(torch.stack(row,-1))
        if max_speed is not None:marginal_cross.append(torch.stack(marginal_row,1))
    cross=torch.stack(cross,1)
    probability=F.softmax(p['mode_logits'],dim=-1)
    es=((first*probability).sum(-1)-.5*(cross*probability[:,:,None]*probability[:,None,:]).sum((1,2)))/denom
    has_target=(batch['target_mask'].sum((1,2))>0)
    existence=F.binary_cross_entropy_with_logits(p['presence_logits'],batch['presence'],reduction='none')
    existence=(existence*batch['presence_mask']).sum(-1)/batch['presence_mask'].sum(-1).clamp_min(1)
    population=poisson_loss(p,batch['unseen_positions'],batch['unseen_mask'],batch['robotxy'],batch['unseen_velocity'],batch['unseen_velocity_valid'])
    # Declared proper-score combination. The Poisson term includes count and
    # spatial intensity, with every empty target contributing its integral rate.
    if population_objective=='set':
        from .population import set_nll
        population=set_nll(p,batch,condition_minimum,compulsory_unique_episode,max_speed,compulsory_observed_clique)
        per_anchor=es*has_target+.05*population
    else:per_anchor=es*has_target+.05*existence+.05*population
    if endpoint_weight and tail!='gaussian':raise ValueError('exact marginal likelihood implemented for Gaussian coefficient law only')
    if max_speed is not None:
        # A nonlinear bounded path is not Gaussian. Score its true native point
        # marginals by a proper Energy Score instead of a fabricated density.
        marginal_first=((paths-batch['target'][:,None,None]).norm(dim=-1).mean(2)*probability[:,:,None,None]).sum(1)
        mcross=torch.stack(marginal_cross,1)
        second=(mcross*probability[:,:,None,None,None]*probability[:,None,:,None,None]).sum((1,2))
        marginal=((marginal_first-.5*second)*batch['target_mask']).sum((1,2))/batch['target_mask'].sum((1,2)).clamp_min(1)
    else:marginal=native_marginal_nll(p,batch,persistent_offset_velocity) if tail=='gaussian' else torch.zeros_like(es)
    per_anchor=per_anchor+endpoint_weight*marginal
    return (per_anchor*batch['weight']).mean(),dict(es=es.detach(),population=population.detach(),marginal_score=marginal.detach())
