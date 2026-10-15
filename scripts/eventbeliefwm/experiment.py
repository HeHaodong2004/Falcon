"""Bounded root-sequence experiments. All future fields are loss/metric targets."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import torch
from torch.nn import functional as F
from eventbeliefwm.data import RootTransitionDataset, collate_roots, sha256
from eventbeliefwm.model import BeliefWorldModel, BeliefConfig
from eventbeliefwm.metrics import trajectory_energy, trajectory_nll, mean_valid

torch.set_num_threads(1)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def source_commit():
    return subprocess.check_output(['git','rev-parse','HEAD'], text=True,
                                  cwd=Path(__file__).resolve().parents[2]).strip()


def source_fingerprints():
    directory=Path(__file__).resolve().parent
    return {name:sha256(directory/name) for name in ('model.py','data.py','metrics.py','experiment.py')}


def load_split(cache, split, device='cpu', limit=None):
    dataset = RootTransitionDataset(cache, split)
    for row in dataset.rows[:min(len(dataset),limit or len(dataset))]:
        if sha256(Path(cache)/row['file'])!=row['sha256']:
            raise ValueError('Immutable transition cache payload changed: '+row['root_id'])
    samples = [dataset[i] for i in range(min(len(dataset), limit or len(dataset)))]
    if not samples:
        raise ValueError('No roots in '+split)
    return collate_roots(samples, device)


def select_batch(full, indices):
    ix = torch.as_tensor(indices, device=full['query'].device, dtype=torch.long)
    result = {}
    for key, value in full.items():
        if key == 'prepared_history':
            continue
        elif key == 'base':
            result[key] = value.select(ix)
        elif isinstance(value, torch.Tensor):
            result[key] = value[ix]
        elif isinstance(value, dict):
            result[key] = {name: tensor[ix] for name, tensor in value.items()}
        else:
            result[key] = [value[int(i)] for i in indices]
    # Keep the exact history; crop padding only, never observations or identities.
    base = result['base']
    length = int((~base.history_padding).sum(1).max())
    n = max(1, int((~base.human_padding).sum(1).max()))
    base.humans = base.humans[:, :n, :length]
    base.human_padding = base.human_padding[:, :n]
    base.robot = base.robot[:, :length]
    base.history_padding = base.history_padding[:, :length]
    if 'prepared_history' in full:
        result['prepared_history']=full['prepared_history'].select(ix,n=n,t=length)
    if 'history_timestamps' in result:
        result['history_timestamps']=result['history_timestamps'][:,:length]
    if 'track_acquisition_times' in result:
        result['track_acquisition_times']=result['track_acquisition_times'][:,:n,:length]
    track_axes = dict(track_positions=2, track_valid=2, initial_track_positions=1,
        initial_track_valid=1, observed_tracks=3, observed_valid=3,
        actual_observed_positions=3, actual_acquisition_times=3)
    for key, axis in track_axes.items():
        if key in result['targets']:
            slices = [slice(None)]*result['targets'][key].ndim
            slices[axis] = slice(0,n)
            result['targets'][key] = result['targets'][key][tuple(slices)]
    return result


def encode(model, batch):
    return model.encode_history(batch['base'],
        {'features': batch['geometry'], 'support': batch['geometry_support']},
        batch['sensor_config'], batch['robot_geometry'],
        history_timestamps=batch.get('history_timestamps'),
        track_acquisition_times=batch.get('track_acquisition_times'),
        prepared_history=batch.get('prepared_history'))


def stationary_query(batch_size, horizon, device, start=None):
    query = torch.zeros(batch_size, 2, 10, device=device)
    query[:, 1, 2] = horizon
    query[:, :, 6] = 1
    query[:, 1, 9] = 1
    if start is not None:
        query[..., :2] = start[:,None]
    return query


def occupancy_targets(batch, length, grid_size=32, extent=8.):
    """Physical truth raster, including zero-human roots; never an input.

    Anonymous targets remove only uniquely associated identity positions. This
    is target bookkeeping, not a supplied privileged human count.
    """
    target = batch['targets']
    points = target['all_human_positions'][:, :length]
    valid = target['all_human_valid'][:, :length].clone()
    known = target['track_positions'][:, :length]
    known_valid = target['track_valid'][:, :length]
    distance = (points[:, :, :, None]-known[:, :, None]).norm(dim=-1)
    associated = ((distance < 1e-3) & known_valid[:,:,None]).any(-1)
    coord = ((points+extent)/(2*extent)*grid_size).floor().long()
    valid &= ((coord>=0) & (coord<grid_size)).all(-1)
    index = coord[...,1].clamp(0,grid_size-1)*grid_size + coord[...,0].clamp(0,grid_size-1)
    b, t, _ = valid.shape
    all_grid = points.new_zeros(b,t,grid_size*grid_size)
    anonymous = torch.zeros_like(all_grid)
    all_grid.scatter_add_(2,index,valid.to(points.dtype))
    anonymous.scatter_add_(2,index,(valid & ~associated).to(points.dtype))
    return all_grid.clamp_max(1).reshape(b,t,grid_size,grid_size), anonymous.clamp_max(1).reshape(b,t,grid_size,grid_size)


def event_loss(forecast, targets, length, fully_tracked=None):
    cumulative = forecast.cumulative_collision
    previous = torch.cat([torch.zeros_like(cumulative[...,:1]), cumulative[...,:-1]], -1)
    hazard = (forecast.collision_interval/(1-previous).clamp_min(1e-7)).clamp(1e-6,1-1e-6)
    target = targets['collision_interval'][...,:length]
    valid = targets['collision_valid'][...,:length] & forecast.validity['query_time']
    if fully_tracked is not None:
        valid &= fully_tracked[:,None,None]
    point = F.binary_cross_entropy(hazard, target.to(hazard.dtype), reduction='none')
    per_query = (point*valid).sum(-1)
    per_root = (per_query*valid.any(-1)).sum(-1)/valid.any(-1).sum(-1).clamp_min(1)
    return mean_valid(per_root,valid.flatten(1).any(1))


def loss_bundle(model, batch, horizon=3., correction=False):
    belief = encode(model,batch)
    forecast = model.imagine(belief, batch['query'], horizon_s=horizon,
                             rasterize=False, allow_query_hold=True,detach_risk_states=True)
    length = forecast.human_positions.shape[2]
    target = batch['targets']
    gt, mask = target['track_positions'][:,:length], target['track_valid'][:,:length]
    energy, valid = trajectory_energy(forecast.human_positions, gt, mask, forecast.scenario_weights)
    nll, nvalid = trajectory_nll(forecast.human_positions, gt, mask, forecast.scenario_weights, forecast.position_scale)
    physical = mean_valid(energy,valid)
    likelihood = mean_valid(nll,nvalid)
    fully_tracked = torch.tensor([m['association']['associated']==m['physical_humans_target_only']
        for m in batch['metadata']],device=gt.device)
    event = event_loss(forecast,target,length,fully_tracked)
    observed_probability = (forecast.observation_probability *
        forecast.scenario_weights[:,None,:,None,None]).sum(2).clamp(1e-6,1-1e-6)
    observed_mask = target['observed_valid'][:,:,:length] & forecast.validity['query_time'][...,None]
    observed_bce = F.binary_cross_entropy(observed_probability,
        target['observed_tracks'][:,:,:length].to(observed_probability.dtype),reduction='none')
    detection = mean_valid(observed_bce,observed_mask)
    _, anonymous = occupancy_targets(batch,length,model.config.grid_size,model.config.grid_extent_m)
    anon_prediction = forecast.anonymous_occupancy[:,None].expand_as(anonymous).clamp(1e-7,1-1e-7)
    anon_loss = F.binary_cross_entropy(anon_prediction,anonymous,reduction='none').mean((-1,-2))
    anon = mean_valid(anon_loss,target['physical_exposure'][:,:length])
    # Primary objective is physical free-running prediction. Risk uses only
    # swept predicted positions; a small calibration loss cannot bypass dynamics.
    loss = physical + .02*likelihood + .02*event + 5.*anon + .02*detection
    correction_loss = loss.new_zeros(())
    if correction and horizon > 1.:
        first = model.imagine(belief, stationary_query(len(gt),1.,gt.device),1.,rasterize=False)
        j = round(1./model.config.dt)-1
        # Query b032 is the recorded WAIT branch; any data still obeys exposure.
        k = batch['query'].shape[1]-1
        if not all(m['rows'][k]['variant']=='WAIT' for m in batch['metadata']):
            raise ValueError('Actual correction diagnostic requires the aligned recorded WAIT branch')
        observed = target['observed_tracks'][:,k,j] & target['observed_valid'][:,k,j]
        root_times = gt.new_tensor([m['timestamp'] for m in batch['metadata']],dtype=torch.float64)
        arrived = dict(positions=target['actual_observed_positions'][:,k,j], observed=observed,
            acquisition_times=target['actual_acquisition_times'][:,k,j]-root_times[:,None],
            timestamp_s=gt.new_ones(len(gt)),source='actual_arrived')
        corrected = model.observe(first.next_belief,arrived)
        second = model.imagine(corrected,stationary_query(len(gt),horizon-1.,gt.device),horizon-1.,rasterize=False)
        score, supported = trajectory_energy(second.human_positions,gt[:,j+1:],mask[:,j+1:],second.scenario_weights)
        correction_loss = mean_valid(score,supported & observed.any(-1))
        loss = loss+.1*correction_loss
    return loss, dict(energy=float(physical.detach()), nll_per_coordinate=float(likelihood.detach()),
        event_nll=float(event.detach()),event_supported_roots=int(fully_tracked.sum()), observation_bce=float(detection.detach()),
        anonymous_bce=float(anon.detach()), correction_energy=float(correction_loss.detach()),
        loss=float(loss.detach()), supported_roots=int(valid.sum())), forecast


@torch.no_grad()
def validation_energy(model,full,horizon,batch_size):
    """Validation only; complete identity paths, equal weight per root."""
    was_training=model.training;model.eval();scores=[]
    for offset in range(0,len(full['metadata']),batch_size):
        batch=select_batch(full,list(range(offset,min(offset+batch_size,len(full['metadata'])))))
        belief=encode(model,batch)
        forecast=model.imagine(belief,stationary_query(len(batch['metadata']),horizon,full['query'].device),
                               horizon,rasterize=False)
        length=forecast.human_positions.shape[2]
        target=batch['targets']['track_positions'][:,:length]
        mask=batch['targets']['track_valid'][:,:length]
        mask=mask & mask.all(1)[:,None]
        energy,valid=trajectory_energy(forecast.human_positions,target,mask,forecast.scenario_weights)
        scores.extend(energy[valid].cpu().tolist())
    model.train(was_training)
    return dict(energy=float(np.mean(scores)) if scores else None,supported_roots=len(scores))


class ResearchGuard:
    """Read-only collector monitoring; only the caller research job may stop."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.last = 0.
        self.samples = []
        self.bad = 0
        self.start = self.sample()
        self.baseline = self.start.get('qph')

    @staticmethod
    def sample():
        path=Path('/home/haodong/workspace/datasets/eventwm-falcon/state/v4_continuous/supervisor.json')
        value=json.loads(path.read_text())
        return dict(timestamp=time.time(), heartbeat_timestamp=value['timestamp'],
                    collector_pid=value['collector_pid'],state=value['state'],
                    queries=value['progress']['primary_queries'],
                    qph=value.get('throughput',{}).get('queries_per_hour'))

    def check(self, force=False):
        if not force and time.time()-self.last<60:
            return
        self.last=time.time()
        s=self.sample()
        self.samples.append(s)
        old=[v for v in self.samples if s['timestamp']-v['timestamp']>=600]
        rate=None
        if old:
            prior=old[-1]
            rate=(s['queries']-prior['queries'])*3600/(s['timestamp']-prior['timestamp'])
        self.bad=self.bad+1 if rate is not None and self.baseline and rate<.65*self.baseline else 0
        with (self.directory/'collector_monitor.jsonl').open('a') as f:
            f.write(json.dumps(dict(**s,recent_qph=rate,consecutive_degraded=self.bad))+'\n')
        import shutil
        if shutil.disk_usage(self.directory).free<50*1024**3:
            raise RuntimeError('Research stopped: project disk reserve below50GiB')
        if (s['collector_pid']!=self.start['collector_pid'] or s['timestamp']-s['heartbeat_timestamp']>180
                or s['state']!='running' or self.bad>=3):
            raise RuntimeError('Research stopped: collector heartbeat/ownership/throughput guard')


def train(args):
    run=Path(args.run); run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists():
        raise FileExistsError('Completed run is immutable: '+str(run))
    manifest=json.loads((Path(args.cache)/'manifest.json').read_text())
    config=BeliefConfig(kind=args.kind,width=args.width,modes=args.modes,trained_horizon_s=args.horizon)
    protocol=dict(config=asdict(config),snapshot_sha256=manifest['snapshot_sha256'],
        cache=str(Path(args.cache).resolve()),cache_manifest_sha256=sha256(Path(args.cache)/'manifest.json'),
        steps=args.steps,batch_size=args.batch_size,learning_rate=args.lr,seed=args.seed,
        correction_every=4,source_commit=source_commit(),source_fingerprints=source_fingerprints(),
        root_sampling='uniform without replacement cycles',
        selection=('prespecified final nuisance-calibration checkpoint; CV physical transitions fixed' if args.kind=='cv'
                   else 'minimum validation full-horizon physical energy every256steps; final checkpoint also retained'),
        loss='joint trajectory energy + .02 normalized joint Gaussian-mixture NLL + .02 native event NLL + 5 anonymous BCE + .02 detection BCE + .1 actual-observation correction energy every4steps',
        anonymous_prior_scope='one learned time-averaged spatial prior; no anonymous identity trajectories or unknown-risk claim',
        event_exposure_scope='full canonical intervals on roots with all physical humans uniquely associated; partial final intervals omitted; event gradients only calibrate physical hazard offset/temperature')
    protocol_path=run/'config.json'
    if protocol_path.exists():
        previous_protocol=json.loads(protocol_path.read_text())
        # A documentation-only later commit is harmless; training source bytes
        # and every scientific/configuration identity must remain identical.
        protocol['source_commit']=previous_protocol['source_commit']
        if previous_protocol!=protocol:
            raise ValueError('Exact run configuration/source changed; use new run directory')
    save_json(protocol_path,protocol)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no driver repair or CPU fallback')
        torch.cuda.set_device(args.device)
        if torch.cuda.mem_get_info(args.device)[0]<4*1024**3:raise RuntimeError('Insufficient free CUDA memory')
        torch.cuda.reset_peak_memory_stats(args.device)
    elif args.steps>128:
        raise RuntimeError('Substantial training requires measured CUDA resource availability')
    guard=ResearchGuard(run);guard.check(True)
    full=load_split(args.cache,'train',args.device)
    validation=load_split(args.cache,'validation',args.device)
    total_roots=len(full['metadata'])
    model=BeliefWorldModel(config).to(args.device)
    with torch.no_grad():
        for cache in (full,validation):
            cache['prepared_history']=model.prepare_history(cache['base'],
                history_timestamps=cache.get('history_timestamps'),
                track_acquisition_times=cache.get('track_acquisition_times'))
    if args.kind=='cv':
        # Fair nuisance calibration without learning any physical CV transition.
        for name,parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('anonymous_head.') or name in
                ('log_contact_temperature','contact_offset','detection_logit','log_position_scale'))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=.01)
    rng=np.random.default_rng(args.seed)
    order=rng.permutation(total_roots).tolist();cursor=0;step=0
    seen=set();starting=time.time();total_elapsed=0.;best_energy=float('inf');best_step=None
    resume=run/'last.pt'
    if args.resume:
        state=torch.load(resume,map_location=args.device)
        if state['protocol']!=protocol:raise ValueError('Resume provenance mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['sampler_rng'];torch.set_rng_state(state['torch_rng'].cpu())
        if args.device.startswith('cuda'):torch.cuda.set_rng_state(state['cuda_rng'].cpu(),args.device)
        order,cursor,step,seen=state['order'],state['cursor'],state['step'],set(state['seen'])
        total_elapsed=state['elapsed_seconds']
        best_energy=state.get('best_energy',float('inf'));best_step=state.get('best_step')
    elif resume.exists():raise FileExistsError('Use --resume for existing run')
    save_json(run/'architecture.json',model.parameter_counts())

    def checkpoint(status):
        elapsed=total_elapsed+time.time()-starting
        state=dict(protocol=protocol,model=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(args.device) if args.device.startswith('cuda') else None,
            order=order,cursor=cursor,step=step,seen=sorted(seen),elapsed_seconds=elapsed,
            best_energy=best_energy,best_step=best_step)
        temp=run/'last.pt.tmp';torch.save(state,temp);os.replace(temp,run/'last.pt')
        save_json(run/'progress.json',dict(pid=os.getpid(),step=step,target_steps=args.steps,status=status,
            unique_training_roots=len(seen),available_training_roots=total_roots,elapsed_seconds=elapsed,
            steps_per_second=step/max(elapsed,1e-6),snapshot_sha256=manifest['snapshot_sha256']))
        return state

    model.train();metrics={};gradients={}
    try:
        while step<args.steps:
            if (run/'STOP_RESEARCH').exists():
                checkpoint('stopped');return
            guard.check()
            if cursor>=len(order):order=rng.permutation(total_roots).tolist();cursor=0
            indices=order[cursor:cursor+args.batch_size];cursor+=len(indices)
            seen.update(full['metadata'][i]['root_id'] for i in indices)
            batch=select_batch(full,indices)
            optimizer.zero_grad(set_to_none=True)
            loss,metrics,_=loss_bundle(model,batch,args.horizon,correction=step%4==0)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite physical training loss')
            loss.backward()
            if step<4:
                gradients[str(step)]={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite physical gradients')
            optimizer.step();step+=1
            if step%32==0 or step<=4 or step==args.steps:
                record=dict(step=step,elapsed_seconds=total_elapsed+time.time()-starting,
                    gradient_norm=float(norm),unique_roots=len(seen),**metrics)
                with (run/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
            if step%256==0 or step==args.steps:
                validation_score=validation_energy(model,validation,args.horizon,args.batch_size)
                improved=(step==args.steps if args.kind=='cv' else
                    validation_score['energy'] is not None and validation_score['energy']<best_energy)
                if improved:
                    best_energy=validation_score['energy'] if validation_score['energy'] is not None else float('inf')
                    best_step=step
                state=checkpoint('running')
                if improved:
                    torch.save(state,run/'best.pt.tmp');os.replace(run/'best.pt.tmp',run/'best.pt')
                with (run/'validation.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=step,**validation_score,best_step=best_step))+'\n')
        checkpoint('complete')
        save_json(run/'gradients.json',gradients)
        guard.check(True)
        save_json(run/'result.json',dict(protocol=protocol,last_metrics=metrics,
            unique_training_roots=len(seen),root_presentations=args.steps*args.batch_size,
            elapsed_seconds=total_elapsed+time.time()-starting,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None,
            best_step=best_step,best_validation_energy=best_energy if np.isfinite(best_energy) else None,
            selected_checkpoint_sha256=sha256(run/'best.pt') if (run/'best.pt').exists() else None,
            checkpoint_sha256=sha256(run/'last.pt'),status='bounded_training_complete_not_scientific_acceptance'))
    except BaseException as error:
        checkpoint('failed')
        save_json(run/'failure.json',dict(error=repr(error),step=step))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--run',required=True)
    p.add_argument('--kind',choices=['cv','deterministic','stochastic'],default='stochastic')
    p.add_argument('--steps',type=int,default=4096);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--width',type=int,default=128);p.add_argument('--modes',type=int,default=6)
    p.add_argument('--horizon',type=float,default=3.);p.add_argument('--device',default='cuda:1')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();train(args)


if __name__=='__main__':main()
