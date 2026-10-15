"""Secondary train-only scalar emission calibration; physical dynamics stay fixed."""
import argparse
import math
from pathlib import Path
import torch
from eventbeliefwm.model import BeliefWorldModel,BeliefConfig
from eventbeliefwm.experiment import load_split,select_batch,encode,stationary_query,save_json,source_commit
from eventbeliefwm.data import sha256


def profile_scale(squared_error,count,weights):
    """Minimize root-average joint-mixture NLL over one shared residual scale.

    A log grid finds a bracket before bounded golden refinement. The scalar
    changes likelihood and occupancy emission only, not persistent scenarios,
    transition states, collision risk or the observation correction gain.
    """
    squared_error,count,weights=[v.double().cpu() for v in (squared_error,count,weights)]
    if not len(count) or (count<=0).any():raise ValueError('Calibration needs exposed training trajectories')
    def score(log_sigma):
        joint=-.5*squared_error/math.exp(2*log_sigma)-count[:,None]*(log_sigma+.5*math.log(2*math.pi))
        return float((-torch.logsumexp(weights.clamp_min(1e-12).log()+joint,1)/count).mean())
    grid=torch.linspace(math.log(.03),math.log(3.),81).tolist()
    scores=[score(x) for x in grid];index=min(range(len(grid)),key=scores.__getitem__)
    left,right=grid[max(0,index-1)],grid[min(len(grid)-1,index+1)]
    ratio=(math.sqrt(5)-1)/2
    for _ in range(48):
        a=right-ratio*(right-left);b=left+ratio*(right-left)
        if score(a)<score(b):right=b
        else:left=a
    optimum=(left+right)/2
    return optimum,score(optimum),score


@torch.no_grad()
def calibrate(checkpoint,cache,output,horizon=3.):
    checkpoint,cache,output=Path(checkpoint),Path(cache),Path(output)
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError(output)
    state=torch.load(checkpoint,map_location='cpu')
    model=BeliefWorldModel(BeliefConfig(**state['protocol']['config'])).eval()
    model.load_state_dict(state['model'])
    full=load_split(cache,'train','cpu')
    full['prepared_history']=model.prepare_history(full['base'],history_timestamps=full['history_timestamps'],
        track_acquisition_times=full['track_acquisition_times'])
    errors=[];counts=[];weights=[];roots=[]
    for offset in range(0,len(full['metadata']),8):
        batch=select_batch(full,list(range(offset,min(offset+8,len(full['metadata'])))))
        forecast=model.imagine(encode(model,batch),stationary_query(len(batch['metadata']),horizon,'cpu'),
                               horizon,rasterize=False)
        length=forecast.human_positions.shape[2]
        valid=batch['targets']['track_valid'][:,:length]
        valid=valid & valid.all(1)[:,None]
        count=valid.sum((1,2))*2;supported=count>0
        error=((forecast.human_positions-batch['targets']['track_positions'][:,None,:length]).square()
               *valid[:,None,:,:,None]).sum((2,3,4))
        errors.append(error[supported]);counts.append(count[supported]);weights.append(forecast.scenario_weights[supported])
        roots.extend(m['root_id'] for m,keep in zip(batch['metadata'],supported.tolist()) if keep)
    optimum,after,score=profile_scale(torch.cat(errors),torch.cat(counts),torch.cat(weights))
    before=float(model.log_position_scale)
    model.log_position_scale.fill_(optimum)
    info=dict(method='train-only root-balanced one-dimensional joint-emission NLL profile',
        source_commit=source_commit(),script_sha256=sha256(Path(__file__)),parent_checkpoint=str(checkpoint),
        parent_checkpoint_sha256=sha256(checkpoint),cache_manifest_sha256=sha256(cache/'manifest.json'),
        training_root_ids=roots,horizon_s=horizon,training_roots=len(roots),
        sigma_before_m=math.exp(before),sigma_after_m=math.exp(optimum),
        training_nll_before=score(before),training_nll_after=after,
        changed_parameters=['log_position_scale'],physical_trajectories_changed=False,collision_risk_changed=False,
        scope='Secondary emission-score fairness check; no physical uncertainty or event calibration claim',
        resumable_training_checkpoint=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),protocol=state['protocol'],step=state['step'],postfit_calibration=info),output)
    info['checkpoint_sha256']=sha256(output);save_json(output.with_suffix('.json'),info)
    print({k:v for k,v in info.items() if k!='training_root_ids'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--horizon',type=float,default=3.)
    args=parser.parse_args();calibrate(args.checkpoint,args.cache,args.output,args.horizon)
