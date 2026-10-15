"""Single-scene inference from legal history only; no future/cache targets read."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from eventquerywm.inputs import tensorize,collate
from eventforecastwm.benchmark import load_model
from eventforecastwm.data import nominal_query
from eventforecastwm.sampling import joint_samples
from eventforecastwm.readouts import tracked_occupancy
from eventbeliefwm.data import sha256


def select_static_geometry(model,static_geometry):
    """Use maps only through an explicitly trained physical or occupancy path.

    Historical checkpoints retain an unused compatibility geometry branch. The
    development benchmark supplies unavailable geometry to those checkpoints;
    supplying a map at deployment must not silently activate its random weights.
    """
    trained=(model.recipe.get('variant')=='geometry_occupancy_v2' or
             hasattr(model.core,'map_patch_encoder'))
    return static_geometry if trained else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--history',required=True)
    p.add_argument('--sensor');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--cpu-preparation',action='store_true',help='Prepare parameter-free legal history features on CPU before model inference')
    maps=p.add_mutually_exclusive_group();maps.add_argument('--static-map',help='NPZ with features[2,32,32], support[32,32] in legal root frame')
    maps.add_argument('--navmesh',help='Known static navmesh; derive features at legal robot pose')
    p.add_argument('--horizon',type=float,default=3.);p.add_argument('--samples',type=int,default=6);p.add_argument('--seed',type=int,default=7)
    a=p.parse_args();out=Path(a.output)
    if out.suffix!='.npz':raise ValueError('Output must explicitly end in.npz')
    if out.exists():raise FileExistsError(out)
    history=json.loads(Path(a.history).read_text())
    cpu_legal=collate([tensorize(dict(history=history,history_images=None,query=nominal_query(history)))])
    legal=cpu_legal.to(a.device)
    ids=sorted({t['track_id'] for h in history for t in h['human_tracks']})
    cpu_clocks=torch.tensor([[h['timestamp'] for h in history]],dtype=torch.float64)
    cpu_acquisition=torch.zeros(1,max(1,len(ids)),len(history),dtype=torch.float64)
    for j,h in enumerate(history):
        for tr in h['human_tracks']:cpu_acquisition[0,ids.index(tr['track_id']),j]=tr['last_seen_time']
    clocks=cpu_clocks.to(a.device);acquisition=cpu_acquisition.to(a.device)
    sensor=json.loads(Path(a.sensor).read_text()) if a.sensor else {}
    static_map=None
    if a.static_map:
        source=np.load(a.static_map)
        static_map=dict(features=torch.as_tensor(source['features'],device=a.device)[None],
            support=torch.as_tensor(source['support'],device=a.device,dtype=torch.bool)[None])
    elif a.navmesh:
        from eventforecastwm.geometry import extract_frame
        features=extract_frame(a.navmesh,history[-1]['robot_position'],history[-1]['robot_yaw'])
        static_map=dict(features=features[None].to(a.device),support=torch.ones(1,32,32,dtype=torch.bool,device=a.device))
    model=load_model(a.checkpoint).to(a.device).eval()
    conditioning_geometry=select_static_geometry(model,static_map)
    with torch.no_grad():
        prepared=model.core.prepare_history(cpu_legal,history_timestamps=cpu_clocks,
            track_acquisition_times=cpu_acquisition).to(a.device) if a.cpu_preparation else None
        belief=model.encode_history(legal,conditioning_geometry,sensor,{'radius':.25},history_timestamps=clocks,
            track_acquisition_times=acquisition,prepared_history=prepared)
        forecast=model.imagine(belief,horizon_s=a.horizon)
        paths,modes=joint_samples(forecast,a.samples,torch.Generator(device=a.device).manual_seed(a.seed))
        tracked=tracked_occupancy(forecast)
        checkpoint=torch.load(a.checkpoint,map_location='cpu');prior=checkpoint.get('anonymous_occupancy_prior')
        horizon_steps=forecast.positions.shape[2]
        anonymous_supported=prior is not None and len(prior)>=horizon_steps
        if hasattr(forecast,'anonymous_occupancy'):
            anonymous_supported=bool(forecast.anonymous_occupancy_supported.all())
            combined=1-(1-tracked)*(1-forecast.anonymous_occupancy)
        else:
            combined=(1-(1-tracked)*(1-prior[:horizon_steps].to(a.device)[None])) if anonymous_supported else tracked
    out.parent.mkdir(parents=True,exist_ok=True)
    distribution=dict(position_variance=forecast.position_variance.cpu().numpy())
    if hasattr(forecast,'anonymous_occupancy'):
        distribution.update(anonymous_center_occupancy=forecast.anonymous_occupancy.cpu().numpy(),
            anonymous_occupancy_supported=forecast.anonymous_occupancy_supported.cpu().numpy())
    if hasattr(forecast,'concentration'):
        distribution.update(scene_scale_concentration=forecast.concentration.cpu().numpy(),
            scene_scale_rate=forecast.rate.cpu().numpy())
    if getattr(forecast,'decay_rate',None) is not None:
        distribution['residual_velocity_decay_rate']=forecast.decay_rate.cpu().numpy()
    if hasattr(forecast,'agent_scale_probabilities'):
        distribution.update(agent_variance_values=forecast.scale_values.cpu().numpy(),
            agent_variance_probabilities=forecast.agent_scale_probabilities.cpu().numpy())
        if hasattr(forecast,'branch_probabilities'):
            rho=torch.stack([forecast.agent_scale_probabilities,forecast.stopped_scale_probabilities],-2)
            marginal=(rho*forecast.branch_probabilities[...,None]).sum(-2)
            distribution.update(agent_variance_probabilities=marginal.cpu().numpy(),
                agent_regime_probabilities=forecast.branch_probabilities.cpu().numpy(),
                agent_regime_variance_probabilities=rho.cpu().numpy(),
                agent_regime_names=np.asarray(['learned','stopped']),
                regime_positions=torch.stack([forecast.moving_positions,forecast.stopped_positions],2).cpu().numpy(),
                regime_velocities=torch.stack([forecast.moving_velocities,forecast.stopped_velocities],2).cpu().numpy(),
                positions_semantics='scene-conditional mean over persistent agent regimes')
    elif hasattr(forecast,'scale_values'):
        distribution.update(scene_variance_values=forecast.scale_values.cpu().numpy(),
            scene_variance_probabilities=forecast.scale_probabilities.cpu().numpy())
    np.savez_compressed(out,positions=forecast.positions.cpu().numpy(),samples=paths.cpu().numpy(),
        modes=modes.cpu().numpy(),weights=forecast.weights.cpu().numpy(),covariance=forecast.covariance.cpu().numpy(),
        readout_scale=forecast.readout_scale.cpu().numpy(),valid=forecast.valid.cpu().numpy(),times=forecast.times.cpu().numpy(),
        tracked_center_occupancy=tracked.cpu().numpy(),combined_center_occupancy=combined.cpu().numpy(),
        anonymous_prior_available=anonymous_supported,geometry_available=static_map is not None,
        geometry_conditioning_used=conditioning_geometry is not None,
        within_trained_horizon=forecast.validity['within_trained_horizon'].cpu().numpy(),
        legal_track_ids=np.asarray(ids),frame_origin=np.asarray(history[-1]['robot_position']),frame_yaw=history[-1]['robot_yaw'],
        covariance_semantics='unscaled within-branch residual position/velocity covariance; categorical laws are exported separately',**distribution)
    print(json.dumps(dict(output=str(out),checkpoint_sha256=sha256(a.checkpoint),legal_history_sha256=sha256(a.history),
        cpu_preparation=a.cpu_preparation,
        geometry_conditioning_used=conditioning_geometry is not None,
        inputs='arrived legal observations, exact acquisition timestamps, known robot geometry; no future states',
        uncertainty='categorical joint scene modes plus explicitly sampled trajectory residuals',
        occupancy='human-center cell marginals; combined uses empirical anonymous prior only when available, never a certified free-space map',
        limitations='existing legal tracks; identity birth/reassociation and full anonymous dynamics remain separate')))


if __name__=='__main__':main()
