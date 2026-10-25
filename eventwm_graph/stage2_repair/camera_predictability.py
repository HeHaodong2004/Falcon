"""Offline-only test of camera/base mismatch recoverability from legal poses.

Camera truth is a diagnostic response, never a reconstruction-model input.
The fitted diagnostic regression is not loaded by a deployed predictor.
"""
import math
import numpy as np
from .diagnose import OUT,HF,read,save,source_rows,member
from .geometry_diagnostic import camera


def features(ob,previous):
    p=np.asarray(ob['robot_pose']);q=np.asarray(previous['robot_pose']);dt=ob['timestamp']-previous['timestamp'];dy=math.atan2(math.sin(p[3]-q[3]),math.cos(p[3]-q[3]));delta=p[:3]-q[:3];t=ob['timestamp'];yaw=p[3]
    values=[1.,math.cos(yaw),math.sin(yaw),math.cos(2*yaw),math.sin(2*yaw),dy/max(dt,1e-6),np.linalg.norm(delta[[0,2]])/max(dt,1e-6),delta[1]/max(dt,1e-6)]
    for frequency in [.5,1.,2.,3.,4.,5.,6.,8.,10.]:values.extend([math.sin(2*math.pi*frequency*t),math.cos(2*math.pi*frequency*t)])
    return values


def collect(rows):
    x=[];y=[]
    for row in rows:
        legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz');initial=legal['history'][-1]
        for n in nodes:
            ob=n['observation'];previous=initial if n['parent']is None else nodes[n['parent']]['observation'];v=camera(ob);v[3]=math.atan2(math.sin(v[3]),math.cos(v[3]));x.append(features(ob,previous));y.append(v)
    return np.asarray(x,np.float64),np.asarray(y,np.float64)


def main():
    rows,_=source_rows('train');ids=set(read(HF/'protocol.json')['pilot_fit']);train=[r for r in rows if r['root_id']in ids][:256];dev,_=source_rows('dev');ids=set(read(HF/'protocol.json')['pilot_selection_roots']);dev=[r for r in dev if r['root_id']in ids]
    x,y=collect(train);vx,vy=collect(dev);mean=y.mean(0);ridge=1e-3*np.eye(x.shape[1]);ridge[0,0]=0;w=np.linalg.solve(x.T@x+ridge,x.T@y);pred=vx@w
    before=np.sqrt(np.mean((vy-mean)**2,axis=0));after=np.sqrt(np.mean((vy-pred)**2,axis=0));r2=1-np.sum((vy-pred)**2,axis=0)/np.sum((vy-mean)**2,axis=0)
    result=dict(train_roots=len(train),train_frames=len(x),dev_roots=len(dev),dev_frames=len(vx),columns=['forward_offset_m','vertical_offset_m','lateral_offset_m','yaw_rad','pitch_rad','roll_rad'],constant_calibration_rmse=before.tolist(),legal_pose_clock_linear_rmse=after.tolist(),r2=r2.tolist(),
        scope='Diagnostic only, scene-disjoint train/dev. Actual future camera as response; no camera field enters features or any production model. This linear basis tests predictability, not an irreducible ceiling.')
    save(OUT/'camera_predictability.json',result);print(result,flush=True)


if __name__=='__main__':main()
