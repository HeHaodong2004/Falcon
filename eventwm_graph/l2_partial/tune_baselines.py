"""Bounded dev-only CV diffusion selection, with proper distribution scores."""
import argparse,json,pathlib
import numpy as np
from .evaluate import weighted_energy

def tune(dev,output):
    rows=[r for r in np.load(dev,allow_pickle=True) if r['valid'].any()];results=[]
    for diffusion in (.01,.04,.16,.64):
        scores=[]
        for r in rows:
            rng=np.random.default_rng(19);values=[]
            for j,t in enumerate((.5,1.,2.,3.)):
                select=r['valid'][:,j]
                if not select.any():continue
                m=r['mean'][select];C=r['covariance'][select];F=np.c_[np.eye(2),t*np.eye(2)]
                cov=F@C@F.T+np.eye(2)[None]*(diffusion*t**3/3+1e-8)
                sample=t*m[None,:,2:]+np.einsum('nij,wnj->wni',np.linalg.cholesky(cov),rng.normal(size=(64,len(m),2)))
                values.append(weighted_energy(sample,r['targets'][select,j],np.full(64,1/64)))
            scores.append(float(np.mean(values)))
        results.append({'diffusion':diffusion,'dev_energy_score':float(np.mean(scores)),'roots':len(rows)})
    best=min(results,key=lambda r:r['dev_energy_score']);pathlib.Path(output).write_text(json.dumps({'grid':results,'selected':best},indent=2));print(results)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dev');p.add_argument('output');a=p.parse_args();tune(**vars(a))
