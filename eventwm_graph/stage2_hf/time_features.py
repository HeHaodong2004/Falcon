"""Clock-shift invariant legal observation-recency angular covariates.

Only starting-map cell identities/states/last_seen and the supplied first pose
are used. Angular moments are soft learned covariates, never occlusion masks.
"""
import numpy as np
import torch

def moment_matrix(keys,states,last,now,position,resolution):
 last=np.asarray(last,np.float64);age=float(now)-last
 if not np.isfinite(last).all()or np.any(age < -1e-7):raise ValueError('future or nonfinite known observation time')
 age=np.maximum(age,0);xy=(keys.astype(np.float64)+.5)*resolution-np.asarray(position,np.float64);norm=np.maximum(np.linalg.norm(xy,axis=1),1e-6);u=xy[:,0]/norm;v=xy[:,1]/norm
 angular=np.column_stack([u,v,u*u-v*v,2*u*v,u*u*u-3*u*v*v,3*u*u*v-v*v*v]);matrix=np.zeros((7,32),np.float64);column=0
 for state in (1,2):
  select=states==state;count=max(int(select.sum()),1)
  for decay in ((age<=.10000001).astype(np.float64),np.exp(-age/.5)):
   weight=decay*select;total=weight.sum();mom=(angular*weight[:,None]).sum(0)/max(total,1e-12);matrix[0,column]=total/count;column+=1
   for order in range(3):
    a,b=mom[2*order:2*order+2];i=1+2*order;matrix[i,column]=a;matrix[i+1,column]=b;column+=1;matrix[i,column]=-b;matrix[i+1,column]=a;column+=1
 matrix[0,28]=np.log1p(np.count_nonzero(states==1))/10;matrix[0,29]=np.log1p(np.count_nonzero(states==2))/10
 for j,state in enumerate((1,2)):
  select=states==state;matrix[0,30+j]=np.minimum(age[select],5).mean()/5 if np.any(select)else 1.
 return matrix.astype(np.float32)

def basis_numpy(relative_xy):
 x=np.asarray(relative_xy,np.float32);n=np.maximum(np.linalg.norm(x,axis=1),1e-6);u=x[:,0]/n;v=x[:,1]/n
 return np.column_stack([np.ones(len(x),np.float32),u,v,u*u-v*v,2*u*v,u*u*u-3*u*v*v,3*u*u*v-v*v*v]).astype(np.float32)

def basis_torch(x):
 xy=x[:,20:22];n=xy.square().sum(1).sqrt().clamp_min(1e-6);u=xy[:,0]/n;v=xy[:,1]/n
 return torch.stack([torch.ones_like(u),u,v,u*u-v*v,2*u*v,u*u*u-3*u*v*v,3*u*u*v-v*v*v],-1)
