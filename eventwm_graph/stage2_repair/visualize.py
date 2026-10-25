"""Same-coordinate certificate overlays; no hidden occupancy map is plotted."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .diagnose import OUT,read


def main():
    meta=read(OUT/'trace.json')['examples'];dest=OUT/'figures';dest.mkdir(exist_ok=True)
    fitpath=OUT/'smallfit_original_point_predictions.npz'
    with np.load(fitpath,allow_pickle=False)as fit:
        for i,r in enumerate(meta,1):
            with np.load(OUT/'examples'/('%02d.npz'%i),allow_pickle=False)as z:a={k:z[k]for k in z.files}
            keys=a['keys'];known=a['gkeys'];res=float(a['geometry'][0]);path=a['path'];truth=a['y'];pred=a['prediction'];fitted=fit['prediction_%02d'%i]
            lo=np.minimum(keys.min(0),known.min(0));hi=np.maximum(keys.max(0),known.max(0))+1;shape=hi-lo
            def field(kk,ss):
                x=np.zeros(tuple(shape),np.uint8);ij=kk-lo;x[ij[:,0],ij[:,1]]=ss;return x
            base=field(known,a['gstates']);gt=field(keys,truth);pp=field(keys,pred);fp=field(keys,fitted)
            errors=np.zeros(len(keys),np.uint8);errors[(truth==0)&(pred>0)]=1;errors[(truth>0)&(pred==0)]=2;errors[(truth>0)&(pred>0)&(truth!=pred)]=3
            extent=[lo[0]*res,hi[0]*res,lo[1]*res,hi[1]*res]
            fig,axs=plt.subplots(1,5,figsize=(20,4.5),sharex=True,sharey=True)
            for ax,image,title in zip(axs,[base,gt,pp,field(keys,errors),fp],['Starting legal graph','True certificate increment','Frozen baseline increment','FP red / FN amber / wrong purple','Same-example fitted increment']):
                cmap=ListedColormap(['#f0f0f0','#38a169','#374151']) if ax is not axs[3] else ListedColormap(['#f0f0f0','#e53e3e','#d69e2e','#805ad5'])
                ax.imshow(image.T,origin='lower',extent=extent,interpolation='nearest',cmap=cmap,vmin=0,vmax=2 if ax is not axs[3] else 3)
                ax.plot(path[:,1],path[:,3],color='#2474c6',lw=1.3);step=max(1,len(path)//10);p=path[::step]
                ax.quiver(p[:,1],p[:,3],np.cos(p[:,4]),-np.sin(p[:,4]),angles='xy',scale_units='xy',scale=2,color='#2474c6',width=.006)
                ax.set_title(title,fontsize=9);ax.set_xlabel('world x (m)');ax.set_aspect('equal')
            axs[0].set_ylabel('world z (m)');fig.suptitle('%02d | %s | %s | %d → %d cells | %s [%s, %s]'%(i,r['scene_id'],r['motion'],r['known'],r['target'],r['root_id'][:12],r['start_node'],r['end_node']),fontsize=10)
            fig.tight_layout();fig.savefig(dest/('example_%02d.png'%i),dpi=140);plt.close(fig)
    print('Saved',len(meta),'certificate overlays to',dest)


if __name__=='__main__':main()
