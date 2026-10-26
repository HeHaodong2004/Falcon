"""Same-world-cell certificate plots, including every positive/error cell."""
from collections import defaultdict
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from .diagnose import OUT,HF,read,save,source_rows
from eventwm_graph.stage2_segment.data import replay
from eventwm_graph.stage2_hf.packed import load_arrays
from scripts.stage2_diagnostic.common import extended_rates


def run(source=None,name='figures_current_best'):
    source=Path(source)if source else OUT/'range_runtime_spatial_stride1_radix';summary=read(source/'summary.json');examples=[]
    for path in sorted(source.glob('*.npz')):
        rid=path.stem.rsplit('_',1)[0];ordinal=int(path.stem.rsplit('_',1)[1]);record=read(source/(rid+'.json'))['records'][ordinal-1]
        with np.load(path)as z:a={k:z[k]for k in z.files}
        cm=np.bincount(a['y'].astype(int)*3+a['prediction'],minlength=9).reshape(3,3)
        cm[1,0]+=record['out_of_domain'];rates=extended_rates(cm)
        p=a['path'];distance=float(np.linalg.norm(np.diff(p[:,[1,3]],axis=0),axis=1).sum());rotation=float(np.abs(np.arctan2(np.sin(np.diff(p[:,4])),np.cos(np.diff(p[:,4])))).sum())
        examples.append(dict(root_id=rid,ordinal=ordinal,record=record,archive=str(path),true_count=int((a['y']>0).sum()+record['out_of_domain']),pred_count=int((a['prediction']>0).sum()),iou=rates['label_aware_delta_iou'],distance=distance,rotation=rotation))
    nonempty=[e for e in examples if e['true_count']>=50]
    choices=[('success',max(nonempty,key=lambda e:e['iou'])),('failure',min(nonempty,key=lambda e:e['iou']))]
    mixed=[e for e in nonempty if e['distance']>.1 and e['rotation']>.1]
    if mixed:choices.append(('mixed_motion',sorted(mixed,key=lambda e:e['iou'])[len(mixed)//2]))
    rotation=[e for e in nonempty if e['distance']<.02 and e['rotation']>.1]
    if rotation:choices.append(('rotation',min(rotation,key=lambda e:e['iou'])))
    empty=[e for e in examples if e['true_count']==0]
    if empty:choices.append(('no_new_information',max(empty,key=lambda e:e['pred_count'])))
    raw,_=source_rows('dev');raw={r['root_id']:r for r in raw};dest=OUT/name;dest.mkdir(exist_ok=True);reports=[]
    for label,e in choices:
        with np.load(e['archive'])as z:a={k:z[k]for k in z.files}
        target=None
        for ex in replay(raw[e['root_id']],'mixed'):
            if 'audit'in ex:continue
            if (ex['start_node'],ex['end_node'])==(e['record']['start_node'],e['record']['end_node']):target=ex['target'];break
        assert target is not None
        known={tuple(k):int(s)for k,s in zip(a['gkeys'],a['gstates'])};pred={tuple(k):int(s)for k,s in zip(a['keys'],a['prediction'])if s}
        truth={k:int(v[0])for k,v in target.items()}
        assert not (known.keys()&truth.keys())and not(known.keys()&pred.keys())
        independent_y=np.array([truth.get(tuple(k),0)for k in a['keys']],np.uint8);np.testing.assert_array_equal(independent_y,a['y'])
        union=set(map(tuple,a['keys']))|truth.keys();cm=np.zeros((3,3),np.int64)
        for key in union:cm[truth.get(key,0),pred.get(key,0)]+=1
        rates=extended_rates(cm)
        packed=read(HF/'packed/dev_pilot'/(e['root_id']+'.json'));resolution=float(load_arrays(packed)['geometry'][0])
        allkeys=np.array(list(known.keys()|truth.keys()|pred.keys()),np.int32)
        pathkeys=np.floor(a['path'][:,[1,3]]/resolution).astype(np.int32);allkeys=np.concatenate([allkeys,pathkeys])
        low=allkeys.min(0)-10;high=allkeys.max(0)+11;shape=high-low
        base=np.full((shape[1],shape[0],3),.86,np.float32)
        for key,state in known.items():x,z=np.array(key)-low;base[z,x]=[.98,.98,.98]if state==1 else[.20,.20,.20]
        target_image=base*.45+.55;pred_image=target_image.copy();errors=target_image.copy()
        palette={1:np.array([.02,.65,.35]),2:np.array([.02,.02,.02])}
        for key,state in truth.items():x,z=np.array(key)-low;target_image[z,x]=palette[state]
        for key,state in pred.items():x,z=np.array(key)-low;pred_image[z,x]=palette[state]
        counts=defaultdict(int)
        for key in truth.keys()|pred.keys():
            t,p=truth.get(key,0),pred.get(key,0)
            if t==p:color=[.55,.85,.60];counts['correct_new']+=1
            elif t and not p:color=[.02,.52,.87];counts['missed_new']+=1
            elif p and not t:color=[.80,.10,.65];counts['false_new']+=1
            else:color=[.97,.50,.08];counts['wrong_class']+=1
            x,z=np.array(key)-low;errors[z,x]=color
        extent=[low[0]*resolution,high[0]*resolution,low[1]*resolution,high[1]*resolution]
        fig,axes=plt.subplots(1,4,figsize=(16,4.9),sharex=True,sharey=True)
        titles=['Current legal graph','True new certificates','Hypothetical predicted increment','False / missed / wrong class']
        for ax,image,title in zip(axes,[base,target_image,pred_image,errors],titles):
            ax.imshow(image,origin='lower',extent=extent,interpolation='nearest');p=a['path'];ax.plot(p[:,1],p[:,3],color='#1957be',linewidth=1.5)
            ax.quiver(p[:,1],p[:,3],np.cos(p[:,4]),-np.sin(p[:,4]),color='#1957be',angles='xy',scale_units='xy',scale=2.3,width=.005)
            ax.scatter(p[0,1],p[0,3],s=24,color='#ffd32a',edgecolor='black',zorder=5);ax.set_title(title,fontsize=11);ax.set_xlabel('world x (m)');ax.set_aspect('equal')
        axes[0].set_ylabel('world z (m)')
        caption=('Empty target; predicted%d new cells'%len(pred))if not truth else'Label-aware IoU %.3f; revelation IoU %.3f'%(rates['label_aware_delta_iou'],rates['revelation_iou'])
        fig.suptitle('%s | Root %s | %s'%(label,e['root_id'][:12],caption),fontsize=12)
        handles=[Patch(color=palette[1],label='certificate FREE'),Patch(color=palette[2],label='certificate BLOCKED'),Patch(color=[.02,.52,.87],label='missed new'),Patch(color=[.80,.10,.65],label='false new'),Patch(color=[.97,.50,.08],label='wrong FREE/BLOCKED')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.01),ncol=5,frameon=False,fontsize=9);fig.tight_layout(rect=[0,.12,1,.93]);output=dest/(label+'.png');fig.savefig(output,dpi=170);plt.close(fig)
        reports.append(dict(category=label,root_id=e['root_id'],start_node=e['record']['start_node'],end_node=e['record']['end_node'],resolution=resolution,confusion=cm.tolist(),metrics=rates,errors=dict(counts),out_of_domain_positives=e['record']['out_of_domain'],figure=str(output)))
    save(dest/'examples.json',dict(role='Actual observation-certificate labels, not physical occupancy; all positive/error cells shown on identical world-cell axes. Metrics include original full candidate domain and out-of-domain positives.',checkpoint=summary['checkpoint'],checkpoint_sha256=summary['checkpoint_sha256'],records=reports))
    print([(r['category'],r['metrics']['label_aware_delta_iou'],r['errors'])for r in reports],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--name',default='figures_current_best');run(**vars(p.parse_args()))
