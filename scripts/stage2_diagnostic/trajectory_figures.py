"""Cached-map figures only: fixed-hash and labeled strongest/weakest cases.

This module never imports a model or recomputes inference. Full map caches and
original F32 probability arrays own the evidence behind each exported figure.
"""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
from .common import OUT,digest,save


def main(name='primary_v1'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from eventwm_graph.stage2_segment.evaluate import rates
    root=OUT/'trajectory'/name
    summary=json.loads((root/'summary.json').read_text());selected={}
    probability_summary=json.loads((OUT/'probability/summary.json').read_text());rule=probability_summary['diagnostic_scene_validation']['selected_rules']['graphdelta_iou']
    theta=np.float64(rule.split('_',1)[1]);assert theta==np.float64(.20)
    for group,ids in summary['visual_selection'].items():
        if isinstance(ids,list):
            for rid in ids:selected.setdefault(rid,[]).append(group)
    dest=root/'figures';dest.mkdir(exist_ok=False)
    artifacts=[]
    for rid,kinds in selected.items():
        record=json.loads((root/'roots'/(rid+'.json')).read_text());cache=root/'roots'/(rid+'.npz');maps=root/'roots'/(rid+'.maps.json.gz')
        assert digest(cache)==record['cache_sha256'] and digest(maps)==record['maps_cache_sha256']
        with gzip.open(maps,'rt') as f:graphs=json.load(f)
        with np.load(cache,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        keys=arrays['keys'];resolution=float(graphs['current_legal_graph']['resolution']);lo=keys.min(0)-2;hi=keys.max(0)+3;shape=hi-lo
        if int(shape[0])*int(shape[1])>4000000:raise ValueError('plot domain too large')
        extent=(lo[0]*resolution,hi[0]*resolution,lo[1]*resolution,hi[1]*resolution)
        ij=keys-lo;known=np.zeros((shape[1],shape[0]),np.uint8)
        for row in graphs['current_legal_graph']['cells']:
            x,z=np.asarray(row['cell'])-lo
            if 0<=x<shape[0] and 0<=z<shape[1]:known[z,x]=row['state']
        pa=np.asarray(record['pair']['trajectory_a']);pb=np.asarray(record['pair']['trajectory_b'])
        fig,axes=plt.subplots(4,3,figsize=(15,17),sharex=True,sharey=True)
        def base(ax):
            ax.imshow(known,origin='lower',extent=extent,cmap=ListedColormap(['white','#d8d8d8','#5c5c5c']),vmin=0,vmax=2,interpolation='nearest')
            ax.plot(pa[:,1],pa[:,3],'-',color='#2166ac',lw=1.6,label='A')
            ax.plot(pb[:,1],pb[:,3],'-',color='#ef8a62',lw=1.6,label='B')
            for path,color in ((pa,'#2166ac'),(pb,'#ef8a62')):
                ax.arrow(path[-1,1],path[-1,3],.6*np.cos(path[-1,4]),-.6*np.sin(path[-1,4]),color=color,width=.025,head_width=.17,length_includes_head=True)
            ax.set_aspect('equal');ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.tick_params(labelsize=7)
        def labels(ax,value,title):
            base(ax);image=np.full((shape[1],shape[0]),np.nan);mask=value>0;image[ij[mask,1],ij[mask,0]]=value[mask]
            ax.imshow(image,origin='lower',extent=extent,cmap=ListedColormap(['#31a354','#d73027']),vmin=1,vmax=2,interpolation='nearest')
            ax.set_title(title,fontsize=10)
        def probability(ax,value,title,delta=False,mask=None):
            base(ax);image=np.full((shape[1],shape[0]),np.nan);valid=np.ones(len(keys),bool) if mask is None else mask
            image[ij[valid,1],ij[valid,0]]=value[valid]
            im=ax.imshow(image,origin='lower',extent=extent,cmap='RdBu_r' if delta else 'viridis',vmin=-1 if delta else 0,vmax=1,interpolation='nearest')
            ax.set_title(title,fontsize=10);fig.colorbar(im,ax=ax,fraction=.035,pad=.02)
        base(axes[0,0]);axes[0,0].set_title('Common legal starting map + real paths',fontsize=10)
        axes[0,0].legend(handles=[Patch(color='#2166ac',label='Path A'),Patch(color='#ef8a62',label='Path B'),Patch(color='#31a354',label='New FREE'),Patch(color='#d73027',label='New BLOCKED')],fontsize=6,loc='lower left')
        labels(axes[0,1],arrays['y_a'],'Recorded new observations A')
        labels(axes[0,2],arrays['y_b'],'Recorded new observations B')
        labels(axes[1,0],arrays['production_a'].argmax(1),'Production hard prediction A (raw argmax)')
        labels(axes[1,1],arrays['production_b'].argmax(1),'Production hard prediction B (raw argmax)')
        intersection=arrays['support_a']&arrays['support_b']
        qa=arrays['direct_a'][:,1]+arrays['direct_a'][:,2];qb=arrays['direct_b'][:,1]+arrays['direct_b'][:,2]
        probability(axes[1,2],qa-qb,'Direct network revelation difference A-B on I',True,intersection)
        def threshold(p):
            reveal=(p[:,1]+p[:,2]).astype(np.float64)
            return np.where(reveal>=theta,np.where(p[:,1]>=p[:,2],1,2),0)
        labels(axes[2,0],threshold(arrays['production_a']),'A-selected theta=0.20 hard A (NOT production)')
        labels(axes[2,1],threshold(arrays['production_b']),'A-selected theta=0.20 hard B (NOT production)')
        probability(axes[2,2],(arrays['y_a']>0).astype(float)-(arrays['y_b']>0).astype(float),'Recorded revelation difference A-B',True)
        probability(axes[3,0],qa,'Direct fixed-U P(revelation | A) [extended U]')
        probability(axes[3,1],qb,'Direct fixed-U P(revelation | B) [extended U]')
        qn=arrays['neutral'][:,1]+arrays['neutral'][:,2]
        probability(axes[3,2],qn,'NEUTRAL INPUT / OOD / NO RECORDED GT')
        for ax in axes[-1,:]:ax.set_xlabel('world x (m)',fontsize=8)
        for ax in axes[:,0]:ax.set_ylabel('world z (m)',fontsize=8)
        mode=record['modes']['direct_fixed_union_I'];own=rates(np.array(mode['own_confusion']))['label_aware_delta_iou'];cross=rates(np.array(mode['cross_confusion']))['label_aware_delta_iou']
        group='; '.join(kinds)
        fig.suptitle(f'{rid[:16]} | {group}\nreal matched horizon {record["duration"]:.3f}s / {record["poses"]} poses / start={record["pair"]["start_node"]}; '
                     f'direct-I CE gain {mode["gain"][0]:+.5f}, Brier gain {mode["gain"][1]:+.5f}; own/cross IoU {own:.4f}/{cross:.4f}',fontsize=11)
        fig.text(.5,.012,'Both branches have recorded legal targets; neutral has none. U is candidate union; I is candidate intersection. '
                 'Known-cell corrections excluded from displayed increments. Fixed-hash cases and outcome-selected extremes are named explicitly. '
                 'Probability colors are linear [0,1]; difference colors [-1,1].',ha='center',fontsize=7,wrap=True)
        fig.tight_layout(rect=(0,.035,1,.95))
        pdf=dest/(rid+'.pdf');png=dest/(rid+'.png');fig.savefig(pdf);fig.savefig(png,dpi=150);plt.close(fig)
        artifacts.append(dict(root_id=rid,selection_labels=kinds,pdf=str(pdf),pdf_sha256=digest(pdf),png=str(png),png_sha256=digest(png),
            input_probability_sha256=record['cache_sha256'],input_maps_sha256=record['maps_cache_sha256'],direct_I_gain=mode['gain']))
    save(dest/'manifest.json',dict(script_sha256=digest(__file__),supplemental_threshold_rule=rule,supplemental_rule_source_sha256=digest(OUT/'probability/summary.json'),primary_B_metrics_unchanged=True,summary_sha256=digest(root/'summary.json'),selection=summary['visual_selection'],figures=artifacts,
        all_probabilities_loaded_from_exact_caches=True,model_inference_calls=0,display_scope='Common candidate-union bounding view; scores use complete declared domains.'))
    print(json.dumps(dict(figures=len(artifacts),folder=str(dest))),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--name',default='primary_v1');main(p.parse_args().name)
