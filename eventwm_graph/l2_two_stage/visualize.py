"""Offline real-development figures. No future truth enters prediction."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Wedge
from matplotlib.collections import PolyCollection
from .common import OUT,DOC,dump,sha,seed_for
from .context import OnlineContext
from .data import load_records,jobs_for,read,selected_branch,native_index
from .features import positives,extract
from .model import Stage1WorldModel


def run(checkpoint,split='dev_native_v2',name='stage1_examples'):
    records=load_records(split);strata={}
    with gzip.open(DOC/'reviews/native_occlusion_development/anchors.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);strata[(row['root_id'],row['anchor_index'])]=row
    # Illustration choices are fixed by legal/status metadata and native target
    # availability, never by forecast error or a handpicked good-looking result.
    predicates={
        'visible':lambda r,s:bool(np.any(r['age']<1e-7)) and r['times'][3]>0,
        'native_gap':lambda r,s:bool(np.any((r['age']>=.75-1e-7)&(r['age']<=1.25+1e-7)&(r['physical']>=0))) and r['anchor_index']>0 and r['times'][3]>0,
        'empty_with_occlusion':lambda r,s:s.get('cohort_supported') and s.get('valid_empty') and s.get('has_native_occlusion') and r['times'][1]>0,
    }
    selected={key:next(r for r in records if predicate(r,strata.get((r['root_id'],r['anchor_index']),{}))) for key,predicate in predicates.items()}
    folders={r['root_id']:Path(folder) for folder,r in jobs_for('dev')}
    from .conditioned import HistoryConditionedWorldModel,ConditionedHumanWorlds
    model=(HistoryConditionedWorldModel(checkpoint) if 'conditioned_config' in torch.load(checkpoint,map_location='cpu') else Stage1WorldModel(checkpoint))
    dest=DOC/'figures'/name;dest.mkdir(parents=True,exist_ok=False);reports=[]
    plt.rcParams.update({'font.size':9})
    for kind,r in selected.items():
        folder=folders[r['root_id']];legal=read(folder/'legal_root.json.gz');_,_,branch=selected_branch(folder)
        context=OnlineContext.from_raw(legal['history'],legal['partial_graph'],legal['history_actions'])
        if isinstance(model,HistoryConditionedWorldModel):context=context.with_association()
        for node in branch[:r['anchor_index']]:context=context.ingest(node['observation'],node['action'])
        assert abs(context.history[-1]['timestamp']-r['timestamp'])<1e-7
        f=extract(context);horizon=float(r['times'][r['times']>0].max())
        bank=model.predict(context,context.history[-1],horizon=min(3.,horizon),worlds=32,seed=seed_for(r['root_id'],41,r['anchor_index']))
        with np.load(folder/'privileged_future.npz') as z:native_times=z['timestamps'].copy();truth=z['positions'][...,[0,2]].copy()
        current_index=native_index(native_times,r['timestamp']);future=(native_times>r['timestamp'])&(native_times<=r['timestamp']+min(3.,horizon)+1e-7)
        fig,(ax,count_ax)=plt.subplots(1,2,figsize=(11,5),gridspec_kw={'width_ratios':[3,1]})
        robot=f['robotxy'];ax.set_facecolor('#dddddd');ax.set_xlim(robot[0]-7,robot[0]+7);ax.set_ylim(robot[1]-7,robot[1]+7)
        graph=context.partial_graph;resolution=float(graph['resolution'])
        for state,color in [(1,'#f8fbf8'),(2,'#52575b')]:
            cells=np.array([c['cell'] for c in graph['cells'] if c['state']==state]).reshape(-1,2)
            if len(cells):
                corners=(cells[:,None,:]+np.array([[0,0],[1,0],[1,1],[0,1]])[None])*resolution
                ax.add_collection(PolyCollection(corners,facecolors=color,edgecolors='none',rasterized=True,zorder=1))
        nodes=graph['global_nodes']
        for edge in graph['global_edges']:
            path=np.asarray(edge['path']);ax.plot(path[:,0],path[:,2],color='#77aa77',lw=.6,zorder=2)
        for frame in context.history:
            for tr in positives(frame):
                p=np.asarray(tr['position'])[[0,2]];ax.scatter(*p,c='#1572b6',s=5,alpha=.35,zorder=3)
        ages=f['age'];n=len(f['keys'])
        for person in range(bank.positions.shape[2]):
            color='#e48525' if person<n and ages[person]>1e-7 else '#2c9a9a'
            for world in np.flatnonzero(bank.present[:,person]):
                if isinstance(bank,ConditionedHumanWorlds):
                    event_times=context._association_graph.times
                    ancestors=bank.association.event_person[world]==person
                    currently_visible=bool(np.any(ancestors&(event_times==r['timestamp'])))
                    color='#e48525' if bank.previously_observed[world,person] and not currently_visible else '#2c9a9a'
                p=bank.positions[world,:,person];ax.plot(p[:,0],p[:,1],color=color,alpha=.12,lw=.55,zorder=4)
        for person in range(truth.shape[1]):
            p=np.r_[truth[current_index:current_index+1,person],truth[future,person]]
            ax.plot(p[:,0],p[:,1],'--',color='#c3318a',lw=1.4,zorder=5)
        if n:
            ax.scatter(f['last_positions'][:,0],f['last_positions'][:,1],c=np.where(ages>1e-7,'#e48525','#1572b6'),s=32,edgecolor='white',zorder=6)
        pose=context.history[-1]['robot_pose'];ax.scatter(*robot,marker='^',c='black',s=60,zorder=7)
        ax.set_aspect('equal');ax.set_xlabel('world x (m)');ax.set_ylabel('world z (m)')
        ax.set_title(kind.replace('_',' ')+'; actual partial graph\n'+r['root_id'][:12]+f" anchor {r['anchor_index']}, future ≤{min(3.,horizon):.2f}s")
        legend=[Patch(color='#dddddd',label='UNKNOWN'),Patch(color='#f8fbf8',label='Observed robot-free'),
            Patch(color='#52575b',label='Observed robot Cspace obstruction'),
            Line2D([0],[0],color='#1572b6',marker='o',lw=0,label='Actual positive history'),
            Line2D([0],[0],color='#e48525',label='Predicted previously seen, now-invisible human'),
            Line2D([0],[0],color='#2c9a9a',label='Other predicted joint-world paths'),
            Line2D([0],[0],color='#c3318a',ls='--',label='PRIVILEGED truth: offline only')]
        ax.legend(handles=legend,loc='upper left',fontsize=6,framealpha=.9)
        counts=bank.present.sum(-1);bins=np.arange(max(int(counts.max()),truth.shape[1])+2)-.5
        count_ax.hist(counts,bins=bins,density=True,color='#2c9a9a',alpha=.7)
        count_ax.axvline(truth.shape[1],color='#c3318a',ls='--',label='Offline true count')
        count_ax.set_xlabel('Complete-world person count');count_ax.set_ylabel('32-world empirical probability');count_ax.legend(fontsize=7)
        count_ax.set_title('Population is not only matched tracks')
        fig.text(.5,.01,'Gray is unknown, not free or occupied. Magenta truth is never an online input. Illustrated paths are joint samples, not independent occupancy frames.',ha='center',fontsize=7)
        fig.tight_layout(rect=[0,.025,1,1]);fig.savefig(dest/(kind+'.png'),dpi=180);fig.savefig(dest/(kind+'.pdf'));plt.close(fig)
        reports.append(dict(kind=kind,root_id=r['root_id'],scene_id=r['scene_id'],anchor_index=r['anchor_index'],timestamp=r['timestamp'],
            true_count=truth.shape[1],forecast_counts=counts.tolist(),source_files={p:sha(folder/p) for p in ['legal_root.json.gz','future_observations.json.gz','privileged_future.npz']}))
    dump(dest/'manifest.json',dict(schema='real_development_stage1_figures_1',checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=sha(checkpoint),
        data_manifest_sha256=sha(OUT/'prepared'/split/'manifest.json'),source_sha256=sha(__file__),
        selection='First qualifying original development anchor by fixed legal/status/native-support predicates; no error-based selection.',examples=reports,
        stage2_attempted=False,limitation='No predicted Query observations/graph deltas: Stage2 remains blocked until formal Stage1PASS.'))
    print(str(dest),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--split',default='dev_native_v2');parser.add_argument('--name',default='stage1_examples');args=parser.parse_args();run(args.checkpoint,args.split,args.name)
