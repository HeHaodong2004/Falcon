"""Offline figures from retained actual traces; does not supply policy inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter


def load(folder):
    return (json.loads((folder/'actual.json').read_text()),
            json.loads((folder/'report.json').read_text()),
            json.loads((folder/'reference.json').read_text()))


def draw(ax,folder,grid=None,prefix=None):
    trace,report,ref=load(folder)
    poses=np.asarray(trace['poses']);route=np.asarray(ref['nominal_waypoints'])
    if prefix is not None:poses=poses[:prefix+1]
    if grid is not None:
        x,z=grid.origin;nx,nz=grid.free.shape;res=grid.resolution
        ax.imshow(grid.free.T,origin='lower',extent=[x,x+nx*res,z,z+nz*res],cmap='Greys_r',vmin=-1,vmax=1,alpha=.65)
    ax.plot(route[:,0],route[:,2],'--',color='#777777',lw=2,label='Immutable reference')
    ax.plot(poses[:,0],poses[:,2],color='#1269b0',lw=2,label='Actual robot')
    ax.scatter(*route[-1,[0,2]],marker='*',s=100,color='#222222',label='Goal')
    seen=[]
    observations=trace['observations'][:prefix+1] if prefix is not None else trace['observations']
    for ob in observations:
        for h in ob['tracks']:
            if h['visible_now']:seen.append((h['position'][0],h['position'][2],ob['timestamp']))
    if seen:
        h=np.asarray(seen);ax.scatter(h[:,0],h[:,1],c=h[:,2],cmap='Greens',vmin=-.2,s=9,alpha=.7,label='Actual visible humans')
    for i in np.linspace(0,len(poses)-1,min(5,len(poses))).astype(int):
        ax.annotate(f"{trace['timestamps'][i]-trace['timestamps'][0]:.2f}s",poses[i,[0,2]],fontsize=7,xytext=(3,5),textcoords='offset points')
    for kind,color,marker in [('psc_violation','#cf8200','o'),('static_obstacle_contact','#d12a25','x'),('human_mesh_contact','#e00000','x')]:
        ids=[i+1 for i,l in enumerate(trace['offline_labels']) if l[kind] and i+1<len(poses)]
        if ids:ax.scatter(poses[ids,0],poses[ids,2],s=38,edgecolors=color if marker=='o' else None,
                          facecolors='none' if marker=='o' else color,marker=marker,label=kind.replace('_',' '))
    forecasts=trace.get('forecasts',[])
    if forecasts:
        f=next((f for f in forecasts if np.asarray(f['positions']).shape[2]>0),None)
        if f:
            points=np.asarray(f['positions'])[0] # one joint world, not a guaranteed path
            for i in range(points.shape[1]):ax.plot(points[:,i,0],points[:,i,1],':',color='#d67d14',alpha=.75,label='L2 sampled world (first bank)' if i==0 else None)
    bound=np.r_[poses[:,[0,2]],route[:,[0,2]]];lo=bound.min(0)-1.6;hi=bound.max(0)+1.6
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='World X (m)',ylabel='World Z (m)',aspect='equal')
    ax.set_title(f"{report['case_id'].replace('scenario_','')} / {report['variant']}\n{report['status']}; native SR={report['metrics']['success']:.0f}; PSC={report['metrics']['psc']:.3f}",fontsize=10)
    ax.grid(alpha=.12)
    return trace,report,ref


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenarios',type=Path,required=True)
    p.add_argument('--analytic',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    import habitat_sim
    from eventwm_graph.benchmark.map import load_certified_map
    from eventwm_graph.l4.geometry import StaticGrid
    maps={};grids={}
    def grid(folder):
        t,r,ref=load(folder);manifest=r['case']['manifest']
        if manifest not in maps:maps[manifest]=load_certified_map(manifest)
        key=r['case_id']
        if key not in grids:grids[key]=StaticGrid.from_navigation(maps[manifest].navigation,ref['nominal_waypoints'])
        return grids[key]
    names=['head_on','crossing','blocked','multiple','static_detour','occlusion_reappearance']
    fig,axs=plt.subplots(2,3,figsize=(15,9))
    for ax,name in zip(axs.ravel(),names):
        folder=a.scenarios/('scenario_'+name+'_B1');draw(ax,folder,grid(folder));ax.legend(fontsize=6,loc='best')
    fig.suptitle('Native execution: actual observations only; shaded area is supplied static configuration-space support',fontsize=12)
    fig.tight_layout();fig.savefig(a.output/'native_scenarios.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,v in zip(axs,['B1','B2']):
        folder=a.scenarios/('scenario_multiple_'+v);draw(ax,folder,grid(folder));ax.legend(fontsize=7)
    fig.suptitle('Matched multiple-human case: observation-only versus frozen L2 augmentation')
    fig.tight_layout();fig.savefig(a.output/'l2_comparison.png',dpi=160);plt.close(fig)
    # Analytic reobservation is labeled explicitly: the native case's name
    # alone cannot establish that a .5–2 s reobservation actually occurred.
    r=json.loads((a.analytic/'reobservation.json').read_text());poses=np.asarray(r['feedback']['actual_poses']);times=np.asarray(r['feedback']['actual_timestamps'])
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4));ax.plot(poses[:,0],poses[:,2],label='Actual analytic robot')
    ref=np.asarray(r['reference']['nominal_waypoints']);ax.plot(ref[:,0],ref[:,2],'--',label='Reference')
    ts=[];seen=[];last=[]
    for record in r['records']:
        ob=record['observation'];ts.append(ob['timestamp']);seen.append(any(t['visible_now'] for t in ob['tracks']))
        for t in ob['tracks']:
            if t['visible_now']:ax.scatter(t['position'][0],t['position'][2],c='green',s=8)
        last.append(max((t['observation_age'] for t in ob['tracks']),default=0.))
    ax.set(aspect='equal',xlabel='World X (m)',ylabel='World Z (m)',title='Analytic fixture, native ego increments');ax.legend()
    bx.step(ts,seen,where='post',label='Visible now');bx.plot(ts,last,label='Age of retained observation (s)');bx.axvspan(.25,1.1,color='gray',alpha=.15,label='Configured occlusion');bx.set(xlabel='Actual fixture time (s)',title=r['status']);bx.legend()
    fig.tight_layout();fig.savefig(a.output/'reobservation_analytic.png',dpi=160);plt.close(fig)
    # Small, deterministic top-down animation; a displayed human is always a
    # measurement, never the environment's unobserved true position.
    folders=[a.scenarios/('scenario_head_on_'+v) for v in ('B1','B2')]
    traces=[load(f)[0] for f in folders];fig,axs=plt.subplots(1,2,figsize=(10,4))
    indices=np.linspace(0,max(len(t['poses']) for t in traces)-1,35).astype(int)
    def frame(k):
        for ax,folder,t in zip(axs,folders,traces):
            ax.clear();draw(ax,folder,grid(folder),min(int(k),len(t['poses'])-1))
        fig.suptitle('Native head-on avoidance; displayed humans are accumulated legal measurements');fig.tight_layout()
    animation=FuncAnimation(fig,frame,frames=indices,interval=130)
    animation.save(a.output/'head_on_comparison.gif',writer=PillowWriter(fps=8));plt.close(fig)
    (a.output/'sources.json').write_text(json.dumps(dict(scenarios=str(a.scenarios.resolve()),analytic=str(a.analytic.resolve()),
        human_display='legal visible positions only; colored by actual timestamp; no truth/oracle overlays',
        static_display='certified static disk support, not raw wall occupancy',animation='top-down trace animation, not rendered camera video'),indent=2))


if __name__=='__main__':main()
