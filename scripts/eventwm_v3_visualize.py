#!/usr/bin/env python3
"""Explicit offline debug composites. No plots are produced by default collection."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import habitat_sim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle
from collector.core import read, write, LegalRoot, compile_motion
from collector.social_habitat import capture_social_metric
from eventwm_collect import branches

REPO=Path(__file__).resolve().parents[1]
COLORS=['#27937d','#d79422','#c83e4d']


class ReferencePSC:
    """Habitat pathfinder-only adapter for the exact audited distance operations."""
    def __init__(self,pf,robot,human):self.pathfinder,self.robot,self.human=pf,robot,human
    def get_agent_state(self,i):return SimpleNamespace(position=self.robot if i==0 else self.human)
    def geodesic_distance(self,a,b):
        p=habitat_sim.MultiGoalShortestPath();p.requested_start=np.array(a,dtype=np.float32)
        p.requested_ends=np.array([b],dtype=np.float32);self.pathfinder.find_path(p)
        return p.geodesic_distance


def render_root(dataset,root,output,max_branches):
    path=dataset/root['path'];gen=read(path/'audits/generation_v2.json')
    legal=read(path/'roots/legal_root.json');pos=np.array(legal['robot_position']);yaw=legal['robot_yaw']
    history=read(path/'legal_observations/history.json');tracks=history[-1]['human_tracks']
    scenes=read(REPO/'evaluation/controlled50/scenes.json')
    scene=next(s for s in scenes if s['name']==root['scene']);pf=habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(REPO/'evaluation/controlled50'/scene['navmesh'])):raise ValueError('Navmesh unavailable')
    labels={l['branch_id']:l for l in read(path/'labels/labels.json')}
    raw=[b for b in branches(path) if b['branch_id'].startswith('b')]
    selected=[]
    for cls in (0,1,2):
        available=[b for b in raw if labels[b['branch_id']]['social_outcome_class']==cls]
        if available:selected.append(available[0])
    selected=(selected+[b for b in raw if b not in selected])[:max_branches]
    qs={q['query']['query_id']:q['query'] for q in read(path/'queries/queries.json')}
    xs=np.linspace(pos[0]-2.7,pos[0]+2.7,110);zs=np.linspace(pos[2]-2.7,pos[2]+2.7,110)
    nav=np.array([[pf.is_navigable([x,pos[1],z],.05) for x in xs] for z in zs])
    radius=gen['geometry']['radius'];margin=gen['static_config']['safety_margin']
    required=radius-gen['geometry']['navmesh_radius']+margin+gen['static_config']['max_sample_distance']/2
    clear=np.array([[float(pf.distance_to_closest_obstacle([x,pos[1],z],5)) if nav[j,i] else -1
                    for i,x in enumerate(xs)] for j,z in enumerate(zs)])
    extent=[xs[0],xs[-1],zs[0],zs[-1]]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(20,12),facecolor='#f5f6f8')
    gs=gridspec.GridSpec(2,3,figure=fig,height_ratios=[1.55,1],hspace=.29,wspace=.21)
    axes=[fig.add_subplot(gs[0,i]) for i in range(3)]
    for ax in axes:
        ax.set_facecolor('#e7e9ed');ax.set_aspect('equal');ax.set(xlim=extent[:2],ylim=extent[2:],xlabel='world x (m)',ylabel='world z (m)')
        ax.plot(*pos[[0,2]],'k^',ms=8,zorder=8,label='robot root')
        ax.arrow(pos[0],pos[2],.32*np.cos(yaw),-.32*np.sin(yaw),width=.018,color='#14263d',zorder=8)
    axes[0].imshow(nav,cmap='gray',vmin=0,vmax=1,extent=extent,origin='lower',alpha=.9)
    axes[0].set_title('A  LEGAL / PLANNER VIEW\nNavmesh navigable center region (agent r=0.25m)',loc='left',fontsize=11,fontweight='bold')
    axes[1].imshow(np.where(nav,np.where(clear>=required,2,1),0),cmap=matplotlib.colors.ListedColormap(['#a9afb8','#f4d9b7','#d1ece2']),vmin=0,vmax=2,extent=extent,origin='lower')
    axes[1].set_title('B  STATIC FEASIBILITY\nGreen = clearance-admissible centers (not a full certificate)',loc='left',fontsize=11,fontweight='bold')
    axes[1].add_patch(Circle(pos[[0,2]],radius,fill=False,color='#173953',lw=2,label=f'Spot conservative circle {radius:.3f}m'))
    axes[1].add_patch(Circle(pos[[0,2]],radius+margin,fill=False,color='#173953',lw=1,ls='--',label=f'+ safety margin {margin:.2f}m'))
    refpath=path/'audits/debug_static_reference.json'
    if not refpath.exists():raise ValueError('Native debug reference absent; explicitly collect with --debug-visuals')
    reference=read(refpath);height=np.array(reference['height_relative_to_root'],dtype=float)
    native=axes[2].imshow(height,extent=[reference['xs'][0],reference['xs'][-1],reference['zs'][0],reference['zs'][-1]],origin='lower',cmap='cividis',vmin=-.15,vmax=1.2)
    cb=fig.colorbar(native,ax=axes[2],fraction=.035,pad=.02);cb.set_label('native surface height above root (m)',fontsize=8)
    axes[2].set_title('C  DEBUG GT / PRIVILEGED\nNOT MODEL INPUT • NOT QUERY SELECTION',loc='left',fontsize=11,fontweight='bold',color='#a32e41')
    for i,candidate in enumerate(gen['accepted_candidates']):
        pts=np.array(candidate['path'])
        for ax in axes:
            ax.plot(pts[:,0],pts[:,2],color=f'C{i}',lw=2.2,alpha=.8,label=f'accepted path {i}' if ax==axes[0] else None)
            ax.scatter(pts[-1,0],pts[-1,2],s=42,color=f'C{i}',edgecolors='white',zorder=7)
        axes[0].annotate(str(i),pts[-1,[0,2]],xytext=(5,5),textcoords='offset points',fontsize=10)
    rejects=gen['candidate_rejections']
    for i,r in enumerate(rejects):
        for ax in axes:ax.scatter(r['target'][0],r['target'][2],marker='x',s=40,color='#c55065',alpha=.7,label='rejected endpoint' if i==0 else None)
        pre=r.get('precheck',{});failed=pre.get('failed_pose')
        if failed:
            for ax in axes[1:]:ax.scatter(failed[1],failed[3],marker='P',s=85,c='#77192e',zorder=9,label='recorded first failed pose' if i==0 else None)
            # Recover an actually compatible fixed trajectory for this diagnostic only.
            end=pf.snap_point(r['target']);p=habitat_sim.ShortestPath();p.requested_start=pos;p.requested_end=end
            if pf.find_path(p):
                candidates=[]
                for wait,speed in ((0.,1.),(.5,1.),(1.,1.),(0.,.5)):
                    q=compile_motion(LegalRoot.load(legal),[np.asarray(x).tolist() for x in p.points],r['local_target'],0,wait,speed,next(iter(qs.values()))['generation_config']).data
                    t=np.array(q['nominal_timed_trajectory']);j=pre['failed_segment']
                    if j is not None and j+1<len(t) and t[j,0]-1e-6<=failed[0]<=t[j+1,0]+1e-6:
                        interpolated=t[j]+(t[j+1]-t[j])*(failed[0]-t[j,0])/(t[j+1,0]-t[j,0])
                        if np.allclose(interpolated,failed,atol=1e-5):candidates.append(t)
                if candidates:
                    t=candidates[0];axes[1].plot(t[:,1],t[:,3],ls='--',color='#c55065',lw=1,alpha=.6)
    for track in tracks:
        p=track['position'];visible=track['visible_now']
        axes[0].scatter(p[0],p[2],s=110,marker='o' if visible else 'D',facecolors='#276cb1' if visible else 'none',edgecolors='#276cb1',lw=2,zorder=10)
        axes[0].annotate(f"{'visible' if visible else 'lost'} track {track['track_id']}",(p[0],p[2]),xytext=(6,-13),textcoords='offset points',fontsize=8,color='#174671')
    outside=[t for t in tracks if not (extent[0]<=t['position'][0]<=extent[1] and extent[2]<=t['position'][2]<=extent[3])]
    if outside:
        # Wider legal-only inset keeps distant lost estimates visible without
        # pretending they lie at the edge of the main crop.
        inset=axes[0].inset_axes([.55,.70,.42,.27])
        points=np.array([pos.tolist()]+[t['position'] for t in tracks])
        ix=np.linspace(points[:,0].min()-.4,points[:,0].max()+.4,70)
        iz=np.linspace(points[:,2].min()-.4,points[:,2].max()+.4,45)
        grid=np.array([[pf.is_navigable([x,pos[1],z],.05) for x in ix] for z in iz])
        inset.imshow(grid,origin='lower',extent=[ix[0],ix[-1],iz[0],iz[-1]],cmap='gray',vmin=0,vmax=1)
        inset.plot(pos[0],pos[2],'k^',ms=5)
        for t in tracks:
            p=t['position'];visible=t['visible_now']
            inset.scatter(p[0],p[2],s=45,marker='o' if visible else 'D',facecolors='#276cb1' if visible else 'none',edgecolors='#276cb1',lw=1.5)
            if not visible:inset.annotate('lost',(p[0],p[2]),xytext=(-15,-10),textcoords='offset points',fontsize=7,color='#276cb1')
        inset.set_title('Wider LEGAL track view',fontsize=8);inset.tick_params(labelsize=6)
    # Offline example selection is for display only. Never influences roots/queries.
    example=next((b for b in selected if labels[b['branch_id']]['social_distance_violation']),selected[0])
    lab=labels[example['branch_id']];event=lab['first_social_violation_time']
    if event is None:event=lab['first_human_collision_time'] or 0
    row=min(example['truth'],key=lambda r:abs(r['timestamp']-example['truth'][0]['timestamp']-event))
    for human in row['humans']:
        hp=np.array(human['position']);axes[2].scatter(hp[0],hp[2],marker='*',s=140,c='#a52876',zorder=10,label=f"GT human: {example['branch_id']} at {event:.2f}s")
        # PSC is geodesic, with the canonical no-path fallback. Do not substitute a circle.
        sx=np.linspace(pos[0]-2.7,pos[0]+2.7,65);sz=np.linspace(pos[2]-2.7,pos[2]+2.7,65)
        field=np.array([[capture_social_metric(ReferencePSC(pf,np.array([x,pos[1],z]),hp),1)['distances'][0]['distance'] for x in sx] for z in sz])
        if np.nanmin(field)<1<np.nanmax(field):axes[2].contour(sx,sz,field,levels=[1.],colors=['#d226a0'],linewidths=2)
    contact_example=next((b for b in selected if labels[b['branch_id']]['direct_human_collision']),example)
    contacts=[r for r in contact_example['truth'] if r['native_human_contact']]
    for r in contacts[:1]:
        ids={h['object_id'] for h in r['humans']}
        for cp in r['contacts']:
            if {cp['object_id_a'],cp['object_id_b']} & ids and r['robot']['object_id'] in (cp['object_id_a'],cp['object_id_b']):
                p=cp['position_on_a_in_ws'];axes[2].scatter(p[0],p[2],marker='X',s=110,c='red',zorder=11,label=f"native contact: {contact_example['branch_id']} at {labels[contact_example['branch_id']]['first_human_collision_time']:.2f}s")
    for ax in axes:
        handles,names=ax.get_legend_handles_labels();unique=dict(zip(names,handles))
        ax.legend(unique.values(),unique.keys(),loc='lower left',fontsize=7,framealpha=.9)
    timeline=fig.add_subplot(gs[1,:2]);timeline.set_facecolor('white')
    for i,b in enumerate(selected):
        l=labels[b['branch_id']];ts=np.array([r['timestamp'] for r in b['truth']]);ds=[min((d['distance'] for d in r['social_metric']['distances']),default=np.nan) for r in b['truth']]
        if b['truth'][0]['humans']:
            timeline.plot(ts-ts[0],ds,color=f'C{i}',lw=2,label=f"{b['branch_id']} {l['social_outcome_name']}")
        if l['first_human_collision_time'] is not None:
            timeline.axvline(l['first_human_collision_time'],color=f'C{i}',ls=':',alpha=.7)
    timeline.axhline(1.,color='#c12592',ls='--',label='PSC boundary: distance < 1.0m')
    timeline.set(xlabel='native time after root (s)',ylabel='PSC geodesic / no-path fallback distance (m)',title='D  OUTCOME EVIDENCE • offline examples; collision takes precedence')
    if not row['humans']:
        horizon=max(labels[b['branch_id']]['observed_duration'] for b in selected)
        timeline.set(xlim=(0,horizon),ylim=(0,2))
        timeline.axvspan(0,horizon,color=COLORS[0],alpha=.08)
        timeline.text(.5,.6,'Zero active humans — all full queries SAFE_SUCCESS',ha='center',transform=timeline.transAxes)
    timeline.grid(alpha=.15);timeline.legend(fontsize=8,loc='best')
    notes=fig.add_subplot(gs[1,2]);notes.axis('off')
    counts={name:sum(labels[b['branch_id']]['social_outcome_name']==name for b in raw) for name in ('SAFE_SUCCESS','SOCIAL_VIOLATION_SUCCESS','HUMAN_COLLISION')}
    lines=[f"LEGAL ROOT: {sum(t['visible_now'] for t in tracks)} visible / {sum(not t['visible_now'] for t in tracks)} lost",f"PRIMARY OUTCOMES  {counts['SAFE_SUCCESS']} safe / {counts['SOCIAL_VIOLATION_SUCCESS']} social / {counts['HUMAN_COLLISION']} contact",'',f"{gen['accepted_movement_queries']} paths + WAIT | {gen['attempts']} proposals | {len(rejects)} rejected",f'Navmesh edge clearance required: {required:.3f} m']
    if rejects:
        r=next((r for r in rejects if r.get('precheck')),rejects[0]);p=r.get('precheck',{})
        minimum=p.get('min_static_clearance');lines += [f"Reject: {r['rejection_reason']}",f"First failed segment: {p.get('failed_segment')} | min: {minimum:.3f} m" if minimum is not None else 'No swept certificate: unavailable path / near duplicate']
    lines+=['','EXAMPLE BRANCHES (display only)']
    def fmt(value):return '—' if value is None else f'{value:.3f}'
    for b in selected[:3]:
        l=labels[b['branch_id']];q=qs[b['query_id']]
        lines+=[f"{b['branch_id']}: {l['social_outcome_name']}",f"  wait={q['wait_duration']:.1f}s, speed={q['speed_scale']}; social t={fmt(l['first_social_violation_time'])}",f"  disc clearance={fmt(l['min_human_clearance'])}m; collision t={fmt(l['first_human_collision_time'])}"]
    lines+=['','PSC magenta contour: exact geodesic/fallback 1m.', 'Physical collision: native articulated geometry/contact.', 'No invented circular physical collision boundary.', 'Disc clearance is a 0.25m + 0.30m research proxy.', 'GT panel: native static rays from root +1.2m downward.', 'Unknown ray hits remain blank; this is not occupancy.']
    notes.text(0,1,'\n'.join(lines),va='top',fontsize=8.6,linespacing=1.45)
    fig.suptitle(f"EVENTWM SOCIAL OUTCOME V3  |  {root['scene']}  •  humans={root['human_count']}  •  {root['root_profile']}\n{root['root_id']}   |   checker eventwm-static-swept-v2.2",x=.045,ha='left',fontsize=17,fontweight='bold')
    fig.subplots_adjust(top=.90,bottom=.06,left=.05,right=.96)
    dest=output/f"{root['root_id']}.png";fig.savefig(dest,dpi=180,facecolor=fig.get_facecolor());plt.close(fig)
    return dict(root_id=root['root_id'],path=str(dest),bytes=dest.stat().st_size,examples=[b['branch_id'] for b in selected])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-roots',type=int,default=4);p.add_argument('--max-branches',type=int,default=6);a=p.parse_args()
    if not 1<=a.max_roots<=4 or not 1<=a.max_branches<=6:raise ValueError('Debug limits: <=4 roots, <=6 branches')
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=read(a.dataset/'manifest.json');results=[]
    for root in [r for r in manifest['roots'] if r['state']=='completed'][:a.max_roots]:
        results.append(render_root(a.dataset,root,a.output,a.max_branches));print(results[-1],flush=True)
    write(a.output/'visual_manifest.json',dict(roots=results,png_bytes=sum(r['bytes'] for r in results),
          privileged=True,selection='offline only; no feedback to collector',video=False))

if __name__=='__main__':main()
