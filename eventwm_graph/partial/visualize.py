"""Offline, reproducible scientific figures/timeline from compact legal traces."""
import html
import json
import math
from pathlib import Path
import numpy as np
from ..stage1.storage import read_gzip
from .map import PartialGraph
from .query import HUMAN_ENVELOPE,ROBOT_BOUNDS

COLORS={'human_safe':'#008849','human_intrusion':'#dc2c91','frontier':'#dfad00',
        'goal':'#d96312','topology':'#367bd8','current_position':'#111111'}


def candidate_color(provenance):
    key=provenance['source']+('_'+provenance.get('intended_band','') if provenance['source']=='human' else '')
    return COLORS.get(key,'#666666')


def draw_frame(ax,snapshot,goal=None,selected=0,actual=None,labels=None):
    from matplotlib.patches import Circle,Polygon,Rectangle,Wedge
    from matplotlib.collections import PatchCollection,LineCollection
    graph=snapshot['graph'];row=snapshot['observation'];r=graph['resolution'];pose=row['robot_pose']
    cells=graph['cells'];coords=np.array([[(c['cell'][0]+.5)*r,-(c['cell'][1]+.5)*r] for c in cells])
    ax.set_facecolor('#202a37')
    if len(coords):
        ax.set_xlim(coords[:,0].min()-.7,coords[:,0].max()+.7);ax.set_ylim(coords[:,1].min()-.7,coords[:,1].max()+.7)
    for state,color in ((1,'#dce6df'),(2,'#8d593d')):
        patches=[Rectangle((c['cell'][0]*r,-(c['cell'][1]+1)*r),r,r) for c in cells if c['state']==state]
        ax.add_collection(PatchCollection(patches,facecolor=color,edgecolor='none',rasterized=True,zorder=1))
    # Fine graph vertices and certified local-neighbor edges (bounded display).
    g=PartialGraph.restore(graph);segments=[]
    for k,neighbors in g.adjacency.items():
        for v,_ in neighbors:
            if k<v:
                a,b=g.center(k),g.center(v);segments.append([(a[0],-a[2]),(b[0],-b[2])])
    ax.add_collection(LineCollection(segments,colors='#75a0ac',linewidths=.25,alpha=.30,zorder=2))
    for edge in graph.get('global_edges',[]):
        p=np.asarray(edge['path']);ax.plot(p[:,0],-p[:,2],color='#00a8bd',lw=1.4,alpha=.65,zorder=3)
    for node in graph.get('global_nodes',{}).values():
        p=node['position'];ax.scatter(p[0],-p[2],s=23,c='#1261a0',edgecolors='white',linewidths=.5,zorder=4)
    origin=np.asarray(row['camera_origin']);rotation=np.asarray(row['camera_rotation']);forward=-rotation[:,2]
    angle=math.degrees(math.atan2(-forward[2],forward[0]))
    ax.add_patch(Wedge((origin[0],-origin[2]),10.,angle-45,angle+45,facecolor='#42b1e7',alpha=.11,edgecolor='#299dd0',ls='--',zorder=2))
    for human in row['tracks']:
        p=human['position'];color='#df3b38' if human['visible_now'] else '#939ba3'
        ax.add_patch(Circle((p[0],-p[2]),1.,fill=False,color='#c32e70',ls='--',lw=1.3,zorder=5))
        ax.add_patch(Circle((p[0],-p[2]),HUMAN_ENVELOPE,fill=False,color='#244e83',ls=':',lw=1.4,zorder=5))
        ax.scatter(p[0],-p[2],s=55,c=color,edgecolors='white',zorder=7)
        ax.text(p[0],-p[2]+.12,human['track_id'].replace('track_','h')+(' visible' if human['visible_now'] else f" tracked {human['observation_age']:.2f}s"),fontsize=6,color=color,zorder=8)
    for candidate in snapshot['diagnostics'].get('spatial_candidates',[]):
        p=candidate['position'];provenance=candidate['provenance'];color=candidate_color(provenance)
        marker={'human':'D','frontier':'s','goal':'*','topology':'o','current_position':'P'}[provenance['source']]
        ax.scatter(p[0],-p[2],s=28,c=color,marker=marker,edgecolors='white',linewidths=.4,zorder=7)
    for rejected in snapshot['diagnostics'].get('rejected',[]):
        p=rejected['position'];ax.scatter(p[0],-p[2],s=17,c='#ed6652',marker='x',alpha=.6,zorder=6)
    queries=snapshot.get('queries',[])
    for q in queries:
        p=q['waypoints'][-1];yaw=q['final_yaw'];color=candidate_color(q['provenance'][-1])
        ax.arrow(p[0],-p[2],.28*math.cos(yaw),.28*math.sin(yaw),color=color,width=.009,head_width=.07,length_includes_head=True,zorder=8)
    if queries:
        q=queries[selected%len(queries)];p=np.asarray(q['compiled']['route'])
        ax.plot(p[:,0],-p[:,2],color='#16a05b',lw=3.,zorder=8)
    # Robot all-gait bounding rectangle at actual heading, distinct from .25m
    # native planning radius and .616m human body envelope.
    xmin,xmax,zmin,zmax=ROBOT_BOUNDS;c,s=math.cos(pose[3]),math.sin(pose[3])
    corners=[]
    for x,z in ((xmin,zmin),(xmin,zmax),(xmax,zmax),(xmax,zmin)):
        corners.append((pose[0]+c*x+s*z,-pose[2]+s*x-c*z))
    ax.add_patch(Polygon(corners,fill=False,edgecolor='black',lw=1.1,zorder=10))
    ax.scatter(pose[0],-pose[2],s=65,c='#10191f',edgecolors='white',zorder=11)
    ax.arrow(pose[0],-pose[2],.55*c,.55*s,color='#00e5b4',width=.025,head_width=.15,length_includes_head=True,zorder=12)
    if actual:
        a=np.asarray(actual);ax.plot(a[:,0],-a[:,2],c='#202020',lw=2.,zorder=10)
        if labels:
            for i,l in enumerate(labels):
                if l['psc_violation']:ax.plot(a[i:i+2,0],-a[i:i+2,2],c='#f32161',lw=3.5,zorder=11)
                if l.get('human_mesh_contact') or l.get('static_obstacle_contact'):
                    ax.scatter(a[i+1,0],-a[i+1,2],marker='X',s=100,c='#e00000',edgecolors='white',zorder=15)
    ax.set_aspect('equal');ax.set_xlabel('Habitat x (m)');ax.set_ylabel('−Habitat z (m)')
    ax.set_title(f"t={row['timestamp']:.3f}s · revision {graph['revision']} · {len(queries)} accepted queries",fontsize=9)


def stage0(trace_path,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    trace=read_gzip(trace_path);snapshots=trace['snapshots']
    choice=np.unique(np.linspace(0,len(snapshots)-1,min(6,len(snapshots))).astype(int))
    fig,axes=plt.subplots(2,3,figsize=(17,10),squeeze=False)
    for ax,index in zip(axes.ravel(),choice):draw_frame(ax,snapshots[index])
    for ax in axes.ravel()[len(choice):]:ax.axis('off')
    fig.suptitle(trace['scene_id']+' · causal partial-map / graph-query timeline',fontsize=14)
    fig.text(.02,.015,'Dark: UNKNOWN · pale: KNOWN_FREE · brown: KNOWN_OCCUPIED (inflated) · cyan: global paths · pale blue: local graph\n'
        'Red/gray humans: visible/tracked · dashed pink: 1.0m PSC reference · dotted blue: conservative .616m human collision-body envelope\n'
        'Green diamonds: safe ring · magenta diamonds: intrusion ring · yellow squares: frontier · orange stars: goal · blue dots: topology · red ×: rejected',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.96));image=output/(Path(trace_path).name.replace('.json.gz','')+'.png')
    fig.savefig(image,dpi=150);plt.close(fig)
    frames=[]
    for snapshot in snapshots:
        graph=snapshot['graph'];row=snapshot['observation']
        frames.append({'t':row['timestamp'],'pose':row['robot_pose'],'camera_origin':row['camera_origin'],
            'camera_rotation':row['camera_rotation'],'tracks':row['tracks'],'resolution':graph['resolution'],
            'cells':[[*c['cell'],c['state']] for c in graph['cells']],
            'global_nodes':list(graph['global_nodes'].values()),'global_edges':graph['global_edges'],
            'queries':[{'waypoints':q['waypoints'],'final_yaw':q['final_yaw'],'provenance':q['provenance'],
                        'route':q['compiled']['route'],'actions':len(q['compiled']['actions']),
                        'nominal_seconds':q['compiled']['nominal_seconds'],'terminal_turn_actions':q['compiled']['terminal_turn_actions']} for q in snapshot['queries']],
            'diagnostics':snapshot['diagnostics']})
    data=json.dumps(frames,separators=(',',':')).replace('</','<\\/')
    page=HTML_TEMPLATE.replace('__DATA__',data).replace('__TITLE__',html.escape(trace['scene_id']))
    timeline=output/(Path(trace_path).name.replace('.json.gz','')+'.html');timeline.write_text(page)
    explanations=[]
    for snapshot in snapshots:
        for q in snapshot['queries'][:3]:
            p=q['provenance'][0]
            explanations.append({'time':snapshot['observation']['timestamp'],'query_id':q['query_id'],
                'generated_by':p,'accepted_because':'known-free reachable waypoint; known-space A*; body-clear discrete schedule; legal tracked human envelopes clear at proposal',
                'Astar_route':q['compiled']['route'],'final_yaw':q['final_yaw'],
                'terminal_turn_actions':q['compiled']['terminal_turn_actions'],
                'unknown':'All cells absent from this timestamped partial graph; no full scene geometry is displayed.'})
    from ..config import write_json
    write_json(output/(Path(trace_path).stem+'.explanations.json'),explanations)
    return {'image':str(image.resolve()),'timeline':str(timeline.resolve())}


def stage1(root_folder,output,limit=4):
    """Offline before/after figures from real sealed native query outcomes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from ..config import write_json
    folder=Path(root_folder);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    root=read_gzip(folder/'legal_root.json.gz');queries=read_gzip(folder/'queries.json.gz')
    outcomes=read_gzip(folder/'outcomes.json.gz');nodes=read_gzip(folder/'future_observations.json.gz')
    labels=read_gzip(folder/'privileged_step_labels.json.gz');manifest=json.loads((folder/'manifest.json').read_text())
    # Prefer a same-human, same-direction safe/intrusion comparison. Outcome
    # choice is explicitly offline visualization, never collection admission.
    pairs=[]
    for i,q in enumerate(queries):
        p=q['provenance'][0]
        if p.get('intended_band')!='safe':continue
        for j,r in enumerate(queries):
            s=r['provenance'][0]
            if s.get('intended_band')=='intrusion' and s.get('track_id')==p.get('track_id') and s.get('direction')==p.get('direction'):
                a,b=outcomes[i]['summary'],outcomes[j]['summary']
                score=abs((a['psc'] or 0)-(b['psc'] or 0))+int(a['human_collision']!=b['human_collision'])
                aligned=abs(math.atan2(math.sin(q['final_yaw']-r['final_yaw']),math.cos(q['final_yaw']-r['final_yaw'])))<.1
                pairs.append((aligned,score,i,j))
    selected=[]
    if pairs:
        _,_,i,j=max(pairs);selected=[i,j]
    for i in sorted(range(len(queries)),key=lambda i:(queries[i]['provenance'][0]['source']!='current_position',
            not outcomes[i]['summary']['human_collision'],i)):
        if i not in selected:selected.append(i)
        if len(selected)>=limit:break
    selected=selected[:limit];rows=[]
    with np.load(folder/'privileged_future.npz',allow_pickle=False) as future:
        human_positions=future['positions'].copy()
    fig,axes=plt.subplots(len(selected),2,figsize=(15,5*len(selected)),squeeze=False)
    chart,charts=plt.subplots(len(selected),1,figsize=(12,2.7*len(selected)),squeeze=False)
    for row_index,index in enumerate(selected):
        q=queries[index];o=outcomes[index];ids=o['node_ids'];summary=o['summary']
        graph=PartialGraph.restore(root['partial_graph']);initial=root['history'][-1]
        diagnostics=manifest['sampling']['query_diagnostics']
        before={'graph':graph.snapshot(),'observation':initial,'queries':[q],'diagnostics':diagnostics}
        path=[initial['robot_pose']];ls=[]
        for nid in ids:
            frame=nodes[nid];graph.update(frame['observation']);path.append(frame['observation']['robot_pose']);ls.append(labels[nid])
        after_row=nodes[ids[-1]]['observation'] if ids else initial
        graph.rebuild_global(after_row['robot_pose'][:3])
        after={'graph':graph.snapshot(),'observation':after_row,'queries':[q],
               'diagnostics':{'spatial_candidates':[],'rejected':[]}}
        draw_frame(axes[row_index,0],before);draw_frame(axes[row_index,1],after,actual=path,labels=ls)
        for human in range(human_positions.shape[1]):
            hp=human_positions[:len(ids)+1,human]
            axes[row_index,1].plot(hp[:,0],-hp[:,2],color='#b62f24',lw=1.1,alpha=.75)
            axes[row_index,1].add_patch(Circle((hp[-1,0],-hp[-1,2]),1.,fill=False,color='#cb2862',ls='--',alpha=.8))
        p=q['provenance'][0];intention=p['source']+' '+p.get('direction','')+' '+p.get('intended_band','')
        distance=summary['minimum_official_distance_m'];distance='none' if distance is None else f'{distance:.3f}m'
        axes[row_index,0].set_title('LEGAL INITIAL · '+intention+' · terminal yaw '+f"{q['final_yaw']:.2f}rad")
        axes[row_index,1].set_title(f"ACTUAL OUTCOME · min {distance} · PSC {summary['psc']}\n"
            f"collision={summary['human_collision']} · {summary['elapsed_s']:.3f}s · complete={summary['query_complete']} · {summary['termination']}")
        ax=charts[row_index,0]
        times=np.asarray([l['timestamp']-initial['timestamp'] for l in ls]);dist=[l['official_dmin_m'] for l in ls]
        if times.size and all(d is not None for d in dist):
            ax.plot(times,dist,color='#174b67',label='official-compatible minimum center distance')
            ax.fill_between(times,dist,1.,where=np.asarray(dist)<1.,color='#f32161',alpha=.35,label='PSC violation')
        ax.axhline(1.,color='#c32e70',ls='--',label='1m PSC threshold');ax.set_ylabel('distance (m)');ax.set_xlabel('elapsed native time (s)')
        ax.set_title(intention+' · '+summary['termination']);ax.legend(fontsize=7,loc='best')
        rows.append({'query_index':index,'query_id':q['query_id'],'proposal':q['provenance'],
                     'final_yaw':q['final_yaw'],'measured_outcome':summary})
    fig.suptitle('Same legal root, independent native query branches · '+manifest['scene_id'])
    fig.text(.02,.008,'Dark unknown; pale known free; brown occupied; cyan certified static paths. '
        'Black actual robot route; pink segments PSC violations. Red human trajectories are PRIVILEGED OUTCOME OVERLAYS ONLY. '
        'Dashed circles:1m PSC; dotted circles:.616m conservative body envelope.',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,.98));chart.tight_layout()
    stem=manifest['root_id'][:16];picture=output/(stem+'_before_after.png');proximity=output/(stem+'_distance.png')
    fig.savefig(picture,dpi=130);chart.savefig(proximity,dpi=140);plt.close(fig);plt.close(chart)
    write_json(output/(stem+'.json'),{'root':str(folder.resolve()),'examples':rows,
        'selection':'offline same-human/direction and matched-terminal-heading pair when available, then measured contrast; orientation examples'})
    page=output/(stem+'.html')
    page.write_text('<!doctype html><meta charset="utf-8"><title>Measured query outcomes</title>'
        '<style>body{font:16px system-ui;max-width:1600px;margin:auto}img{width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>Actual Habitat counterfactual outcomes</h1><p>All queries share one legally observed root. '
        'Human trajectories in outcome panels are privileged supervision; they were unavailable to query generation.</p>'
        f'<img src="{picture.name}"><img src="{proximity.name}"><pre>'+html.escape(json.dumps(rows,indent=2))+'</pre>')
    return {'image':str(picture.resolve()),'proximity':str(proximity.resolve()),'html':str(page.resolve()),'examples':rows}


HTML_TEMPLATE='''<!doctype html><meta charset="utf-8"><title>__TITLE__ — partial graph queries</title>
<style>body{font:15px system-ui;margin:24px;background:#f4f6f8;color:#203047}h1{font-size:22px}main{display:flex;gap:20px}canvas{background:#202a37;border-radius:8px;width:68vw;height:72vh}aside{width:28vw}pre{white-space:pre-wrap;font:12px monospace;max-height:40vh;overflow:auto}select,input{width:95%;margin:8px 0}small{display:block;line-height:1.6}button{padding:7px 20px}</style>
<h1>__TITLE__ · observation-limited graph query timeline</h1>
<main><canvas id="map" width="1200" height="900"></canvas><aside><button id="play">Play / pause</button>
<input type="range" id="time" min="0" value="0"><b id="stamp"></b><select id="query"></select><pre id="detail"></pre>
<small>Dark = UNKNOWN; pale = KNOWN_FREE; brown = KNOWN_OCCUPIED in robot-center configuration space.<br>
Cyan paths = global graph. Blue dots = local graph. Blue fan = current camera horizontal projection (vertical/range tests are applied by the adapter).<br>
Red people = visible; gray = remembered tracks. Pink dashed circle = 1.0m PSC reference; blue dotted circle = .616m conservative human body envelope (a bound, not the official collision rule).<br>
Green = safe human ring; magenta = intrusion ring; yellow = frontier; orange = goal; blue = topology; red × = rejected. Arrows = final headings. Thick green = selected A* route.<br>
Select a query to inspect its source observation, acceptance and terminal turn. Rejected candidates/reasons are listed below. Nothing outside the current accumulated map is shown.</small><pre id="rejected"></pre></aside></main>
<script>const frames=__DATA__;const canvas=document.querySelector('#map'),ctx=canvas.getContext('2d'),slider=document.querySelector('#time'),select=document.querySelector('#query');slider.max=frames.length-1;
const colors={human_safe:'#008849',human_intrusion:'#dc2c91',frontier:'#dfad00',goal:'#d96312',topology:'#367bd8',current_position:'#111111'};let playing=false;
function draw(){const f=frames[+slider.value],r=f.resolution;let xs=f.cells.map(c=>(c[0]+.5)*r),zs=f.cells.map(c=>(c[1]+.5)*r);xs.push(f.pose[0]);zs.push(f.pose[2]);let xmin=Math.min(...xs)-.8,xmax=Math.max(...xs)+.8,zmin=Math.min(...zs)-.8,zmax=Math.max(...zs)+.8;const scale=Math.min(1160/(xmax-xmin),860/(zmax-zmin));const X=x=>25+(x-xmin)*scale,Y=z=>875-(z-zmin)*scale;
ctx.fillStyle='#202a37';ctx.fillRect(0,0,1200,900);for(const c of f.cells){ctx.fillStyle=c[2]===1?'#dce6df':'#8d593d';ctx.fillRect(X(c[0]*r),Y((c[1]+1)*r),r*scale+.2,r*scale+.2);if(c[2]===1){ctx.fillStyle='#8aaeb9';ctx.fillRect(X((c[0]+.5)*r),Y((c[1]+.5)*r),1.3,1.3);}}
function path(points,color,width=2){if(!points.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(X(p[0]),Y(p[2])):ctx.moveTo(X(p[0]),Y(p[2])));ctx.stroke();}function circle(p,r,color,dash=[]){ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.arc(X(p[0]),Y(p[2]),r*scale,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);}function dot(p,color,size=5){ctx.fillStyle=color;ctx.beginPath();ctx.arc(X(p[0]),Y(p[2]),size,0,Math.PI*2);ctx.fill();}function arrow(p,a,color,length=.35){path([p,[p[0]+length*Math.cos(a),p[1],p[2]-length*Math.sin(a)]],color,2);}
    let forward=f.camera_rotation.map(row=>-row[2]),a=Math.atan2(-forward[2],forward[0]),o=f.camera_origin;ctx.fillStyle='#38aee822';ctx.beginPath();ctx.moveTo(X(o[0]),Y(o[2]));for(let t=a-Math.PI/4;t<=a+Math.PI/4+.01;t+=.025)ctx.lineTo(X(o[0]+10*Math.cos(t)),Y(o[2]-10*Math.sin(t)));ctx.closePath();ctx.fill();
for(const e of f.global_edges)path(e.path,'#00a8bd',2);for(const n of f.global_nodes)dot(n.position,'#1261a0',4);for(const h of f.tracks){circle(h.position,1,'#c32e70',[7,5]);circle(h.position,.616,'#244e83',[2,4]);dot(h.position,h.visible_now?'#df3b38':'#939ba3',7);}
for(const c of f.diagnostics.spatial_candidates){let p=c.provenance,key=p.source+(p.source==='human'?'_'+p.intended_band:'');dot(c.position,colors[key]||'#888',5);}for(const q of f.queries){let p=q.provenance[0],key=p.source+(p.source==='human'?'_'+p.intended_band:'');arrow(q.waypoints.at(-1),q.final_yaw,colors[key]||'#888');}for(const v of f.diagnostics.rejected){ctx.strokeStyle='#ed6652';let x=X(v.position[0]),y=Y(v.position[2]);ctx.beginPath();ctx.moveTo(x-3,y-3);ctx.lineTo(x+3,y+3);ctx.moveTo(x-3,y+3);ctx.lineTo(x+3,y-3);ctx.stroke();}
let q=f.queries[+select.value||0];if(q)path(q.route,'#16a05b',5);dot(f.pose,'#10191f',8);arrow(f.pose,f.pose[3],'#00e5b4',.6);document.querySelector('#stamp').textContent='t = '+f.t.toFixed(3)+'s · '+f.queries.length+' accepted';document.querySelector('#detail').textContent=q?JSON.stringify(q,null,2):'No query is certified at this time. The ego anchor may still be isolated or body support unknown.';document.querySelector('#rejected').textContent=JSON.stringify(f.diagnostics.rejected.map(v=>({source:v.provenance,reason:v.reason})),null,2);}
function frame(){let f=frames[+slider.value];select.innerHTML='';f.queries.forEach((q,i)=>{let o=document.createElement('option');o.value=i;o.textContent=i+' · '+q.provenance[0].source+' '+(q.provenance[0].intended_band||'')+' · final '+q.final_yaw.toFixed(2)+' rad';select.append(o)});draw();}slider.oninput=frame;select.onchange=draw;document.querySelector('#play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){slider.value=(+slider.value+1)%frames.length;frame()}},900);frame();</script>'''
