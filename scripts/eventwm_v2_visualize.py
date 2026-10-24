#!/usr/bin/env python3
"""Small offline static-map/query overlays from saved v2 root diagnostics."""
import argparse
from pathlib import Path
import numpy as np
import habitat_sim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collector.core import read

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,default=REPO/'output/eventwm_v2_visuals');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    scenes=read(REPO/'evaluation/controlled50/scenes.json');m=read(a.dataset/'manifest.json')
    for root in m['roots']:
        d=a.dataset/root['path'];g=d/'audits/generation_v2.json'
        if not g.exists():continue
        gen=read(g);legal=read(d/'roots/legal_root.json');position=np.array(legal['robot_position'])
        scene=next(s for s in scenes if s['name']==root['scene']);pf=habitat_sim.PathFinder()
        pf.load_nav_mesh(str(REPO/'evaluation/controlled50'/scene['navmesh']))
        xs=np.linspace(position[0]-2.7,position[0]+2.7,150);zs=np.linspace(position[2]-2.7,position[2]+2.7,150)
        mask=np.array([[pf.is_navigable([x,position[1],z],.05) for x in xs] for z in zs])
        fig,ax=plt.subplots(figsize=(7,7));ax.imshow(mask,origin='lower',extent=[xs[0],xs[-1],zs[0],zs[-1]],cmap='gray',vmin=0,vmax=1,alpha=.5)
        rejects=gen['candidate_rejections']
        if rejects:
            xy=np.array([r['target'] for r in rejects]);ax.scatter(xy[:,0],xy[:,2],marker='x',color='red',s=24,label='rejected proposals')
        for i,r in enumerate(gen['accepted_candidates']):
            pts=np.array(r['path']);ax.plot(pts[:,0],pts[:,2],'-o',markersize=3,label=f'accepted path {i}')
        ax.scatter(*position[[0,2]],marker='^',s=70,color='black',label='root')
        yaw=legal['robot_yaw'];ax.arrow(position[0],position[2],.35*np.cos(yaw),-.35*np.sin(yaw),width=.015,color='black')
        ax.add_patch(plt.Circle(position[[0,2]],gen['geometry']['radius']+gen['static_config']['safety_margin'],fill=False,color='black',linestyle='--',label='footprint + margin'))
        ax.set(xlabel='world x (m)',ylabel='world z (m)',title=f"{root['scene']} h{root['human_count']} {root['root_profile']}\n{gen['accepted_movement_queries']}/{gen['requested_movement_queries']} paths; {gen['attempts']} proposals")
        ax.set_aspect('equal');ax.legend(fontsize=8);fig.tight_layout();dest=a.output/f"{root['root_id']}.png";fig.savefig(dest);plt.close(fig);print(dest)

if __name__=='__main__':main()
