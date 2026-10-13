#!/usr/bin/env python3
"""Bounded static-only preparation measurement; not native retained throughput."""
import argparse,json,time
from pathlib import Path
import eventwm_mass_collection as mass
import eventwm_geometry_v4 as geometry
from collector.core import read
p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=16);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
scenes=[s for s in read(mass.REPORT/'scene_inventory.json')['scenes'] if s.get('compatible')][:a.count]
rows=[]
for i,s in enumerate(scenes):
 t=time.perf_counter();sampler=dict(version=geometry.VERSION,component='geometry_feasible')
 try:
  ep,audit=geometry.episode(s,732000000+i,2,{},sampler)
  row=dict(scene=s['name'],seconds=time.perf_counter()-t,success=True,static_audit=audit,start=ep['start_position'])
 except RuntimeError as exc:row=dict(scene=s['name'],seconds=time.perf_counter()-t,success=False,error=str(exc))
 rows.append(row);print(json.dumps(row),flush=True)
a.output.write_text(json.dumps(dict(version=geometry.VERSION,scope='Static geometry preparation only; distribution-changing TRAIN research; no retained q/h measurement',rows=rows),indent=2))
