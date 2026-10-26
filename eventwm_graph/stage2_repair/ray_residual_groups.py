"""Camera-line queries from the supplied path and legally current camera only."""
import numpy as np
from .diagnose import HF,read,source_rows,member
from .range_ray_data import camera_lines
from .legal_camera import current_extrinsic
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment


def attach(groups,training=False):
    rows,_=source_rows('train'if training else'dev');rows={r['root_id']:r for r in rows};cache_rid=None
    for group in groups:
        if 'ray_origins'in group:continue
        rid=group['root_id']
        if rid!=cache_rid:
            legal=member(rows[rid],'legal_root.json.gz');nodes=member(rows[rid],'future_observations.json.gz');packed=read(HF/'packed'/('pilot_fit'if training else'dev_pilot')/(rid+'.json'));a=load_arrays(packed);lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(packed['segments'])};cache_rid=rid
        sid=group['start_node'];current=legal['history'][-1]if sid is None else nodes[sid]['observation'];origins=np.empty((len(group['y']),3,2),np.float32)
        for e in group['segments']:
            path=get_segment(a,lookup[e['start_node'],e['end_node']])[3]
            if abs(float(current['timestamp'])-float(path[0,0]))>1.e-7:raise ValueError('current camera timestamp differs from query start')
            translation,_=current_extrinsic(path[:1],current['camera_origin'],current['camera_rotation']);origins[e['start']:e['end']]=camera_lines(path,translation,a['geometry'][0],group['low'])
        group['ray_origins']=origins
    return groups
