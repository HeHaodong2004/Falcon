"""Bounded static-mesh fallback for configuration squares that cannot certify.

Uses isolated, previously audited asset FK and stage triangle tests from the
2026-10-20 research branch. No live human state or simulator physics is read.
"""
import math
from pathlib import Path
import numpy as np
from .search import human_cost,path_score


def attach(geometry,hmap,joints):
    from eventwm_graph.native import StaticGeometry
    from eventwm_graph.config import file_digest,digest
    from .mesh_visibility import StaticVisibility
    from .mesh_collision import StaticCollision
    from .ego_geometry import EgoGeometryCalibration
    asset=Path(__file__).with_name('assets')/'native_gait_boxes.npz'
    if file_digest(asset)!='d3ea9f1bb0dde811a217484453d5e525fcf6a3fe9cc26488249e1319217f7142':
        raise ValueError('Changed frozen robot geometry calibration')
    if not hasattr(hmap,'_l4_static_visibility'):
        static=StaticGeometry(hmap.source.metadata['scene'])
        try:hmap._l4_static_visibility=StaticVisibility.from_native(static)
        finally:static.close()
    with np.load(asset) as data:
        boxes=data['all_link_boxes'].copy();gait_joints=data['joint_positions'].copy()
    calibration=EgoGeometryCalibration(hmap.robot_config,boxes,gait_joints)
    try:receipt=calibration.update(joints,0)
    except Exception:calibration.close();raise
    union=receipt.gait_boxes[0].copy()
    union[:,:3]=receipt.gait_boxes[:,:,:3].min(0);union[:,3:6]=receipt.gait_boxes[:,:,3:6].max(0)
    precise=StaticCollision(hmap._l4_static_visibility,union)
    precise.arm_obbs=receipt.arm_obbs
    precise.gait_boxes=receipt.gait_boxes;precise.forward_count=0
    geometry.precise=precise;geometry.predictor=hmap.model
    geometry.version=digest(dict(grid=geometry.version,mesh=hmap._l4_static_visibility.version,
        calibration=file_digest(asset),initial_boxes=union.tolist(),margin=.025))
    return calibration


def update(geometry,calibration,joints,forward_count):
    receipt=calibration.update(joints,forward_count)
    geometry.precise.forward_count=forward_count
    geometry.precise.gait_boxes=receipt.gait_boxes
    if receipt.changed:
        boxes=receipt.gait_boxes[0].copy()
        boxes[:,:3]=receipt.gait_boxes[:,:,:3].min(0);boxes[:,3:6]=receipt.gait_boxes[:,:,3:6].max(0)
        geometry.precise.boxes=boxes;geometry.precise.arm_obbs=receipt.arm_obbs


def proposals(pose,target,terminal_yaw,terminal,people,geometry,spec,horizon,preferred):
    # A tight starting pose can require translating out before turning.
    # Keep these bounded native escapes even when they initially move away
    # from the reference. Their full sweeps undergo the same collision checks.
    variants=[(0,0,0),(0,0,4),(0,0,8)]+[(side,turns,forward) for side in (1,2) for turns in (4,8,16,24) for forward in (4,8)]
    best=None;best_score=math.inf;checks=0;cache={};partial=None;partial_key=(-1,-math.inf)
    for side,turns,forward in variants:
        path=[np.asarray(pose,float)];actions=[];valid=True;forward_count=geometry.precise.forward_count
        for depth in range(horizon):
            a=path[-1];dx=target[0]-a[0];dz=target[2]-a[2];arrived=math.hypot(dx,dz)<=.155
            desired=terminal_yaw if arrived else math.atan2(-dz,dx)
            angle=math.atan2(math.sin(desired-a[3]),math.cos(desired-a[3]))
            if arrived and (not terminal or abs(angle)<=spec.turn/2+1e-5):break
            if depth<turns:action=side
            elif depth<turns+forward:action=0
            else:action=(1 if angle>0 else 2) if abs(angle)>spec.turn/2+1e-5 else 0
            key=(tuple(a),action,forward_count)
            if key in cache:b,free=cache[key]
            else:
                b=geometry.predict(a,spec.actions[action],spec)
                # Sliding belongs to native dynamics, but zero translation is
                # not progress and never silently changes the candidate action.
                current=0 if forward_count==0 else 1+((forward_count-1)*5)%756
                following=1+(forward_count*5)%756 if action==0 else current
                boxes=geometry.precise.gait_boxes[current].copy()
                other=geometry.precise.gait_boxes[following]
                boxes[:,:3]=np.minimum(boxes[:,:3],other[:,:3]);boxes[:,3:6]=np.maximum(boxes[:,3:6],other[:,3:6])
                free=geometry.precise.swept(a,b,boxes=boxes,margin=.025)
                if action==0 and np.linalg.norm((b-a)[[0,2]])<.02:free=False
                cache[key]=(b,free)
            checks+=1
            if not free:valid=False;break
            safe,_,_=human_cost(a,b,depth,people,spec.max_dt)
            if not safe:valid=False;break
            if action==0:forward_count+=1
            actions.append(action);path.append(b)
        if valid and actions:
            actions=np.asarray(actions,dtype=np.int64);path=np.asarray(path)
            score=path_score(path,actions,target,terminal_yaw,terminal,preferred)
            if score<best_score:best_score=score;best=(actions,path)
            # Prefer the certified turn-then-forward proposal when available.
            if side==0:break
        elif actions:
            prefix=np.asarray(path);codes=np.asarray(actions,dtype=np.int64)
            key=(len(codes),-path_score(prefix,codes,target,terminal_yaw,terminal,preferred))
            if key>partial_key:partial_key=key;partial=(codes,prefix)
    if best is None:best=partial
    return (np.empty(0,np.int64),np.empty((0,4)),checks) if best is None else (*best,checks)
