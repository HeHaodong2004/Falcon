"""Incremental benchmark surface extraction; original assets are read only."""
from pathlib import Path
import fcntl
import time
from ..config import REPO,read_json,write_json,digest,file_digest,load_robot_config


def inventory(output=None):
    paths=read_json(REPO/'output/stage0_benchmark/audit/scene_inventory.json')
    known={r['id']:r for r in read_json(REPO/'configs/stage0_scenes.json')}
    rows=[];unique={}
    for raw in paths:
        p=Path(raw);family='mp3d' if '/mp3d/' in raw else 'hm3d'
        name=p.name.split('.')[0];sid=family+'_'+name
        candidates=[p.with_suffix('.navmesh'),REPO/'data/scene_datasets/navmeshes'/(name+'.navmesh')]
        row={'id':sid,'scene':str(p.resolve()),'family':family,'bytes':p.stat().st_size,
             'navmesh':next((str(x.resolve()) for x in candidates if x.exists()),None)}
        row['aliases']=[str(p.resolve())]
        if sid in known:row.update(known[sid])
        if sid in unique:
            old=unique[sid]
            if file_digest(old['scene'])!=file_digest(p):raise ValueError('Nonidentical scene alias: '+sid)
            old['aliases'].append(str(p.resolve()))
        else:unique[sid]=row;rows.append(row)
    # Deterministic scene diversity, with MP3D every fourth scene until exhausted.
    groups={f:sorted([r for r in rows if r['family']==f],key=lambda r:digest(r['id'])) for f in ('mp3d','hm3d')}
    result=[]
    while any(groups.values()):
        for f in ('mp3d','hm3d','hm3d','hm3d'):
            if groups[f]:result.append(groups[f].pop(0))
    if output:write_json(output,result)
    return result


def build_certified_scene(row,output,robot_path=None):
    """Construct/certify one scene; UNKNOWN failures remain visible artifacts."""
    from ..native import StaticGeometry
    from ..cli import prepare_navmesh
    from ..map import build_navmesh_hierarchical_map
    from .certify import certify_scene
    from .acceptance import grade
    from .map import load_certified_map
    output=Path(output).resolve();scene_dir=output/'scenes'/row['id'];scene_dir.mkdir(parents=True,exist_ok=True)
    with (scene_dir/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=scene_dir/'ready.json'
        if receipt.exists():
            saved=read_json(receipt);load_certified_map(saved['manifest']);return saved
        # Reuse the independently audited six-scene suite exactly.
        existing=REPO/'output/stage0_benchmark/certified'/row['id']/'latest.json'
        began=time.perf_counter()
        if existing.exists():
            manifest=read_json(existing)['manifest'];h=load_certified_map(manifest)
            source_metrics={'reuse':'certified_stage0_suite','seconds':time.perf_counter()-began}
        else:
            robot=load_robot_config(robot_path or REPO/'output/stage0/audit/robot.json')
            cache=output/'geometry_cache'
            with StaticGeometry(row['scene']) as geometry:
                navmesh,mode=prepare_navmesh(row['scene'],robot,cache,geometry,row.get('navmesh'))
                h=build_navmesh_hierarchical_map(row['id'],navmesh,robot,cache,scene=row['scene'],geometry=geometry)
                source_metrics={'navmesh_mode':mode,**h.build_metrics}
                source=h.manifest_path
            raw=certify_scene(source,output/'execution_evidence',multi_routes=20,contact_diagnostics=False)
            graded=grade(output/'execution_evidence'/row['id']/raw['map_id']/'manifest.json',output/'certified')
            manifest=output/'certified'/row['id']/graded['map_id']/'manifest.json'
            # Hard gate. A completed extraction with failed coverage does not
            # become production support; the failure report is kept for repair.
            h=load_certified_map(manifest)
        result={'scene':row,'manifest':str(Path(manifest).resolve()),'map_id':h.map_id,
                'report':h.report,'source_metrics':source_metrics,'total_seconds':time.perf_counter()-began,
                'ready':True,'schema':'corpus_map_readiness_1'}
        write_json(receipt,result);return result
