"""Versioned Stage2 research paths and artifact identities."""
import hashlib
import json
import os
from pathlib import Path

BASE=Path(os.environ.get('EVENTWM_STAGE2_BASE',Path(__file__).resolve().parents[2])).resolve()
OUT=BASE/'output/stage2_graphdelta_20261024'
FINAL=BASE/'output/l2_two_stage_20261023/final_collection_20261024/manifest.json'
OLD=Path('/research-user/workspace/projects/eventwm-falcon-l2-partial-20261021/output/l2_graph_wm_120k_20261021')
SCHEMA='trajectory_conditioned_observed_graphdelta_1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def seed(root,query=''):
    return int(hashlib.sha256(('stage2-v1:'+root+':'+query).encode()).hexdigest()[:8],16)


def source_identity():
    files={'eventwm_graph/stage2/'+p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    from eventwm_graph.partial import map as partial_map
    files['eventwm_graph/partial/map.py']=sha(partial_map.__file__)
    return files
