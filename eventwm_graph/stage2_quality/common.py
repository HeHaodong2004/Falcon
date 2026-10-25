import hashlib
import json
import os
from pathlib import Path
from eventwm_graph.stage2.common import dump,sha,seed
BASE=Path(os.environ.get('EVENTWM_STAGE2_RESEARCH_BASE',Path(__file__).resolve().parents[2])).resolve()
ORIGINAL=Path(os.environ.get('EVENTWM_STAGE2_BASE','/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023')).resolve()
OUT=BASE/'output/stage2_quality_runtime_20261024'
SCHEMA='trajectory_conditioned_graphdelta_quality_2'
MODES=('baseline','trajectory','map','both')
FEATURES=92

def identity():
    files={str(p.relative_to(BASE)):sha(p) for p in sorted((BASE/'eventwm_graph/stage2_quality').glob('*.py'))}
    return dict(files=files,sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())


def snapshot(purpose):
    """Own immutable source bytes for each actual fit/extraction phase."""
    data={str(p.relative_to(BASE)):p.read_bytes() for p in sorted((BASE/'eventwm_graph/stage2_quality').glob('*.py'))}
    files={k:hashlib.sha256(v).hexdigest() for k,v in data.items()};key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    dest=OUT/'source_snapshots'/key
    for rel,value in data.items():
        path=dest/rel;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if path.read_bytes()!=value:raise ValueError('immutable source snapshot collision')
        else:path.write_bytes(value);path.chmod(0o444)
    manifest=dict(identity=key,files=files,purpose=purpose)
    if not (dest/'manifest.json').exists():dump(dest/'manifest.json',manifest)
    return dict(identity=key,path=str(dest),files=files)
