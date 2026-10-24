"""Experiment paths and immutable-input utilities."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT = BASE/'output/l2_intermittent_500k_20261022'
DOC = BASE/'docs/l2/intermittent_500k_20261022'
BASELINE = Path('/research-user/workspace/projects/eventwm-falcon-l2-partial-20261021')
PRIOR = BASELINE/'output/l2_intermittent_wm_20261022'
OLD = BASELINE/'output/l2_graph_wm_120k_20261021'
SOURCE = Path('/research-user/workspace/projects/eventwm-falcon-acceleration-v2/output/stage01_acceleration_v2_20261021/collection_user_authorized_5s_20261021_1')
CORE = ('map.py', 'observer.py', 'query.py', 'metrics.py', 'execution.py',
        'eventwm_graph/stage1/runtime.py', 'scripts/collector/observation.py',
        'falcon/additional_action.py', 'falcon/additional_metric.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def copy_frozen(source,target,expected=None):
    import os
    import shutil
    source,target=Path(source),Path(target)
    if source.is_symlink() or target.exists():raise ValueError('require an ordinary source and new destination')
    before=sha(source)
    if expected is not None and before!=expected:raise ValueError('source hash differs: '+str(source))
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    if sha(target)!=before or sha(source)!=before:raise ValueError('source/copy changed')
    if source.stat().st_ino==target.stat().st_ino:raise ValueError('mutable inode alias')
    os.chmod(target,0o444)
    return before
