import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'output/l2_final_800k_20261023'
DOC=BASE/'docs/l2/final_800k_20261023'
VALIDATED=Path('/research-user/workspace/projects/eventwm-falcon-l2-intermittent-500k-20261022')
PREVIOUS=VALIDATED/'output/l2_intermittent_500k_20261022'
LEGACY=Path('/research-user/workspace/projects/eventwm-falcon-l2-partial-20261021')
OLD=LEGACY/'output/l2_graph_wm_120k_20261021'
PRIOR=LEGACY/'output/l2_intermittent_wm_20261022'
SOURCE=Path('/research-user/workspace/projects/eventwm-falcon-acceleration-v2/output/stage01_acceleration_v2_20261021/collection_user_authorized_5s_20261021_1')
CORE=('map.py','observer.py','query.py','metrics.py','execution.py','eventwm_graph/stage1/runtime.py',
      'scripts/collector/observation.py','falcon/additional_action.py','falcon/additional_metric.py')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
