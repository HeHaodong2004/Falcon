import hashlib
import json
from pathlib import Path
from eventwm_graph.stage2.common import dump, sha

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'output/stage2_segment_20261024'
QUALITY = Path('/home/haodong/workspace/projects/eventwm-falcon-stage2-quality-20261024')
CHECKPOINT = QUALITY / 'output/stage2_quality_runtime_20261024/training/final/last.pt'
CHECKPOINT_SHA = '9a2cc072f9d8047ea2b9f7b66712c67dbb260d81d40b51bea984ebf6175b6504'
SCHEMA = 'segment_conditioned_legal_observation_1'

def stable(value):
    return int(hashlib.sha256(('segment-2424:' + value).encode()).hexdigest()[:16], 16)

def sources():
    paths = [*BASE.joinpath('eventwm_graph/stage2_segment').glob('*.py'),
             *BASE.joinpath('eventwm_graph/stage2_quality').glob('*.py'),
             *BASE.joinpath('eventwm_graph/stage2').glob('*.py'),
             *BASE.joinpath('scripts').glob('stage2_segment*.py'),
             BASE / 'eventwm_graph/partial/map.py']
    return {str(p.relative_to(BASE)): sha(p) for p in sorted(paths)}

def snapshot():
    files = sources()
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    dest = OUT / 'sources' / identity
    for rel, expected in files.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            assert sha(p) == expected
        else:
            p.write_bytes((BASE / rel).read_bytes())
            p.chmod(0o444)
    if not (dest / 'manifest.json').exists():
        dump(dest / 'manifest.json', dict(identity=identity, files=files))
    return dict(identity=identity, path=str(dest))
