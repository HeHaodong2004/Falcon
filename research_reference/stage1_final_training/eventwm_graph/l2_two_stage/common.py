"""Research paths and content identities; never collector control operations."""
from pathlib import Path
import os
import hashlib
from eventwm_graph.l2_final.common import SOURCE, OLD, CORE, sha, dump

BASE = Path(os.environ.get('EVENTWM_TWO_STAGE_BASE',Path(__file__).resolve().parents[2])).resolve()
OUT = BASE / "output/l2_two_stage_20261023"
DOC = BASE / "docs/l2/two_stage_20261023"
PREVIOUS = Path("/research-user/workspace/projects/eventwm-falcon-l2-final-800k-20261023/output/l2_final_800k_20261023")


def seed_for(root,stream,anchor=0):
    return int.from_bytes(hashlib.sha256(('two-stage-A:%d:%s:%d'%(stream,root,anchor)).encode()).digest()[:4],'little')
