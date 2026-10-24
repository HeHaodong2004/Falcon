"""Read-only collector/host sampling; all writes stay in NEW research output."""
import json
import time
from .common import OUT
from eventwm_graph.l2_120k.monitor import sample


if __name__=='__main__':
    marker=OUT/'MONITOR_ACTIVE';marker.touch(exist_ok=False)
    try:
        with open(OUT/'resources.jsonl','a',buffering=1) as f:
            while marker.exists():
                try:f.write(json.dumps(sample())+'\n')
                except Exception as e:f.write(json.dumps(dict(time=time.time(),error=repr(e)))+'\n')
                time.sleep(30)
    finally:
        marker.unlink(missing_ok=True)
