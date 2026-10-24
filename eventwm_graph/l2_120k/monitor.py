"""Read-only production and host observations; never controls the collector."""
import json
from pathlib import Path
import os
import shutil
import time

from .data import OUT

SOURCE=Path('/research-user/workspace/projects/eventwm-falcon-acceleration-v2/output/stage01_acceleration_v2_20261021/collection_user_authorized_5s_20261021_1')


def sample():
    status=json.loads((SOURCE/'status.json').read_text())
    keys=('collection_status','service_status','active_worker_count','expected_worker_count','automatic_training',
          'outcomes','roots','last_heartbeat_at','last_successful_outcome_at','live_cpu_seconds',
          'aggregate_rss_bytes','configuration_identity','collector_restart_count')
    memory={r.split(':')[0]:r.split(':')[1].strip() for r in Path('/proc/meminfo').read_text().splitlines()}
    return dict(time=time.time(),collector={k:status.get(k) for k in keys},load=os.getloadavg(),
        cpu_pressure=Path('/proc/pressure/cpu').read_text(),memory_available=memory['MemAvailable'],
        disk_free=shutil.disk_usage(OUT).free)


if __name__=='__main__':
    marker=OUT/'CAMPAIGN_ACTIVE';marker.touch(exist_ok=True)
    with open(OUT/'resources.jsonl','a',buffering=1) as f:
        while marker.exists():
            try:f.write(json.dumps(sample())+'\n')
            except Exception as e:f.write(json.dumps(dict(time=time.time(),error=repr(e)))+'\n')
            time.sleep(30)
