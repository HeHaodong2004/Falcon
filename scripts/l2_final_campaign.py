"""One complete maximum-compatible campaign, preserving completed shards."""
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from eventwm_graph.l2_final.common import OUT,sha,dump
from eventwm_graph.l2_final import prepare
prepare.run(3)
main=json.loads((OUT/"preparation.json").read_text());dump(OUT/"preparation_main.json",main)
supp=OUT/"supplement_preparation";supp.mkdir(exist_ok=True)
if not (supp/"snapshot").exists():(supp/"snapshot").symlink_to(OUT/"supplement",target_is_directory=True)
prepare.OUT=supp;prepare.run(3)
extra=json.loads((supp/"preparation.json").read_text())
shards=main["shards"]+[dict(r,path="supplement_preparation/"+r["path"]) for r in extra["shards"]]
scenes=set()
import numpy as np
for row in shards:
    scenes.update(r["scene_id"] for r in np.load(OUT/row["path"],allow_pickle=True))
statuses=dict(main["statuses"])
for k,v in extra["statuses"].items():statuses[k]=statuses.get(k,0)+v
hygiene=dict(main["target_hygiene"])
for k,v in extra["target_hygiene"].items():hygiene[k]=hygiene.get(k,0)+v
dump(OUT/"preparation.json",dict(eligible_roots=main["eligible_roots"]+extra["eligible_roots"],motion_roots=main["motion_roots"]+extra["motion_roots"],motion_scenes=len(scenes),shards=shards,statuses=statuses,target_hygiene=hygiene,final_snapshot_sha256=sha(OUT/"final_snapshot_manifest.json"),main=main,supplement=extra))
subprocess.run([sys.executable,"-u","-m","eventwm_graph.l2_final.train"],cwd=root,check=True)
