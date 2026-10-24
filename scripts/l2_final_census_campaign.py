"""Full immutable train-only native census alongside bounded fitting."""
from pathlib import Path
import subprocess
import sys
import time
root=Path(__file__).resolve().parents[1]
out=root/"output/l2_final_800k_20261023"
while not (out/"training/configuration.json").exists():time.sleep(1)
for name,folder in (("main",out/"snapshot"),("supplement",out/"supplement")):
    subprocess.run([sys.executable,"-u",str(root/"scripts/l2_final_native_census.py"),"--snapshot",str(folder),"--output",str(out/("native_census_"+name+".json")),"--workers","3"],cwd=root,check=True)
