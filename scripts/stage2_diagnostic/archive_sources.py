"""Content-addressed, read-only source bundle for the diagnostic release."""
import hashlib
import json
from pathlib import Path
from .common import ROOT,OUT,digest,save

def main():
    files=list((ROOT/'scripts/stage2_diagnostic').glob('*.py'))
    for folder in ('stage2','stage2_quality','stage2_segment'):
        files.extend((ROOT/'eventwm_graph'/folder).glob('*.py'))
    files.extend([ROOT/'eventwm_graph/partial/map.py',ROOT/'scripts/stage2_segment_run.py'])
    mapping={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))}
    identity=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest();dest=OUT/'sources'/identity
    for relative,expected in mapping.items():
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target)!=expected:raise ValueError('source archive conflict')
        else:target.write_bytes((ROOT/relative).read_bytes());target.chmod(0o444)
    if not (dest/'manifest.json').exists():save(dest/'manifest.json',dict(identity=identity,files=mapping))
    save(OUT/'source_manifest.json',dict(identity=identity,archive=str(dest),files=mapping))
    print(identity)

if __name__=='__main__':main()
