"""Freeze executable training/preparation modules before a substantial job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from .common import BASE,OUT,sha,dump


def run():
    package=Path(__file__).parent
    files={str(p.relative_to(BASE)):sha(p) for p in sorted(package.glob('*.py'))}
    for name in ('eventwm_graph/__init__.py','eventwm_graph/l2_final/__init__.py','eventwm_graph/l2_final/common.py',
                 'eventwm_graph/partial/__init__.py','eventwm_graph/partial/map.py'):
        files[name]=sha(BASE/name)
    stubs={'eventwm_graph/__init__.py': '"""Frozen Stage1 namespace; unrelated simulator imports excluded."""\n',
           'eventwm_graph/partial/__init__.py': '"""Frozen causal map dependency."""\nfrom .map import PartialGraph, UNKNOWN, KNOWN_FREE, KNOWN_OCCUPIED\n'}
    original_parent_hashes={name:files[name] for name in stubs}
    for name,value in stubs.items():files[name]=hashlib.sha256(value.encode()).hexdigest()
    identity=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    dest=OUT/'frozen_sources'/identity
    if not dest.exists():
        dest.mkdir(parents=True)
        for name,expected in files.items():
            target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
            if name in stubs:target.write_text(stubs[name])
            else:shutil.copyfile(BASE/name,target)
            assert sha(target)==expected;os.chmod(target,0o444)
        dump(dest/'manifest.json',dict(schema='two_stage_frozen_executable_source_1',identity=identity,files=files,workspace=str(BASE),
            namespace_stubs=stubs,original_parent_hashes=original_parent_hashes))
        os.chmod(dest/'manifest.json',0o444)
    print(str(dest),flush=True)


if __name__=='__main__':run()
