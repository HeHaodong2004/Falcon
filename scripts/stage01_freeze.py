#!/usr/bin/env python
"""Freeze the validated collector while retaining the existing asset root."""
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eventwm_graph.config import digest,file_digest,write_json


def freeze(output):
    tracked=subprocess.check_output(['git','ls-files','eventwm_graph'],cwd=ROOT,text=True).splitlines()
    paths=[ROOT/p for p in tracked if p.endswith('.py')]
    paths += list((ROOT/'scripts').glob('stage01_*.py'))
    if not any(p.parent.name=='partial' for p in paths):
        raise ValueError('Commit the reviewed partial pipeline before freezing deployment')
    paths += [ROOT/'scripts/collector/observation.py',ROOT/'scripts/collector/__init__.py',ROOT/'configs/stage0_scenes.json']
    paths=[p for p in paths if p.exists() and '__pycache__' not in p.parts]
    sources={str(p.relative_to(ROOT)):file_digest(p) for p in paths}
    dest=Path(output)/digest(sources)
    if (dest/'manifest.json').exists():return dest.resolve()
    for path in paths:
        target=dest/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    config=dest/'eventwm_graph/config.py'
    old='REPO = Path(__file__).resolve().parents[1]'
    shim=('REPO = Path('+repr(str(ROOT))+')\nimport sys as _bundle_sys\n'
          'if str(REPO) not in _bundle_sys.path: _bundle_sys.path.append(str(REPO))')
    if old not in config.read_text():raise ValueError('Unknown asset root implementation')
    config.write_text(config.read_text().replace(old,shim))
    external={p:file_digest(ROOT/p) for p in ('falcon/additional_action.py','falcon/additional_metric.py')}
    write_json(dest/'manifest.json',{'source_id':digest(sources),'original_sources':sources,
        'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'external_sources':external,'repository_assets':str(ROOT),
        'frozen_sources':{str(p.relative_to(dest)):file_digest(p) for p in dest.rglob('*') if p.is_file()},
        'automatic_training':False})
    for path in dest.rglob('*'):
        if path.is_file():path.chmod(0o444)
    return dest.resolve()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    print(freeze(p.parse_args().output))
