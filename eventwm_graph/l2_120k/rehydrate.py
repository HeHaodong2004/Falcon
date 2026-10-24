"""Recreate exactly the sealed manifest's eligible payloads, never a new live cut."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
from eventwm_graph.l2_partial.snapshot import sha


def main(manifest,destination):
    source=Path(manifest);raw=gzip.decompress(source.read_bytes()) if source.suffix=='.gz' else source.read_bytes()
    m=json.loads(raw);dest=Path(destination).resolve();repo=Path(__file__).resolve().parents[2]
    if not dest.is_relative_to(repo/'output') or dest.exists():raise ValueError('new research output snapshot required')
    dest.mkdir(parents=True)
    for i,r in enumerate(m['rows']):
        if r['split'] not in ('train','dev') or r['source_split']!='train':raise ValueError('protected payload requested')
        src=Path(r['source']);target=dest/'roots'/r['root_id'];target.mkdir(parents=True)
        for name,expected in r['files'].items():
            if Path(name).name!=name or (src/name).is_symlink():raise ValueError('invalid sealed payload path')
            if sha(src/name)!=expected:raise ValueError('source hash mismatch')
            shutil.copyfile(src/name,target/name)
            if sha(target/name)!=expected or sha(src/name)!=expected:raise ValueError('changed sealed source/copy')
            os.chmod(target/name,0o444)
        if (i+1)%1000==0:print('rehydrated',i+1,flush=True)
    (dest/'manifest.json').write_bytes(raw);os.chmod(dest/'manifest.json',0o444)
    print(json.dumps(dict(manifest_sha256=hashlib.sha256(raw).hexdigest(),roots=len(m['rows']),
        method='exact frozen eligible root IDs; no live database read or new split',
        original_journal_sha256=m['journal_sha256'],journal_recreated=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('destination');main(**vars(p.parse_args()))
