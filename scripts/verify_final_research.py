"""Verify saved research artifacts without training or opening protected labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
REGISTRY=BASE/'docs/l2/final_stage1_stage2_20261024/ARTIFACTS.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def check(item):
    name,value=item;path=Path(name)
    if not path.is_file():return dict(path=name,error='missing')
    if path.stat().st_size!=value['bytes'] or sha(path)!=value['sha256']:return dict(path=name,error='checksum')
    return None


def main(full_dataset=False):
    registry=json.loads(REGISTRY.read_text());errors=[];checked=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(check,registry['files'].items()):
            checked+=1
            if result:errors.append(result)
    if full_dataset:
        manifest_path=BASE/'output/research_transition_20261024/final_collection_all_splits/manifest.json'
        manifest=json.loads(manifest_path.read_text());index=Path(manifest['rows']['path'])
        if sha(index)!=manifest['rows']['sha256']:errors.append(dict(path=str(index),error='index checksum'))
        def members():
            with gzip.open(index,'rt') as stream:
                for line in stream:
                    row=json.loads(line);folder=Path(row['folder'])
                    for name,digest in row['files'].items():
                        path=folder/name
                        yield str(path),dict(bytes=path.stat().st_size,sha256=digest)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(check,members()):
                checked+=1
                if result:errors.append(result)
    print(json.dumps(dict(status='PASS' if not errors else 'FAIL',checked_files=checked,errors=errors,
        scope='Saved artifact bytes and optional opaque dataset bytes; not a scientific acceptance claim.'),indent=2))
    raise SystemExit(bool(errors))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full-dataset',action='store_true');main(**vars(p.parse_args()))
