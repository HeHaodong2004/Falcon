"""Archive completed structural evidence; never read native future payloads.

Initial invocation creates a new artifacts directory. --refresh-checksums only
updates the publication checksum list after documentation/link corrections; it
never overwrites sealed experiments, checkpoints or the evidence manifest.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO/'docs/l2/partial_graph_wm_structural_20261021'
OUTPUT = REPO/'output/l2_partial_graph_wm_structural_20261021'
RUNTIME_REVISION = 'ec2ad5b1da4437b2a0d9d747d7659323af07450a'
ASSETS = {
    'graph_dynamics.pt': ('output/l2_partial_graph_wm_explore_20261021/learned_posdiff/checkpoint.pt',
                         '57c91156101320e2be6a02aee23d214a0405959e823b73be3786b0287bd25c05'),
    'population_prior.json': ('output/l2_partial_graph_wm_explore_20261021/train.npy.population.json',
                         'b7df625ac62e0eae09656fa32532b72e28039af140fc7b2803e7866b8f1f539c'),
    'observation_fit.json': ('output/l2_partial_graph_wm_repair_20261021/observation_fit.json',
                         'ace29a5160b4a8510ac96d3c7bfb3cbad0936fc809c2dd35bf302aaa874e8793'),
    'native_events_v5_scene.json': ('output/l2_partial_graph_wm_repair_20261021/native_events_v5_scene.json',
                         'feec91f6918963f618c8a44694fb210bcd92549c9a92e52fc394f2c0b6aa736b')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundle():
    target=DOCS/'artifacts';target.mkdir()
    (target/'raw').mkdir();(target/'checkpoints').mkdir()
    entries=[]
    def record(source,destination,encoding='identity'):
        entries.append({'source':str(source.relative_to(REPO)),'source_sha256':sha(source),
                        'source_bytes':source.stat().st_size,'bundle':str(destination.relative_to(DOCS)),
                        'bundle_sha256':sha(destination),'bundle_bytes':destination.stat().st_size,
                        'encoding':encoding})
    for source in sorted(OUTPUT.iterdir()):
        if not source.is_file():continue
        encoding='gzip' if source.stat().st_size>65536 and source.suffix in ('.json','.jsonl','.log') else 'identity'
        destination=target/'raw'/(source.name+('.gz' if encoding=='gzip' else ''))
        if encoding=='gzip':
            with source.open('rb') as src,destination.open('xb') as dst:
                with gzip.GzipFile(fileobj=dst,filename='',mode='wb',mtime=0,compresslevel=6) as stream:
                    shutil.copyfileobj(src,stream)
            if hashlib.sha256(gzip.decompress(destination.read_bytes())).hexdigest()!=sha(source):
                raise ValueError('archive round trip mismatch:'+str(source))
        else:
            with source.open('rb') as src,destination.open('xb') as dst:shutil.copyfileobj(src,dst)
        record(source,destination,encoding)
    for name,(relative,expected) in ASSETS.items():
        source=REPO/relative
        if sha(source)!=expected:raise ValueError('changed frozen checkpoint:'+relative)
        destination=target/'checkpoints'/name;shutil.copyfile(source,destination);record(source,destination)
    # Already frozen metadata; no dataset outcome/future payload is inspected.
    for relative,name in (
        ('docs/l2/partial_graph_wm_repair_20261021/artifacts/dataset_manifest.json.gz','dataset_manifest.json.gz'),
        ('docs/l2/partial_graph_wm_repair_20261021/artifacts/native_clock_diagnostic.json','native_clock_diagnostic.json'),
        ('output/l2_partial_graph_wm_explore_20261021/learned_posdiff/training.json','original_dynamics_training.json')):
        source=REPO/relative;destination=target/name;shutil.copyfile(source,destination);record(source,destination)
    manifest_hash=hashlib.sha256(gzip.decompress((target/'dataset_manifest.json.gz').read_bytes())).hexdigest()
    if manifest_hash!='92b80427e258db7f1ddef94a70e0d932dbe6d2ba834809ac322715f16599210c':
        raise ValueError('source snapshot manifest mismatch')
    config={'class':'eventwm_graph.l2_partial.structural_model:StructuralGraphBeliefWorldModel',
            'runtime_revision':RUNTIME_REVISION,'model_version':'partial-graph-structural-0.2-eba90f705a4a3d33',
            'kwargs':{'optimized':True,'residual_policy':'bound','maneuver_rate':2.,
                      'reset_std':2.,'maneuver_retention':1.},
            'checkpoints':{k:{'bundle_path':'checkpoints/'+k,'sha256':v[1]} for k,v in ASSETS.items()},
            'training_performed_in_this_round':False,'statistical_fitting':False,
            'state_schema':'partial_graph_structural_state_2','execution_device':'cpu',
            'decision':'REVISE — STRUCTURAL REVISE','latency_approved':False,
            'unresolved':['continuous exact refinement/bridge','global projection bounds','full-call and dense latency gates']}
    (target/'inference_config.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest={'schema':'structural_evidence_manifest_1','runtime_revision':RUNTIME_REVISION,
              'preserved_previous_revision':'d697d9665399793b426d77a315476214d3ccbd21',
              'snapshot_manifest_sha256':manifest_hash,'entries':entries,
              'note':'Complete source experiment bytes including failed attempts are retained. Empty stdout files are not successful JSON experiments. No new fit or future/holdout payload is included.'}
    (target/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Make raw evidence links usable in a checkout lacking ignored output files.
    lookup={str((REPO/e['source']).resolve()):DOCS/e['bundle'] for e in entries}
    for path in DOCS.rglob('*.md'):
        text=path.read_text()
        def replace(match):
            link=match.group(1)
            if '://' in link or link.startswith('#'):return match.group(0)
            resolved=str((path.parent/link).resolve())
            if resolved not in lookup:return match.group(0)
            import os
            return ']('+os.path.relpath(lookup[resolved],path.parent)+')'
        changed=re.sub(r'\]\(([^)]+)\)',replace,text)
        if changed!=text:path.write_text(changed)


def checksums():
    target=DOCS/'artifacts'
    if not (target/'evidence_manifest.json').exists():raise FileNotFoundError('create the evidence bundle first')
    files=set(p for p in DOCS.rglob('*') if p.is_file())
    files.update((REPO/'eventwm_graph/l2_partial').glob('*.py'))
    files.update((REPO/'tests/l2_partial_structural').glob('*.py'))
    files.update((REPO/'scripts').glob('l2_partial_structural_*.py'))
    checksum_file=target/'release_files.sha256';files.discard(checksum_file)
    checksum_file.write_text(''.join(sha(p)+'  '+str(p.relative_to(REPO))+'\n' for p in sorted(files)))
    return len(files)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-checksums',action='store_true')
    args=parser.parse_args()
    if not args.refresh_checksums:bundle()
    print(json.dumps({'docs':str(DOCS.relative_to(REPO)),'checksummed_files':checksums(),
                      'runtime_revision':RUNTIME_REVISION,'source_experiments_overwritten':False}))


if __name__=='__main__':main()
