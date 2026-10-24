"""Build a fresh guarded reproduction from EXACT archived scientific sources.

Run this as a file before importing eventwm_graph. It never uses the superseded
main-tree native selector and never writes to the historical input directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def setup(destination,input_root):
    base=Path(__file__).resolve().parents[2]
    archive=base/'docs/l2/intermittent_wm_20261022/artifacts/verified_execution'
    manifest=json.loads((archive/'manifest.json').read_text())
    first=archive/'shards/evaluation_shard_0000_0528/forecast_execution_guard.json'
    guard=json.loads(first.read_text());old_base=Path(manifest['source_root'])
    old_runtime=Path(guard['executed_model_file']).parents[2]
    input_root=Path(input_root).resolve();destination=Path(destination).resolve()
    if destination.exists():raise ValueError('preserve existing reproduction directory')
    verified={}
    for original,value in guard['inputs_before'].items():
        original=Path(original)
        if original.is_relative_to(old_runtime):relative=original.relative_to(old_runtime)
        elif original.is_relative_to(old_base):relative=original.relative_to(old_base)
        else:raise ValueError('unmapped historical input: '+str(original))
        current=input_root/relative
        if sha(current)!=value:raise ValueError('immutable input differs: '+str(current))
        verified[str(current)]=value
    for name,value in manifest['sealed_sources'].items():
        if sha(archive/'sources'/name)!=value:raise ValueError('archived executable source differs: '+name)
    destination.mkdir(parents=True)
    runtime=destination/'corrected_science_runtime';runtime.mkdir()
    for name,value in manifest['sealed_sources'].items():
        target=runtime/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(archive/'sources'/name,target)
        if sha(target)!=value:raise ValueError('source copy mismatch')
    (runtime/'output').mkdir();(runtime/'output/read_only_import_guard').mkdir()
    (runtime/'output/l2_graph_wm_120k_20261021').symlink_to(input_root/'output/l2_graph_wm_120k_20261021',target_is_directory=True)
    output=runtime/'output/l2_intermittent_wm_20261022';output.mkdir()
    for name in ('training','motion.json','motion_fit.json','train.npy','dev.npy'):
        (output/name).symlink_to(input_root/'output/l2_intermittent_wm_20261022'/name,
                                 target_is_directory=name=='training')
    for original in (input_root/'docs/l2').iterdir():
        if original.name!='intermittent_wm_20261022':
            (runtime/'docs/l2'/original.name).symlink_to(original,target_is_directory=True)
    orchestration=archive/'orchestration'
    for name in ('run_corrected_science.py','annotation_accounting.py','acquire_completed_outputs.py',
                 'merge_corrected_native.py','native_annotation_validity.json'):
        shutil.copyfile(orchestration/name,destination/name)
    relocated=dict(manifest,source_root=str(input_root),
        reproduction_original_manifest_sha256=sha(archive/'manifest.json'),
        reproduction_builder_sha256=sha(Path(__file__)),verified_relocated_inputs=verified,
        reproduction_scope='Exact archived149source bytes and delivered frozen inputs; fresh outputs, original inference/scoring semantics')
    (runtime/'manifest.json').write_text(json.dumps(relocated,indent=2)+'\n')
    print(json.dumps(dict(runtime=str(runtime),runner=str(destination/'run_corrected_science.py'),
        sealed_source_files=len(manifest['sealed_sources']),verified_inputs=len(verified),
        manifest_sha256=sha(runtime/'manifest.json')),indent=2))


if __name__=='__main__':
    base=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--input-root',type=Path,default=base)
    args=p.parse_args();setup(args.destination,args.input_root)
