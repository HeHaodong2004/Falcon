"""Versioned, checksummed compact research release; never replaces evidence."""
import argparse
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
from eventwm_graph.stage2_segment.common import BASE,OUT,dump,sha,snapshot


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);args=parser.parse_args()
    dest=BASE/'docs/l2/stage2_segment_20261024/artifacts'/args.name;dest.mkdir(exist_ok=False)
    records=[]
    def copy(path,label=None):
        path=Path(path);rel=Path(label) if label else path.relative_to(OUT);target=dest/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        if path.name=='calls.json':
            target=target.with_suffix('.json.gz')
            with path.open('rb') as source,target.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as encoded:
                shutil.copyfileobj(source,encoded)
        else:shutil.copyfile(path,target)
        records.append(dict(source=str(path),source_sha256=sha(path),source_bytes=path.stat().st_size,
                            artifact=str(target.relative_to(dest)),artifact_sha256=sha(target),artifact_bytes=target.stat().st_size))
    for rel in ('protocol.json','model_selection.json','segmentation_selection.json',
                'evaluation/final_segment_dev/metrics.json','evaluation/final_segment_dev/protocol.json',
                'evaluation/final_segment_dev/complete.json','baseline_untouched_source_equivalence_frozen.json',
                'verification_summary_replacement_audit.json','work_window.json','legacy_regression_fixtures.json'):
        path=OUT/rel
        if path.exists():copy(path)
    for folder in (OUT/'benchmark_inputs',*sorted((OUT/'benchmarks').glob('release_*_v1'))):
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.is_file():copy(path)
    roots=[]
    for path in sorted((OUT/'evaluation/final_segment_dev').glob('*.json.gz')):
        roots.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
    dump(dest/'evaluation_root_artifacts.json',dict(roots=roots,count=len(roots),scope='All local per-Root sufficient statistics; original files retained, not copied into Git'))
    env=dict(python=platform.python_version(),platform=platform.platform(),libc=list(platform.libc_ver()),packages={})
    for name in ('numpy','scipy','numba','llvmlite','torch','matplotlib','PyYAML'):
        try:env['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:env['packages'][name]=None
    dump(dest/'environment.json',env)
    identities=set()
    for rel in ('training/final/config.json','evaluation/final_segment_dev/protocol.json','benchmarks/release_single_v1/protocol.json'):
        data=json.loads((OUT/rel).read_text())
        def collect(value):
            if isinstance(value,dict):
                if 'identity' in value and 'path' in value and (Path(value['path'])/'manifest.json').exists():identities.add(value['identity'])
                for item in value.values():collect(item)
            elif isinstance(value,list):
                for item in value:collect(item)
        collect(data)
    current=snapshot();identities.add(current['identity']);bundles=[]
    for identity in sorted(identities):
        source=OUT/'sources'/identity;target=dest/'sources'/(identity+'.tar.gz');target.parent.mkdir(exist_ok=True)
        manifest=json.loads((source/'manifest.json').read_text())
        for rel,expected in manifest['files'].items():assert sha(source/rel)==expected
        with tarfile.open(target,'w:gz') as archive:archive.add(source,arcname=identity)
        bundles.append(dict(identity=identity,source=str(source),artifact=str(target.relative_to(dest)),sha256=sha(target)))
    dump(dest/'INDEX.json',dict(name=args.name,checkpoint_sha256=sha(OUT/'training/final/last.pt'),
        packaging_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
        exact_current_source=current,source_bundles=bundles,files=records,
        primary_runtime_commit='dede88127b0e94741b02efac5971f96a3455f02d',
        fitting_commit='7fc0f4bbd787e6032b55621748becbc72900af54',
        local_full_output=str(OUT),protected_payloads_opened=False,remote_push=False,
        scope='Compact release of trained/evaluated system; full immutable data, shards, logs and negative experiments retained locally'))
    print(json.dumps(dict(destination=str(dest),files=len(records),root_statistics=len(roots),source_bundles=len(bundles))))


if __name__=='__main__':main()
