"""Version small loadable research checkpoints and exact source bundles."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

BASE=Path(__file__).resolve().parents[1]
DOC=BASE/'docs/l2/final_stage1_stage2_20261024'
OUT=BASE/'output/research_transition_20261024'
S2=BASE/'output/stage2_graphdelta_20261024'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(source,target):
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as stream:
        for p in sorted(source.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix not in ('.py','.json'):continue
            data=p.read_bytes();info=tarfile.TarInfo(str(p.relative_to(source)));info.size=len(data);info.mode=0o444;info.mtime=0
            stream.addfile(info,io.BytesIO(data))
    target.write_bytes(gzip.compress(buffer.getvalue(),mtime=0))


def main():
    dest=DOC/'release';dest.mkdir(exist_ok=True)
    for stage,original in [('stage1',OUT/'stage1_research_release'),('stage2',OUT/'stage2_research_release')]:
        target=dest/stage;target.mkdir(exist_ok=True)
        for p in original.iterdir():
            if p.is_file():shutil.copyfile(p,target/p.name);assert sha(p)==sha(target/p.name)
    source1=Path((OUT/'inference_source_path.txt').read_text().strip())
    source2=Path((S2/'runtime_source_path.txt').read_text().strip())
    for source,name in [(source1,'stage1_inference_source.tar.gz'),(OUT/'training_source','stage1_training_source.tar.gz'),
                        (source2,'stage2_runtime_source.tar.gz'),(Path((S2/'source_path.txt').read_text().strip()),'stage2_training_source.tar.gz')]:
        archive(source,dest/name)
    for name in ('environment.json','package_versions.json','stage1_inference_config.json'):shutil.copyfile(OUT/name,dest/name)
    manifests=[OUT/'final_collection_all_splits/manifest.json',BASE/'output/l2_two_stage_20261023/final_collection_20261024/manifest.json']
    summary=dict(complete_collection=dict(path=str(manifests[0]),sha256=sha(manifests[0]),roots=186089,outcomes=1341492,scenes=879),
        fitting=dict(path=str(manifests[1]),sha256=sha(manifests[1]),roots=105838,outcomes=769349,scenes=476,anchors=322897),
        note='Large immutable data remain at recorded paths; protected payloads are never part of the model/source bundles.')
    (dest/'DATASET.json').write_text(json.dumps(summary,indent=2)+'\n')
    readme=dest/'README.md';readme.write_text('# Loadable research artifacts\n\n'
        'stage1/checkpoint.pt is the final Stage1 research checkpoint; stage2/checkpoint.pt is the initial Stage2 research baseline. Both include exact configurations and full summaries. Source archives preserve the exact Python closures; extract the desired runtime bundle, put it in PYTHONPATH and run outside the repository root. Stage1 is closed; Stage2 is primary.\n\n'
        'The full dataset remains independently frozen at DATASET.json paths. Reproduction and inference commands are in ../REPRODUCE.md. No further training is needed to load either checkpoint.\n')
    for report,stage in [('A_FINAL_STAGE1_RESEARCH_RELEASE.md','stage1'),('B_STAGE2_RESEARCH.md','stage2')]:
        p=DOC/report;text=p.read_text();old=str(OUT/('stage1_research_release' if stage=='stage1' else 'stage2_research_release')/'checkpoint.pt')
        p.write_text(text.replace(old,str(dest/stage/'checkpoint.pt')))
    registry_path=DOC/'ARTIFACTS.json';registry=json.loads(registry_path.read_text())
    for p in list(dest.rglob('*'))+[Path(__file__),DOC/'A_FINAL_STAGE1_RESEARCH_RELEASE.md',DOC/'B_STAGE2_RESEARCH.md']:
        if p.is_file():registry['files'][str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    registry_path.write_text(json.dumps(registry,indent=2)+'\n')
    complete=OUT/'COMPLETE.json';m=json.loads(complete.read_text());m['versioned_release']=str(dest);m['artifacts_sha256']=sha(registry_path)
    complete.write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(dict(versioned_artifacts=str(dest),total_bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()))))


if __name__=='__main__':main()
