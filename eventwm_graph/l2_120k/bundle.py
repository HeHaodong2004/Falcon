"""Bundle immutable evidence/checkpoints and checksums into the research commit."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
import numpy as np
import scipy
import numba
from eventwm_graph.l2_partial.snapshot import sha
from .data import OUT,SNAP

DOC=Path('docs/l2/graph_wm_120k_20261021')


def gzcopy(src,dst):
    with open(src,'rb') as incoming,open(dst,'wb') as out:
        with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as zipped:
            shutil.copyfileobj(incoming,zipped)


def main():
    artifacts=DOC/'artifacts';raw=artifacts/'raw';checkpoints=artifacts/'checkpoints'
    raw.mkdir(parents=True,exist_ok=True);checkpoints.mkdir(parents=True,exist_ok=True)
    copied=[]
    for file in OUT.rglob('*'):
        if not file.is_file() or file.suffix not in ('.json','.jsonl','.log','.stderr'):continue
        if 'snapshot' in file.relative_to(OUT).parts or 'evaluation_pipeline_smoke_v1' in file.parts:continue
        rel=file.relative_to(OUT);target=raw/rel
        target.parent.mkdir(parents=True,exist_ok=True);target=target.with_name(target.name+'.gz');gzcopy(file,target)
        copied.append(dict(source=str(file),source_sha256=sha(file),bundled=str(target),bundled_sha256=sha(target)))
    gzcopy(SNAP/'manifest.json',artifacts/'dataset_manifest.json.gz')
    for name in ('checkpoint.pt','epoch0.pt','last.pt'):
        shutil.copyfile(OUT/'training'/name,checkpoints/name)
    for name in ('configuration.json','training.json','learning.json'):
        shutil.copyfile(OUT/'training'/name,artifacts/name)
    for name in ('population_prior.json','motion.json','fit_input_manifest.json','release_summary.json'):
        shutil.copyfile(OUT/name,artifacts/name)
    shutil.copyfile(OUT/'source_fit_model.py',artifacts/'source_fit_model.py')
    old=Path('docs/l2/partial_graph_wm_structural_20261021/artifacts/checkpoints')
    for name in ('graph_dynamics.pt','native_events_v5_scene.json','observation_fit.json'):
        shutil.copyfile(old/name,checkpoints/('frozen_'+name))
    historical=Path('output/l2_partial_graph_wm_explore_20261021/baseline_calibration.json')
    if historical.exists():shutil.copyfile(historical,artifacts/'historical_cv_calibration.json')
    baseline=dict(A_visible_cv=dict(diffusion=.16,visible_only=True,maneuver_rate=0.),
        B_memory_cv=dict(diffusion=.16,visible_only=False,maneuver_rate=0.),
        B_historical_cv=dict(diffusion=.64,visible_only=False,maneuver_rate=0.),
        C_frozen_graph=dict(checkpoint='frozen_graph_dynamics.pt',maneuver_rate=2.,reset_std=2.),
        C_matched_motion=dict(checkpoint='frozen_graph_dynamics.pt',motion='motion.json'),
        common_history='normalized legal-history filter; all siblings grouped; common train-only prior and optical law')
    (artifacts/'baselines.json').write_text(json.dumps(baseline,indent=2))
    environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,
        scipy=scipy.__version__,numba=numba.__version__,cuda=torch.version.cuda,
        platform=subprocess.run(['uname','-a'],capture_output=True,text=True).stdout.strip(),
        git_base='c532e719',git_source_before_bundle=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
        no_production_changes=True,env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG')})
    (artifacts/'environment.json').write_text(json.dumps(environment,indent=2));(artifacts/'evidence_manifest.json').write_text(json.dumps(copied,indent=2))
    # Hash every new release doc/artifact/source and campaign test; exclude list.
    files=[p for base in (DOC,Path('eventwm_graph/l2_120k'),Path('tests/l2_120k')) for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='release_files.sha256']
    (artifacts/'release_files.sha256').write_text(''.join(sha(p)+'  '+str(p)+'\n' for p in sorted(files)))
    print(json.dumps(dict(files=len(files),selected_checkpoint_sha256=sha(checkpoints/'checkpoint.pt'),bundled_evidence=len(copied))),flush=True)


if __name__=='__main__':main()
