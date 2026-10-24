"""Execution-only Stage2 optimization; reuse the one trained checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'output/stage2_graphdelta_20261024'
PYTHON='/research-user/miniforge3/envs/eventwm-falcon/bin/python'
PARENT='fc67e2ed3dfc88b1bc41dbfa41f01a77015b46a0193e3982ce6228a2fa64e36d'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original=OUT/'frozen_sources'/PARENT;old=json.loads((original/'manifest.json').read_text())
    files={name:(BASE/name).read_bytes() if name.startswith('eventwm_graph/stage2/') else (original/name).read_bytes() for name in old['files']}
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}
    changed=[name for name in hashes if hashes[name]!=old['files'][name]]
    assert set(changed)=={'eventwm_graph/stage2/api.py','eventwm_graph/stage2/features.py','eventwm_graph/stage2/benchmark.py'}
    identity=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest();source=OUT/'frozen_sources'/identity
    checkpoint=OUT/'training/final/last.pt';complete=json.loads((checkpoint.parent/'complete.json').read_text())
    assert sha(checkpoint)==complete['checkpoint_sha256']
    if not source.exists():
        for name,data in files.items():
            p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(0o444)
        value=dict(identity=identity,files=hashes,parent=PARENT,changes=changed,checkpoint_sha256=sha(checkpoint),
            scope='Execution-only; same learned weights, candidate support,47features, classes, heights, deterministic updater and scoring rules.',
            changes_detail=['Reuse exact k4 nearest results; cache pose trig/time; preallocate original rounded47float32 features.',
                'Tile independent NNrows8192; stream batch inputs; avoid copying updater-owned structures twice.',
                'Cold child cwd=/tmp, with explicit actual namespace import paths/hashes; original repository-namespace results retained.'])
        (source/'manifest.json').write_text(json.dumps(value,indent=2)+'\n');(source/'manifest.json').chmod(0o444)
    (OUT/'runtime_source_path.txt').write_text(str(source)+'\n')
    dest=OUT/'optimized_runtime';dest.mkdir(exist_ok=True)
    plan=dict(source=str(source),source_identity=identity,source_parent=PARENT,checkpoint_sha256=sha(checkpoint),
        training_repeated=False,data_changed=False,scoring_protocol_changed=False,
        evaluation='optimized_dev',benchmark='optimized',reference_evaluation='final_dev',reference_benchmark='final')
    (dest/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env=dict(os.environ,EVENTWM_STAGE2_BASE=str(BASE),PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    jobs=[]
    for module,name in [('evaluate','optimized_dev'),('benchmark','optimized')]:
        output=OUT/('evaluation' if module=='evaluate' else 'benchmarks')/name/'metrics.json'
        if output.exists():continue
        log=(dest/(module+'.log')).open('a')
        command=[PYTHON,'-m','eventwm_graph.stage2.'+module,'--checkpoint',str(checkpoint),'--name',name]
        print(json.dumps(dict(command=command,source=str(source))),flush=True)
        jobs.append(subprocess.Popen(command,cwd='/tmp',env=env,stdout=log,stderr=subprocess.STDOUT))
    for job in jobs:
        if job.wait()!=0:raise RuntimeError('Optimized execution evaluation failed; retained all evidence, no new model fit.')
    # Both complete original and optimized evaluations are retained. Compare
    # every scientific summary; measured wall times are deliberately separate.
    reference=OUT/'evaluation/final_dev/metrics.json'
    while not reference.exists():time.sleep(5)
    old_result=json.loads(reference.read_text());new_result=json.loads((OUT/'evaluation/optimized_dev/metrics.json').read_text())
    fields=['roots','scenes','counts','pooled','pooled_confusion','scene_balanced','substantial_information',
            'multiclass_logloss','multiclass_brier','persist_unknown_brier','calibration','height_mae_on_observed_cells','height_support','graph_update_correctness']
    comparison={field:old_result[field]==new_result[field] for field in fields}
    (dest/'scientific_summary_comparison.json').write_text(json.dumps(dict(status='PASS' if all(comparison.values()) else 'REVIEW_REQUIRED',
        exact_equal_fields=comparison,reference_sha256=sha(reference),optimized_sha256=sha(OUT/'evaluation/optimized_dev/metrics.json'),
        original_checkpoint_sha256=old_result['checkpoint_sha256'],optimized_checkpoint_sha256=new_result['checkpoint_sha256']),indent=2)+'\n')
    if not all(comparison.values()):raise RuntimeError('Full-cohort scientific output difference requires independent review; no silent substitution.')
    (dest/'COMPLETE.json').write_text(json.dumps(dict(**plan,scientific_summaries_bitwise_equal=True))+'\n')


if __name__=='__main__':main()
