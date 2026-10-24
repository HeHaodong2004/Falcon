"""Evaluate the one final fit with existing protocols, then freeze research use."""
import json
import os
from pathlib import Path
import subprocess
import time

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'output/l2_two_stage_20261023'
DEST=BASE/'output/research_transition_20261024'
REVIEW=BASE/'docs/l2/final_stage1_stage2_20261024/reviews'
PYTHON='/research-user/miniforge3/envs/eventwm-falcon/bin/python'
CHECKPOINT=OUT/'experiments/final_stage1_research_20261024/last.pt'
SPLIT='train_final_collection_20261024'


def main():
    while not (DEST/'STAGE1_FIT_COMPLETE.json').exists():time.sleep(5)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',EVENTWM_TWO_STAGE_BASE=str(BASE),PYTHONPATH=str(BASE)+':'+str(OUT/'deps'),CUDA_VISIBLE_DEVICES='')
    source=Path(subprocess.check_output([PYTHON,'-m','eventwm_graph.l2_two_stage.freeze_source'],cwd=BASE,env=env,text=True).strip())
    (DEST/'inference_source_path.txt').write_text(str(source)+'\n');env['PYTHONPATH']=str(source)+':'+str(OUT/'deps')
    def run(arguments,log):
        print(json.dumps(dict(command=arguments,log=log,source=str(source))),flush=True)
        with (DEST/log).open('a') as stream:subprocess.run(arguments,cwd='/tmp',env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    forecast=OUT/'evaluation/final_stage1_research_20261024/metrics.json'
    if not forecast.exists():
        run([PYTHON,'-m','eventwm_graph.l2_two_stage.evaluate','final_stage1_research_20261024','--split','dev_native_v2',
            '--fitting-split',SPLIT,'--checkpoint',str(CHECKPOINT),'--checkpoint',str(OUT/'experiments/pilot_k_bounded_unique/best.pt'),
            '--joint-set-score','--worlds','40'],'stage1_forecast_evaluate.log')
    current=OUT/'evaluation/final_stage1_current_20261024'
    if not (current/'summary.json').exists():
        run([PYTHON,'-m','eventwm_graph.l2_two_stage.evaluate_current_population','--name',current.name,'--checkpoint',str(CHECKPOINT),
            '--split','dev_native_v2','--history-cache','dev','--event-cache','dev_native_v2','--fitting-split',SPLIT,
            '--fitting-history-cache',SPLIT,'--worlds','40'],'stage1_current_evaluate.log')
    run(['bash',str(REVIEW/'run_final_stage1_reliability.sh'),str(source),str(CHECKPOINT),str(current)],'stage1_reliability.log')
    run(['bash',str(REVIEW/'run_final_stage1_runtime.sh'),str(source),str(CHECKPOINT),'persistence'],'stage1_persistence.log')
    # Wait for bulk CPU extraction to finish before the isolated runtime gate.
    # Supervised GPU training is lightweight on CPU and may continue independently.
    stage2=BASE/'output/stage2_graphdelta_20261024/prepared/train/manifest.json'
    while not stage2.exists():time.sleep(5)
    run(['bash',str(REVIEW/'run_final_stage1_runtime.sh'),str(source),str(CHECKPOINT),'runtime'],'stage1_runtime.log')
    run([PYTHON,str(REVIEW/'audit_final_stage1.py'),'--source',str(source),'--training-source',str(DEST/'training_source'),
        '--checkpoint',str(CHECKPOINT),'--metrics',str(forecast),'--current',str(current/'summary.json')],'stage1_independent_audit.log')
    (DEST/'STAGE1_EVALUATION_COMPLETE.json').write_text(json.dumps(dict(checkpoint=str(CHECKPOINT),inference_source=str(source),
        research_scope='Final research checkpoint. Existing failed scientific gates remain failed; Stage1 development closed. Stage2 primary.'))+'\n')


if __name__=='__main__':main()
