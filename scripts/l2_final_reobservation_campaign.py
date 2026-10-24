"""Finish the prespecified additional public-API return cohort for A/B/C."""
import gzip,json,os,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from eventwm_graph.l2_final.common import OUT,PREVIOUS,dump,sha
from eventwm_graph.l2_scaling_500k.compare import ci,pair
FIELDS=('prior_future_path_es','posterior_future_path_es','future_path_es_gain','prior_path_es','posterior_path_es','path_es_gain','association_accuracy','false_split','false_merge','duplicate','missing_antecedent_mass','prior_coverage50','posterior_coverage50','prior_coverage90','posterior_coverage90')

def summarize():
    base=OUT/'reobservation_supplement';manifest=json.loads((base/'manifest.json').read_text());data={};result={}
    for arm in ('A','B','C'):
        folder=base/('evaluation_'+arm)
        with gzip.open(folder/'records.jsonl.gz','rt') as f:data[arm]=[json.loads(line) for line in f]
        cfg=json.loads((folder/'configuration.json').read_text());cells=[]
        assert len(data[arm])==manifest['roots']*8
        for scope in ('all_0.5_to_2','practical_0.75_to_1.25'):
            for horizon in (.5,1.,2.,3.):
                part=[r for r in data[arm] if r['H']==horizon and (scope=='all_0.5_to_2' or .75-1e-8<=r['gap']<=1.25+1e-8)]
                from collections import Counter
                cells.append(dict(scope=scope,H=horizon,status=dict(Counter(r['prior_status']+'/'+r['posterior_status'] for r in part)),metrics={key:ci(part,key) for key in FIELDS}))
        counts=[]
        for row in data[arm]:
            if row['H']!=.5 or row['stream']!=41 or 'prior_count' not in row:continue
            r={k:row[k] for k in ('root_id','scene_id')}
            for state in ('prior','posterior'):
                for key,value in row[state+'_count'].items():r[state+'_'+key]=value
            counts.append(r)
        result[arm]=dict(roots=cfg['roots'],scenes=cfg['scenes'],failures=cfg['failures'],cells=cells,
            fallback_roots=sum(bool(v['fallback']) for v in cfg['initialization']),count_metrics={key:ci(counts,key) for key in ('prior_count_brier','posterior_count_brier','prior_count_entropy','posterior_count_entropy','prior_true_count_probability','posterior_true_count_probability')},
            prior_zero_count_support_roots=sum(not r['prior_true_count_support'] for r in counts),posterior_zero_count_support_roots=sum(not r['posterior_true_count_support'] for r in counts),
            true_count_nll='INFINITE if any zero-support root; no floor',count_diagnostic_roots=len(counts),
            max_count_normalization_error=max((r[state+'_normalization_error'] for r in counts for state in ('prior','posterior')),default=None),configuration_sha256=sha(folder/'configuration.json'))
    contrasts={}
    for left,right in (('A','B'),('A','C'),('B','C')):
        paired,ledger=pair(data[left],data[right],'reobservation',FIELDS);cells=[]
        for scope in ('all_0.5_to_2','practical_0.75_to_1.25'):
            for horizon in (.5,1.,2.,3.):
                part=[r for r in paired if r['H']==horizon and (scope=='all_0.5_to_2' or .75-1e-8<=r['gap']<=1.25+1e-8)]
                cells.append(dict(scope=scope,H=horizon,metrics={key:ci(part,key) for key in FIELDS}))
        contrasts[right+'_minus_'+left]=dict(case_ledger=ledger,cells=cells)
    dump(OUT/'reobservation_supplement_metrics.json',dict(variants=result,comparisons=contrasts,manifest_sha256=sha(base/'manifest.json'),
        independent_unit='one native physical future/root; two forecast streams are dependent repetitions; equal-root/equal-scene bootstrap seed73',
        population='prespecified new development cases enriched for a later native physical reappearance, not unconditioned calibration or protected confirmation'))

def main():
    while not ((OUT/'reobservation_supplement/manifest.json').exists() and (OUT/'evaluation_campaign_complete.json').exists()):time.sleep(1)
    checkpoints=dict(A=PREVIOUS/'training/checkpoint.pt',B=PREVIOUS/'training/checkpoint.pt',C=OUT/'training/checkpoint.pt');workers=[]
    for arm in ('A','B','C'):
        env=os.environ.copy();runtime=OUT/'benchmark_runtime_A' if arm=='A' else BASE
        env['PYTHONPATH']=str(runtime);env['L2_REPRO_OUTPUT']=str(runtime/'output/l2_graph_wm_120k_20261021')
        log=(OUT/'logs'/('reobservation_supplement_'+arm+'.log')).open('x')
        command=[sys.executable,'-u',str(BASE/'scripts/l2_final_reobservation_score.py'),'--arm',arm,'--checkpoint',str(checkpoints[arm])]
        workers.append((subprocess.Popen(command,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT),log))
    for p,log in workers:
        code=p.wait();log.close()
        if code:raise RuntimeError('additional return evaluation failed; preserve all outputs')
    summarize()
if __name__=='__main__':main()
