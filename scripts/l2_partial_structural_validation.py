"""Reproduce structural oracles and preserved tests, without fitting or truth.

Only synthetic inputs, existing source, and preserved diagnostic metadata are
used. Every subprocess inherits one CPU-thread and idle-priority limits.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(REPO/'tests/l2_partial_structural'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rss():
    return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))


def test_commands(out):
    commands=[['-m','unittest','discover','-s','tests/l2_partial','-v'],
              ['-m','unittest','discover','-s','tests/l2_partial_structural','-v']]
    commands.extend([['-m','eventwm_graph.l2_partial.'+module,'selftest'] for module in
        ('repair_events','repair_evaluation','repair_oracle_association','repair_risk','repair_fresh','repair_observation_evaluation')])
    commands.append(['scripts/l2_partial_scene_calibration.py','--selftest'])
    records=[]
    for i,args in enumerate(commands):
        started=time.perf_counter()
        result=subprocess.run([sys.executable]+args,cwd=REPO,text=True,capture_output=True)
        log=out.parent/(out.stem+'_tests_'+str(i)+'.log')
        log.write_text(result.stdout+result.stderr)
        counts=re.findall(r'Ran (\d+) tests?',result.stdout+result.stderr)
        records.append({'command':[sys.executable]+args,'returncode':result.returncode,
            'seconds':time.perf_counter()-started,'tests':int(counts[-1]) if counts else None,
            'log':str(log.relative_to(REPO)),'log_sha256':sha(log)})
        print(json.dumps({'test_group':i,'returncode':result.returncode,'tests':records[-1]['tests']}),flush=True)
        if result.returncode:raise RuntimeError('test command failed; preserved '+str(log))
    return records


def oracles():
    import numpy as np
    from dataclasses import replace
    from eventwm_graph.l2_partial.structural_model import StructuralGraphBeliefWorldModel
    from eventwm_graph.l2_partial.structural_state import StructuralBeliefState
    from eventwm_graph.l2_partial.state import BeliefState,Hypothesis,ObservedGraph
    from eventwm_graph.l2_partial.structural_probability import canonical_impulse_velocity_variance
    from eventwm_graph.l2_partial.structural_budget import InferenceBudgetExceeded
    from test_rollout_properties import model_belief,query
    from scripts.l2_partial_repair_systems_stress import fixture

    compositions=[]
    for start in (0.,.013,1.23456789):
        m=StructuralGraphBeliefWorldModel(kind='structured',population_prior=[1.])
        h=Hypothesis(1.,np.array([[1.,.5,.4,-.1]]),np.eye(4)[None]*.15,np.array([start]),np.array([0]),np.array([0.,0.,1.]),
            intent=np.array([[.2,.1]]),diffusion=np.array([.25]),mode=np.array([0]),relaxation=.7,height=np.array([0.]),
            hidden_mean=np.array([[4.,0.,0.,0.]]),hidden_covariance=np.diag([100.,100.,4.,4.])[None])
        b=BeliefState(ObservedGraph('analytic-oracle',0,.2,0.,{}),[h],start,np.zeros(4),model_version=m.model_version)
        direct=m.propagate(b,start+3.)
        for cuts in ((1.5,3.),(1.,2.,3.)):
            other=b
            for cut in cuts:other=m.propagate(other,start+cut)
            errors={}
            for name in ('mean','covariance','hidden_mean','hidden_covariance','maneuver_mean','maneuver_covariance','maneuver_probability','hidden_maneuver_mean','hidden_maneuver_covariance'):
                errors[name]=max(float(np.max(np.abs(getattr(a,name)-getattr(e,name)))) for a,e in zip(direct.hypotheses,other.hypotheses))
            if max(errors.values())>1e-9:raise AssertionError('analytic composition '+repr(errors))
            compositions.append({'absolute_start':start,'cuts':cuts,'maximum_absolute_errors':errors,
                'mass_error':abs(sum(h.weight for h in other.hypotheses)+other.unresolved_mass-1)})

    m,b=model_belief(n=1,hidden=0);f=m.predict_shared(b,.144,num_worlds=16,seed=41)
    direct=m.evaluate_queries(b,[query()],f,budget_ms=10000.).predictions[0]
    no=lambda p:next(br for br in p.observation_branches if br.observation['kind']=='survived_no_detection')
    d=no(direct)
    a=no(m.evaluate_queries(b,[query(actions=1)],f,budget_ms=10000.).predictions[0])
    nextf=m.predict_shared(a.successor,.048,num_worlds=16,seed=999)
    c=no(m.evaluate_queries(a.successor,[query(start=.2,actions=1,name='q2')],nextf,budget_ms=10000.).predictions[0])
    reconstruction=sum(br.probability*br.successor.carrier.weights for br in direct.observation_branches)
    coarse={'seed':41,'worlds':16,'optical_regime_detection_probabilities':[.2,.8],
        'direct_no_detection':d.probability,'segmented_joint_no_detection':a.probability*c.probability,
        'probability_absolute_error':abs(d.probability-a.probability*c.probability),
        'conditional_weights_max_error':float(np.max(np.abs(d.successor.carrier.weights-c.successor.carrier.weights))),
        'finite_measure_marginalization_max_error':float(np.max(np.abs(reconstruction-f.weights))),
        'scope':'finite retained path law, zero unresolved mass; no continuous-refinement claim'}

    variance=[]
    for horizon in (.5,1.,2.,3.):
        m=StructuralGraphBeliefWorldModel(kind='structured',population_prior=[1.])
        h=Hypothesis(1.,np.zeros((1,4)),np.zeros((1,4,4)),np.zeros(1),np.array([0]),np.array([1.]),
            intent=np.zeros((1,2)),diffusion=np.array([.25]),mode=np.array([0]),relaxation=.7,height=np.zeros(1))
        b=BeliefState(ObservedGraph('variance-oracle',0,.2,0.,{}),[h],0.,np.zeros(4),model_version=m.model_version)
        p=m.propagate(b,horizon);v=float(p.hypotheses[0].covariance[0,2,2])
        grid=int(horizon/.048);closed=float(canonical_impulse_velocity_variance(0.,grid))
        variance.append({'horizon_seconds':horizon,'analytic_velocity_variance_per_axis':v,
            'full_canonical_intervals':grid,'full_interval_closed_form_variance':closed,
            'gaussian_moment_speed_over_3':float(np.exp(-9/(2*v))) if v else 0.,
            'gaussian_moment_speed_over_5':float(np.exp(-25/(2*v))) if v else 0.,
            'tail_scope':'isotropic Gaussian moment approximation, not exact jump-mixture tail or native validation'})

    budget_rows=[];m=StructuralGraphBeliefWorldModel()
    for n,certificate in ((6,False),(6,True)):
        b,row,qs=fixture(m,n,certificate)
        m.online_call(b,row,qs)
        fingerprint=b.fingerprint()
        for budget in (0.,1.,10.,65.,100.):
            for repeat in range(3):
                began=time.perf_counter();status='COMPLETE';stage=None;pending=None
                try:m.online_call(b,row,qs,budget_ms=budget)
                except InferenceBudgetExceeded as exc:status=exc.status;stage=exc.stage;pending=exc.pending_observation
                elapsed=(time.perf_counter()-began)*1000.
                if fingerprint!=b.fingerprint():raise AssertionError('budget call mutated its input')
                budget_rows.append({'tracked':n,'certificate':certificate,'budget_ms':budget,
                    'elapsed_ms':elapsed,'overshoot_ms':max(0.,elapsed-budget),'status':status,'stage':stage,
                    'pending_observation':pending,'input_unchanged':True})

    m,b=model_belief(n=1,hidden=0);b=StructuralBeliefState.wrap(b);history=[]
    started=time.perf_counter()
    for i in range(1,501):
        t=i*.048;tracks=[]
        if i%5==0:tracks=[{'visible_now':True,'valid':True,'last_seen_time':t,'position':[2.+.2*t,0.,.5]}]
        row={'timestamp':t,'robot_pose':[0,0,0,0],'static_updates':[], 'tracks':tracks}
        b=m.observe(b,row)
        if i%50==0:
            b.validate()
            history.append({'step':i,'timestamp':t,'rss_kib':rss(),'unresolved_mass':b.unresolved_mass,
                'human_ledger_entries':len(b.evidence.get('human_evidence_ledger',{})),
                'hypotheses':len(b.hypotheses),'resolved_count_values':sorted({h.n+int(k) for h in b.hypotheses if h.weight>0 for k in np.flatnonzero(h.hidden_pmf>0)})})
    if any(row['resolved_count_values']!=[1] for row in history):raise AssertionError('long session count changed')
    restore=StructuralBeliefState.from_dict(json.loads(json.dumps(b.to_dict(),allow_nan=False)))
    if restore.fingerprint()!=b.fingerprint():raise AssertionError('persistent restore mismatch')
    return {'analytic_composition':compositions,'coarse_observation_composition':coarse,
        'physical_variance_law':variance,'cooperative_budgets':budget_rows,
        'persistent_session':{'steps':500,'duration_seconds':24.,'observations_every_step':True,
            'positive_every_five_steps':True,'seconds':time.perf_counter()-started,'samples':history,'restore_fingerprint_equal':True}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--skip-tests',action='store_true')
    args=parser.parse_args();out=(REPO/args.output).resolve()
    if not out.is_relative_to(REPO/'output/l2_partial_graph_wm_structural_20261021') or out.exists():raise ValueError('new structural output required')
    out.parent.mkdir(exist_ok=True,parents=True)
    os.setpriority(os.PRIO_PROCESS,0,19);subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True)
    sources={str(p.relative_to(REPO)):sha(p) for p in sorted((REPO/'eventwm_graph/l2_partial').glob('*.py'))}
    started=time.perf_counter();tests=[];diagnostics=None;failure=None
    try:
        if not args.skip_tests:tests=test_commands(out)
        diagnostics=oracles()
    except Exception as exc:failure=type(exc).__name__+': '+str(exc)
    value={'schema':'structural_validation_1','source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'script_sha256':sha(__file__),'source_sha256':sources,'sources_unchanged':all(sha(REPO/p)==v for p,v in sources.items()),
        'tests':tests,'tests_total':sum(x['tests'] or 0 for x in tests),'diagnostics':diagnostics,
        'failure':failure,'seconds':time.perf_counter()-started,'statistical_fitting':False,'new_truth_payloads_opened':False}
    out.write_text(json.dumps(value,indent=2,allow_nan=False))
    print(json.dumps({'output':str(out),'tests':value['tests_total'],'failure':failure,'sources_unchanged':value['sources_unchanged']}))
    if failure:raise RuntimeError(failure)


if __name__=='__main__':main()
