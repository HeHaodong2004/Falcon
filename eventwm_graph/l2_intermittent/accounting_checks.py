"""Finite public-scoring integration checks; no data/model fitting or forecasts.

Uses the independently authored Reviewer B deterministic old/new source-order
fixtures and compares every score field against the exact executed evaluator.
"""
import ast
import collections
import copy
import importlib.util
import json
import math
from types import SimpleNamespace as NS
import numpy as np
from scipy.spatial.distance import cdist
from .common import BASE,OUT,DOC,sha,dump,truth_at,centers,age_bin,AGE_BINS


def load_helpers(path,fixtures):
    tree=ast.parse(path.read_bytes())
    names={'energy','extend_ancestry','lineage_anchor','lineage_slot','to_previous','score'}
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    constants=[n for n in tree.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id in ('PATH_TIMES','SCORE_TIMES') for t in n.targets)]
    env=dict(np=np,collections=collections,math=math,cdist=cdist,HORIZONS=(.5,1.,2.,3.),
        Hypothesis=fixtures.H,StructuralBeliefState=fixtures.State,seed_for=lambda root,stream:stream,
        truth_at=truth_at,centers=centers,age_bin=age_bin,AGE_BINS=AGE_BINS)
    exec(compile(ast.Module(body=constants+selected,type_ignores=[]),str(path),'exec'),env)
    return env


def run():
    target=OUT/'offline_accounting_integration.json'
    if target.exists():raise ValueError('preserve integration evidence')
    reference=OUT/'primary_agent_notes/corrected_science_runtime/eventwm_graph/l2_intermittent/evaluate.py'
    if not reference.exists():reference=DOC/'artifacts/verified_execution/sources/eventwm_graph/l2_intermittent/evaluate.py'
    if sha(reference)!='a278381a264b564eb51683469c89d307a41268805292ac856cec611434f3c184':raise ValueError('executed reference differs')
    public=BASE/'eventwm_graph/l2_intermittent/evaluate.py'
    fixture_path=DOC/'reviews/reviewer_b_final_owned_evaluator_valid_birth_checks.py'
    spec=importlib.util.spec_from_file_location('_independent_b_fixtures',fixture_path)
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    sources={str(p.relative_to(BASE)):sha(p) for p in (reference,public,fixture_path)}
    a=load_helpers(public,fixtures);b=load_helpers(reference,fixtures)
    ts=np.array([.5,1.,4.]);truth=np.tile(np.array([[[0.,0.],[10.,0.]]]),(3,1,1))
    anchors={(4,8):(0,.3,.5),(4,7):(0,0.,1.),(9,7):(0,0.,.5),(9,8):(0,.3,1.)}
    belief=NS(timestamp=1.,components=[fixtures.component(4),fixtures.component(9)],
        graph=NS(height=0.),robot_pose=np.zeros(4))
    cases=[('permuted_old_new_sources',belief,np.tile([1,0],16))]
    missing=copy.deepcopy(belief);missing.components[0]=fixtures.component(4,slots=(7,))
    cases.append(('positive_unsampled_missing_preferred',missing,np.ones(32,int)))
    recycled=copy.deepcopy(belief);recycled.components[0]=fixtures.component(4,slots=(8,))
    recycled.components[0].birth[:]=.6;cases.append(('recycled_preferred_birth',recycled,np.tile([1,0],16)))
    records=[]
    metadata=dict(root_id='deterministic_fixture',scene_id='fixture',model='dummy',cohort='native',decision='root')
    for name,state,indices in cases:
        for old in (False,True):
            rows=[]
            for env in (a,b):
                model=fixtures.DummyModel(state,indices,env['SCORE_TIMES'],old)
                rows.append(env['score'](model,state,anchors,[0],{0:.5},ts,truth,metadata,41,old))
            if rows[0]!=rows[1]:raise AssertionError('score/source-order difference: '+name+str(old))
            records.append(dict(case=name,old_baseline=old,rows=len(rows[0]),all_fields_equal=True,
                statuses=[r['status'] for r in rows[0]],represented_missing_mass=[r.get('represented_missing_mass') for r in rows[0]]))
    first=fixtures.component(4,slots=(7,));first.last_position[:]=[0.,0.]
    alternative=fixtures.component(4,slots=(7,9));alternative.birth=np.array([0.,1.])
    alternative.last_seen=np.array([.5,1.]);alternative.last_position=np.array([[10.,0.],[0.,0.]])
    other_a=fixtures.component(4);other_a.last_seen=np.array([1.,.5]);other_a.last_position=np.array([[0.,0.],[10.,0.]])
    other_b=fixtures.component(4);other_b.last_seen=np.array([.5,1.]);other_b.last_position=np.array([[10.,0.],[0.,0.]])
    constructions=[('divergent_first_forks',[first,alternative],{}),
        ('first_fork_to_already_tagged_other_person',[other_a,other_b],{(4,8):(1,.3,.5)}),
        ('common_genuine_first',[first,copy.deepcopy(first)],{})]
    for name,cs,initial in constructions:
        tables=[];selected=[]
        for env in (a,b):
            table=dict(initial);env['extend_ancestry'](NS(timestamp=1.,components=cs),table,ts,truth)
            tables.append(table);selected.append([env['lineage_slot'](c,table,0) for c in cs])
        if tables[0]!=tables[1] or selected[0]!=selected[1]:raise AssertionError('first-ancestry difference: '+name)
        records.append(dict(case=name,ancestry_equal=True,selected=selected[0]))
    tie={(4,7):(0,0.,.5),(4,8):(0,.3,.5)}
    if a['lineage_slot'](belief.components[0],tie,0) is not None:raise AssertionError('malformed simultaneous anchor accepted')
    text=public.read_text()
    for required in ('ancestry_before={k:dict(v) for k,v in ancestry.items()}',
        'ancestry_before[k]',"payload_files_hash_verified_per_root=5",'inputs_stable=',
        "else 'reappearance' if reappeared else 'positive' if len(centers(ob)) else 'available_empty'"):
        if required not in text:raise AssertionError('public guard/readout lost: '+required)
    result=dict(passed=True,source_stable=all(sha(BASE/p)==value for p,value in sources.items()),sources=sources,
        public_evaluator_sha256=sha(public),executed_reference_sha256=sha(reference),records=records,
        malformed_simultaneous_anchor_refused=True,pre_arrival_gain_ancestry_preserved=True,
        five_payload_and_complete_input_guards_preserved=True,raw_positive_categories_preserved=True,
        scope='Post-run PUBLIC offline integration; exact old/new fixed-input score equality and common-first lifetime checks. Executed science sources/results, runtime/checkpoints/versions remain unchanged; no new data/model forecast occurs.')
    dump(target,result);print(json.dumps({k:v for k,v in result.items() if k not in ('records','sources')}))
    if not result['source_stable']:raise SystemExit(1)


if __name__=='__main__':run()
