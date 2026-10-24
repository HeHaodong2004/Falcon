"""Real native examples plus adversarial legal-boundary counterexamples."""
import copy
import inspect
import itertools
from pathlib import Path
from types import MappingProxyType
import numpy as np
import pytest
import torch
from eventwm_graph.l2_two_stage.common import OUT
from eventwm_graph.l2_two_stage.data import jobs_for,read,one_root,native_index,collate
from eventwm_graph.l2_two_stage.features import extract
from eventwm_graph.l2_two_stage.types import LegalContext,LegalHistory,SealedObservation,require_stage1_pass
from eventwm_graph.l2_two_stage.model import Stage1WorldModel
from eventwm_graph.l2_two_stage.neural import HumanGraphPredictor
from eventwm_graph.l2_two_stage.population import assignment_log_likelihood
from eventwm_graph.l2_two_stage.topology import topology_features
from eventwm_graph.l2_two_stage.evaluate import count_log_probability,paired


@pytest.fixture(scope='module')
def legal():
    return read(Path(jobs_for('dev')[1][0])/'legal_root.json.gz')


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(1)
    return Stage1WorldModel(OUT/'experiments/pilot_b_set/best.pt')


def context(legal):
    return LegalContext.from_inputs(legal['history'],legal['history'][-1],legal['partial_graph'],legal['history_actions'])


@pytest.mark.parametrize('field',['oracle_id','ground_truth','privileged_future','navmesh','query','human_num'])
def test_reject_privileged_observation_fields(legal,field):
    l=copy.deepcopy(legal);l['history'][-1][field]=123
    with pytest.raises(ValueError):context(l)


def test_graph_requires_exact_arrived_provenance(legal):
    l=copy.deepcopy(legal);l['partial_graph']['cells'][0]['cell']=[99999,99999]
    with pytest.raises(ValueError):context(l)
    l=copy.deepcopy(legal);l['partial_graph']['cells'][-1]=copy.deepcopy(l['partial_graph']['cells'][0])
    with pytest.raises(ValueError):context(l)
    l=copy.deepcopy(legal);l['partial_graph']['cells'][0]['last_seen']+=100
    with pytest.raises(ValueError):context(l)


def test_native_alignment_never_interpolates_or_extrapolates(legal):
    job=jobs_for('dev')[1];folder=Path(job[0]);records,ledger=one_root(job)
    with np.load(folder/'privileged_future.npz') as z:times=z['timestamps'];truth=z['positions'][...,[0,2]]
    for r in records:
        for j in np.flatnonzero(r['times']>0):
            at=native_index(times,r['timestamp']+float(r['times'][j]),1e-6)
            for i in np.flatnonzero(r['target_mask'][:,j]):np.testing.assert_array_equal(r['target'][i,j],truth[at,r['physical'][i]].astype(np.float32))
    with pytest.raises(ValueError):native_index(times,times[-1]+.001)


def test_current_observation_exactly_once(legal):
    a=context(legal)
    b=LegalContext.from_inputs(legal['history'][:-1],legal['history'][-1],legal['partial_graph'],legal['history_actions'])
    assert len(a.history)==len(b.history)==len(legal['history'])
    for key in ('x','graph','global_graph','origin','velocity','global_features'):
        np.testing.assert_array_equal(extract(a)[key],extract(b)[key])


def test_sealed_input_cannot_bypass_validation_or_alias_mutable_input(legal):
    with pytest.raises(TypeError):SealedObservation({'timestamp':100},frozenset(),frozenset())
    with pytest.raises(TypeError):LegalHistory((),{},0)
    l=copy.deepcopy(legal);index=next(i for i,ob in enumerate(l['history']) if ob['tracks']);track=l['history'][index]['tracks'][0];proxy=MappingProxyType(track)
    l['history'][index]['tracks'][0]=proxy;h=LegalHistory.from_frames(l['history']);before=h.frames[index]['tracks'][0]['position']
    track['position'][0]+=100
    assert h.frames[index]['tracks'][0]['position']==before
    with pytest.raises((TypeError,AttributeError)):h.frames[index]['tracks'][0]['position'][0]=1


def test_raw_typed_inference_equivalence_and_checkpoint_reproduction(legal,model):
    raw=model.predict(legal['history'],legal['history'][-1],legal['partial_graph'],actions=legal['history_actions'],seed=73)
    prefix=LegalHistory.from_frames(legal['history'][:-1])
    typed=model.predict(prefix,legal['history'][-1],legal['partial_graph'],actions=legal['history_actions'],seed=73)
    reload=Stage1WorldModel(OUT/'experiments/pilot_b_set/best.pt').predict(prefix,legal['history'][-1],legal['partial_graph'],actions=legal['history_actions'],seed=73)
    np.testing.assert_array_equal(raw.positions,typed.positions);np.testing.assert_array_equal(raw.present,typed.present)
    np.testing.assert_array_equal(raw.positions,reload.positions)


def test_query_independent_coherent_paths(legal,model):
    assert 'query' not in inspect.signature(model.predict).parameters
    f=extract(context(legal));result=model.predict_features(f,times=[0,.25,.5,1,2,3],seed=19)
    result.validate()
    np.testing.assert_allclose(result.positions[:,1],(result.positions[:,0]+result.positions[:,2])/2,rtol=2e-6,atol=2e-6)
    # Existence is world/person-level and cannot change with sensor viewpoint.
    assert result.present.ndim==2 and result.positions.shape[1]==6
    current=f['age']<1e-7
    assert result.present[:,:len(current)][:,current].all()
    assert result.poisson_rate>0 and any('UNKNOWN_GEOMETRY' in s for s in result.support)


def test_assignment_matches_exhaustive_injective_sum():
    p=np.array([.2,.7,.4]);density=np.array([[.8,.3],[.2,.6],[.1,.9]]);intensity=np.array([.4,.5]);rate=.9
    exact=0.
    # Each source is absent or assigned to one unique target; other targets PPP.
    for choices in itertools.product(range(-1,2),repeat=3):
        assigned=[i for i in choices if i>=0]
        if len(assigned)!=len(set(assigned)):continue
        value=np.exp(-rate)
        for s,t in enumerate(choices):value*=1-p[s] if t<0 else p[s]*density[s,t]
        for t in set(range(2))-set(assigned):value*=intensity[t]
        exact+=value
    actual=assignment_log_likelihood(torch.tensor(np.log(p))[None],torch.tensor(np.log1p(-p))[None],
        torch.tensor(np.log(density))[None,None],torch.tensor(np.log(intensity))[None,None],torch.tensor([rate]))
    assert float(actual.exp())==pytest.approx(exact,rel=1e-7)


def test_zero_intensity_does_not_destroy_assigned_support():
    value=assignment_log_likelihood(torch.tensor([[0.]]),torch.tensor([[-torch.inf]]),
        torch.tensor([[[[0.]]]]),torch.tensor([[[-torch.inf]]]),torch.tensor([0.]))
    assert float(value)==0.


def test_count_law_retains_nonzero_small_absence_probability():
    # sigmoid(50) rounds to1, yet absence is exp(-50), not impossible.
    actual=count_log_probability(np.array([50.]),np.array([1.]),.01,0)
    assert actual==pytest.approx(-50.01)


def test_aggregate_paired_ratio_and_identity_integrity():
    a=[dict(root_id=str(i),scene_id='s'+str(i),anchor=0,physical=0,horizon=.5,path_es=x) for i,x in enumerate([1.,100.])]
    b=[dict(r,path_es=x) for r,x in zip(a,[2.,100.])]
    result=paired(a,b,'path_es',True)
    assert result['mean']==pytest.approx(-1/102)
    with pytest.raises(ValueError):paired(a+a,b,'path_es',True)


def test_actual_connectivity_changes_graph_features():
    nodes={str(i):dict(position=[float(i),0.,0.]) for i in range(4)}
    chain=dict(global_nodes=nodes,global_edges=[dict(source=str(i),target=str(i+1)) for i in range(3)])
    star=dict(global_nodes=nodes,global_edges=[dict(source='0',target=str(i)) for i in range(1,4)])
    points=np.array([[.1,0.],[2.1,0.]])
    assert not np.allclose(topology_features(chain,points),topology_features(star,points))


@pytest.mark.parametrize('decision',['STAGE1_REVISE','INCONCLUSIVE','STAGE1_PASS',None])
def test_stage2_requires_every_scientific_runtime_gate(decision):
    with pytest.raises(RuntimeError):require_stage1_pass(dict(decision=decision,gates={k:dict(status='PASS') for k in 'ABC'}))
