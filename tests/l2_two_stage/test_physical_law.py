"""Physical law checks use actual legal features and exact analytic identities."""
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.stats import poisson
from eventwm_graph.l2_two_stage.common import OUT
from eventwm_graph.l2_two_stage.count import certain_sources,sample_conditioned,count_log_probability
from eventwm_graph.l2_two_stage.data import load_records,collate
from eventwm_graph.l2_two_stage.model import Stage1WorldModel
from eventwm_graph.l2_two_stage.neural import HumanGraphPredictor
from eventwm_graph.l2_two_stage.speed import bound_numpy,inverse_numpy,bound_torch
from eventwm_graph.l2_two_stage.population import add_batch,set_nll


def test_unique_source_is_in_sampler_and_exact_count_law():
    age=np.array([1.]);logits=np.array([-100.]);certain=certain_sources(age,1,unique=True)
    present,counts,normalizer=sample_conditioned(logits,age,.6,1,1000,np.random.default_rng(12),certain)
    assert present.all() and normalizer==0.
    for n in range(1,12):
        np.testing.assert_allclose(count_log_probability(logits,age,.6,n,1,certain),poisson.logpmf(n-1,.6),atol=1e-12)
    assert count_log_probability(logits,age,.6,0,1,certain)==-np.inf


def test_smooth_speed_pushforward_inverse_and_jacobian():
    raw=np.array([[.3,-1.2],[8.,2.],[-.2,.7]])
    bounded=bound_numpy(raw,3.)
    np.testing.assert_allclose(inverse_numpy(bounded,3.),raw,atol=1e-12)
    for x,y in zip(raw,bounded):
        tensor=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        jacobian=torch.autograd.functional.jacobian(lambda z:bound_torch(z,3.),tensor).detach().numpy()
        inverse_determinant=(1-(y*y).sum()/9.)**-2
        np.testing.assert_allclose(np.linalg.det(jacobian)*inverse_determinant,1.,atol=1e-12)


def test_rejected_clique_identity_assumption_cannot_construct_public_model():
    with pytest.raises(ValueError,match='greedy tracker keys'):
        HumanGraphPredictor(compulsory_observed_clique=True,max_speed=3.)


def test_real_tracker_reassociation_counterexample_is_not_censored():
    # Reviewer A's valid fitting-validation example, not an invented identity.
    matches=[r for r in load_records('train_pilot_native_v2') if r['root_id'].startswith('eeb53d2974') and r['anchor_index']==3]
    assert len(matches)==1
    r=matches[0];assert len(r['keys'])==r['minimum_count']==3 and r['set_valid']==1.
    b=collate(matches);add_batch(matches,b,'cpu')
    network=HumanGraphPredictor(graph_channels=14,independent_initial=True,max_speed=3.)
    p=network(b['x'],b['graph'],b['mask'],b['global_features'],b['global_graph'])
    # The rejected source-identity assumption excludes a real native target.
    invalid=set_nll(p,b,condition_minimum=True,max_speed=3.,compulsory_observed_clique=True)
    assert torch.isinf(invalid).all()
    # The retained uncertain-association law continues to score the same case.
    supported=set_nll(p,b,condition_minimum=True,max_speed=3.,compulsory_unique_episode=True)
    assert torch.isfinite(supported).all()


def test_real_feature_bounded_future_speed_and_unique_persistence():
    record=next(r for r in load_records('train_pilot_native_v2') if len(r['keys'])==1 and r['age'][0]>.75)
    f={key:record[key] for key in ('keys','x','graph','global_graph','origin','velocity','age','global_features','robotxy','timestamp','minimum_count')}
    predictor=Stage1WorldModel(OUT/'experiments/pilot_g_persistent_velocity/best.pt')
    predictor.network.max_speed=3.;predictor.network.compulsory_unique_episode=True
    result=predictor.predict_features(f,worlds=32,seed=817)
    assert result.present[:,0].all()
    velocity=np.diff(result.positions,axis=1)/np.diff(result.times)[None,:,None,None]
    assert np.linalg.norm(velocity,axis=-1).max()<=3.+2e-4
    last=f['origin'][0]-f['velocity'][0]*f['age'][0]
    assert (np.linalg.norm(result.positions[:,:,0]-last,axis=-1)<=3*(f['age'][0]+result.times)[None]+2e-4).all()
    assert result.law.max_speed==3. and result.law.compulsory_unique_episode
    np.testing.assert_allclose(result.law.count_log_probability(1),-result.poisson_rate,atol=1e-12)
