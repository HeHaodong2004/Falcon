"""Optional H0 calibration retains the full law and positive-history branch."""
from pathlib import Path
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import poisson
import torch

from eventwm_graph.l2_two_stage.common import OUT
from eventwm_graph.l2_two_stage.fit_empty_count import exact_h0_log_probabilities,affine_objective
from eventwm_graph.l2_two_stage.neural import HumanGraphPredictor
from eventwm_graph.l2_two_stage.tilted_count import numpy_count_log_probability


def models():
    saved=torch.load(OUT/'experiments/pilot_s_count_likelihood32/best.pt',map_location='cpu')
    old=HumanGraphPredictor(**saved['model_config']).eval()
    old.load_state_dict(saved['model'])
    new=HumanGraphPredictor(**dict(saved['model_config'],empty_history_count_affine=True)).eval()
    missing=new.load_state_dict(saved['model'],strict=False)
    assert set(missing.missing_keys)=={'empty_count_affine.weight','empty_count_affine.bias'} and not missing.unexpected_keys
    return old,new,saved


def inputs(context):
    count=len(context);rng=np.random.default_rng(73)
    return [torch.as_tensor(rng.normal(size=shape),dtype=torch.float32) for shape in
        ((count,1,28),(count,1,41,14))]+[torch.zeros(count,1),torch.zeros(count,12),torch.zeros(count,41,14)]


def test_actual_checkpoint_zero_initialization_and_positive_history_no_source_counterexample():
    torch.set_num_threads(1);old,new,saved=models()
    context=torch.tensor([[0.,0.,0.],[.1,0.,0.],[.2,.1,.3]])
    args=inputs(context)
    with torch.no_grad():
        before=old(*args,count_context=context);neutral=new(*args,count_context=context)
        for key in before:assert torch.equal(before[key],neutral[key]),key
        new.empty_count_affine.bias.copy_(torch.tensor([.2,-.7,1.,2.,-1.,.4,.5]))
        changed=new(*args,count_context=context)
    for key in before:
        if key!='total_count_tilt_logits':assert torch.equal(before[key],changed[key]),key
    assert torch.equal(before['total_count_tilt_logits'][1:],changed['total_count_tilt_logits'][1:])
    expected=torch.tensor([.2,-.7,1.,2.,-1.,.4,.5,0.])
    torch.testing.assert_close(changed['total_count_tilt_logits'][0]-before['total_count_tilt_logits'][0],expected)
    for key,value in saved['model'].items():assert torch.equal(new.state_dict()[key],value),key


def test_empty_gate_refuses_contradictory_actual_current_or_proposal_evidence():
    _,model,_=models()
    for value in ([0.,.1,0.],[0.,0.,.1]):
        ctx=torch.tensor([value])
        with pytest.raises(ValueError,match='empty actual history'):
            model(*inputs(ctx),count_context=ctx)


@pytest.mark.parametrize('value',[1,None,'yes'])
def test_affine_flag_is_typed_and_requires_the_declared_family(value):
    with pytest.raises(ValueError):HumanGraphPredictor(empty_history_count_affine=value)
    with pytest.raises(ValueError):HumanGraphPredictor(empty_history_count_affine=True)


def test_exact_H0_individual_scores_keep_the_complete_unbounded_tail():
    logits=np.array([[.1,.4,-.5,.2,1.,-.8,.9,-.2]]*4);rate=np.array([.1,1.,5.,100.]);count=np.array([0,3,12,256])
    classes,individual=exact_h0_log_probabilities(logits,rate,count)
    np.testing.assert_allclose(np.exp(classes).sum(1),1.,rtol=0,atol=3e-16)
    for i in range(4):
        p=dict(rate=rate[i],total_count_tilt_logits=logits[i])
        expected=numpy_count_log_probability(p,[],[],int(count[i]))
        assert individual[i]==pytest.approx(expected,abs=2e-13)
        if count[i]>=7:assert individual[i]<classes[i,7]
        assert np.isfinite(numpy_count_log_probability(p,[],[],1000))


def test_affine_gradient_matches_independent_autograd_proper_count_likelihood():
    rng=np.random.default_rng(7);design=np.c_[rng.normal(size=(13,4)),np.ones(13)]
    logits=rng.normal(size=(13,8));rate=np.linspace(.1,3.,13);count=np.arange(13)
    classes,individual=exact_h0_log_probabilities(logits,rate,count)
    weights=rng.uniform(size=13);weights/=weights.sum();x=rng.normal(scale=.1,size=35);penalty=.001
    value,gradient=affine_objective(x,design,classes,individual,count,weights,penalty)
    matrix=torch.tensor(x.reshape(7,5),requires_grad=True)
    delta=torch.cat((torch.tensor(design)@matrix.T,torch.zeros(13,1,dtype=torch.float64)),dim=1)
    norms=torch.logsumexp(torch.tensor(classes)+delta,dim=1)
    actual=torch.tensor(individual)+delta[torch.arange(13),torch.tensor(np.minimum(count,7))]-norms
    loss=-(torch.tensor(weights)*actual).sum()+.5*penalty*matrix.square().sum();loss.backward()
    assert float(loss)==pytest.approx(value,abs=3e-15)
    np.testing.assert_allclose(matrix.grad.numpy().ravel(),gradient,atol=3e-16,rtol=1e-14)
