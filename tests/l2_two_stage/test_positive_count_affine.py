"""The optional source-pooled correction changes only H+ count potentials."""
import numpy as np
import pytest
import torch

from eventwm_graph.l2_two_stage.common import OUT
from eventwm_graph.l2_two_stage.neural import HumanGraphPredictor


def fixture(mode='pooled'):
    torch.set_num_threads(1)
    saved=torch.load(OUT/'experiments/pilot_t_empty_history_affine/best.pt',map_location='cpu')
    old=HumanGraphPredictor(**saved['model_config']).eval();old.load_state_dict(saved['model'])
    model=HumanGraphPredictor(**dict(saved['model_config'],positive_history_count_affine=mode)).eval()
    status=model.load_state_dict(saved['model'],strict=False)
    assert not status.unexpected_keys and set(status.missing_keys)=={'positive_count_affine.weight','positive_count_affine.bias'}
    rng=np.random.default_rng(31)
    args=[torch.tensor(rng.normal(size=shape),dtype=torch.float32) for shape in ((3,3,28),(3,3,41,14))]
    args += [torch.tensor([[0.,0.,0.],[1.,1.,0.],[0.,0.,0.]]),torch.zeros(3,12),torch.zeros(3,41,14)]
    context=torch.tensor([[0.,0.,0.],[.1,0.,.2],[.1,0.,0.]])
    return saved,old,model,args,context


@pytest.mark.parametrize('mode',['pooled','intercept'])
def test_actual_T_zero_initialization_and_H0_preservation(mode):
    saved,old,model,args,context=fixture(mode)
    with torch.no_grad():
        reference=old(*args,count_context=context);neutral=model(*args,count_context=context)
        for key in reference:assert torch.equal(reference[key],neutral[key]),key
        model.positive_count_affine.bias.copy_(torch.arange(7,dtype=torch.float32)/3)
        changed=model(*args,count_context=context)
    for key in reference:
        if key!='total_count_tilt_logits':assert torch.equal(changed[key],reference[key]),key
    assert torch.equal(changed['total_count_tilt_logits'][0],reference['total_count_tilt_logits'][0])
    for row in (1,2):
        torch.testing.assert_close(changed['total_count_tilt_logits'][row]-reference['total_count_tilt_logits'][row],
            torch.cat((torch.arange(7,dtype=torch.float32)/3,torch.zeros(1))))
    for key,value in saved['model'].items():assert torch.equal(model.state_dict()[key],value),key


def test_pooled_correction_is_source_order_invariant_and_ignores_masked_padding():
    _,_,model,args,context=fixture()
    with torch.no_grad():
        model.positive_count_affine.weight.fill_(.04)
        before=model(*args,count_context=context)['total_count_tilt_logits']
        order=torch.tensor([1,0,2]);reordered=[args[0][:,order],args[1][:,order],args[2][:,order],*args[3:]]
        permuted=model(*reordered,count_context=context)['total_count_tilt_logits']
        torch.testing.assert_close(before,permuted,rtol=0,atol=2e-7)
        padded=[x.clone() for x in args];padded[0][:,2]=10000.;padded[1][:,2]=-10000.
        after=model(*padded,count_context=context)['total_count_tilt_logits']
        assert torch.equal(before,after)


@pytest.mark.parametrize('mode',['pooled','intercept'])
def test_source_features_can_condition_the_correction_with_a_matched_intercept_control(mode):
    _,_,model,args,context=fixture(mode)
    with torch.no_grad():
        model.positive_count_affine.weight[3,:48].fill_(.2)
        before=model(*args,count_context=context)['total_count_tilt_logits']
        changed=[x.clone() for x in args];changed[0][1,:2,:4]+=3.
        after=model(*changed,count_context=context)['total_count_tilt_logits']
    if mode=='intercept':assert torch.equal(before,after)
    else:assert abs(float(before[1,3]-after[1,3]))>1e-5
    # A real positive history with no retained proposal sources is H+, not H0.
    assert torch.equal(before[2],after[2])


@pytest.mark.parametrize('mode',[True,False,'automatic','counts'])
def test_unsupported_positive_history_modes_fail_closed(mode):
    with pytest.raises(ValueError):HumanGraphPredictor(total_count_tilt_cutoff=7,positive_history_count_affine=mode)
    with pytest.raises(ValueError):HumanGraphPredictor(positive_history_count_affine='pooled')
