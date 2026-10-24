"""Exact current-cover training certificate, gradients and legacy RNG parity."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from eventwm_graph.l2_two_stage import conditioned_training as training
from eventwm_graph.l2_two_stage.association import AssociationGraph
from eventwm_graph.l2_two_stage.common import OUT
from eventwm_graph.l2_two_stage.conditioned import draw_conditioned_current
from eventwm_graph.l2_two_stage.data import load_records
from eventwm_graph.l2_two_stage.future_population_targets import attach_targets
from eventwm_graph.l2_two_stage.history_proposal import augment_torch
from eventwm_graph.l2_two_stage.last_event_metadata import attach_metadata
from eventwm_graph.l2_two_stage.mixed_count import legal_context
from eventwm_graph.l2_two_stage.neural import HumanGraphPredictor
from eventwm_graph.l2_two_stage.population import set_nll
from eventwm_graph.l2_two_stage.positive_history import PositiveSetHistory,attach_histories
from eventwm_graph.l2_two_stage.tilted_count import LOGITS


OLD_PATH=OUT/'frozen_sources/88584eed7d2173e6e3ab31c7cfc1ed2a908a603734a123cc86f34caf4434c92a/eventwm_graph/l2_two_stage/conditioned_training.py'
NAME='eventwm_graph.l2_two_stage._reviewer_c_current_885_reference'
spec=importlib.util.spec_from_file_location(NAME,OLD_PATH);OLD=importlib.util.module_from_spec(spec)
sys.modules[NAME]=OLD;spec.loader.exec_module(OLD)


def synthetic(times,frames,valid=None,available=None):
    points=[np.asarray(frame,np.float64).reshape(-1,3) for frame in frames]
    h=PositiveSetHistory('test','test','test',len(times)-1,np.array(times),np.r_[0,np.cumsum([len(p) for p in points])],
        np.concatenate(points),np.ones(len(times),bool) if valid is None else valid,
        np.ones(len(times),bool) if available is None else available)
    events=np.arange(len(h.positive_xyz));age=h.timestamps[-1]-np.repeat(h.timestamps,np.diff(h.positive_offsets))
    r=dict(keys=tuple('anonymous'+str(i) for i in events),timestamp=h.timestamps[-1],age=age,
        origin=h.positive_xyz[:,[0,2]].astype(np.float32),velocity=np.zeros((len(events),2),np.float32),
        robotxy=np.zeros(2,np.float32),minimum_count=h.minimum_count,positive_history=h,
        source_last_event_indices=events)
    return r,AssociationGraph.from_history(h)


def problem(records,hidden=None):
    b=len(records);n=max(1,max(len(r['keys']) for r in records));j=3
    p=dict(means=torch.zeros((b,n,4,5,2),dtype=torch.float64),scales=torch.full((b,n,4,5,2),.2,dtype=torch.float64),
        presence_logits=torch.zeros((b,n),dtype=torch.float64),mode_logits=torch.zeros((b,4),dtype=torch.float64),
        rate=torch.ones(b,dtype=torch.float64),spatial_mean=torch.zeros((b,j,2),dtype=torch.float64),
        spatial_scale=torch.ones((b,j,2),dtype=torch.float64),
        spatial_logits=torch.tensor([[0.,-np.inf,-np.inf]]*b,dtype=torch.float64),
        total_count_tilt_logits=torch.tensor([[-.2,.3,-.7,.1,.8,-.4,.2,-.1]]*b,dtype=torch.float64,requires_grad=True))
    origin=np.zeros((b,n,2));velocity=origin.copy();age=np.zeros((b,n));mask=np.zeros((b,n))
    for i,r in enumerate(records):
        m=len(r['keys']);origin[i,:m]=r['origin'];velocity[i,:m]=r['velocity'];age[i,:m]=r['age'];mask[i,:m]=1
    target=np.zeros((b,1,2));target_mask=np.zeros((b,1))
    for i,point in (hidden or {}).items():target[i,0]=point;target_mask[i,0]=1
    batch={k:torch.as_tensor(v) for k,v in dict(x=np.zeros((b,n,1)),origin=origin,velocity=velocity,age=age,mask=mask,
        robotxy=np.zeros((b,2)),minimum_count=np.array([r['minimum_count'] for r in records]),
        set_positions=target,set_mask=target_mask,set_valid=np.ones(b),weight=np.linspace(.4,.9,b)).items()}
    return p,batch


def clone(p):
    q={key:value.detach().clone() for key,value in p.items()};q[LOGITS].requires_grad_(True);return q


def canonical_batch(batch,records,graphs):
    last=np.zeros(tuple(batch['origin'].shape));minimum=[]
    for i,(r,g) in enumerate(zip(records,graphs)):
        _,points,d=training.current_cover_certificate(r,g);last[i,:len(points)]=points
        minimum.append(max(r['minimum_count'],d))
    return dict(batch,source_last_positions=torch.tensor(last),minimum_count=torch.tensor(minimum),
                origin=batch['origin'].double(),age=batch['age'].double())


@pytest.mark.parametrize('colocated',[False,True])
def test_actual_current_multiset_covers_full_history_and_exact_gradient_without_sampling(monkeypatch,colocated):
    points=[[0.,0.,0.],[0. if colocated else 1.,0.,0.]]
    r,g=synthetic([0.,1.],[points,points]);assert g.minimum_current_count==2
    assert training.current_cover_certificate(r,g)[0]
    p,b=problem([r]);q=clone(p)
    def forbidden(*a,**k):raise AssertionError('covered row drew latent worlds')
    monkeypatch.setattr(training,'sample_latents',forbidden)
    rng=np.random.default_rng(73);state=copy.deepcopy(rng.bit_generator.state)
    score,terms=training.loss(p,b,[r],[g],rng,current_likelihood_only=True,analytic_current_cover=True)
    direct=(set_nll(q,canonical_batch(b,[r],[g]),True,True,3.)*b['weight']).mean()
    assert torch.equal(score,direct)
    assert torch.equal(torch.autograd.grad(score,p[LOGITS])[0],torch.autograd.grad(direct,q[LOGITS])[0])
    assert terms['analytic_current_cover'].tolist()==[1.] and terms['proposals'].tolist()==[0]
    assert terms['normalizer_log_estimate'].item()==0. and rng.bit_generator.state==state


def test_H0_exact_normalizer_does_not_force_empty_current_population(monkeypatch):
    r,g=synthetic([0.,1.],[[],[]]);p,b=problem([r],{0:[.3,.1]})
    with monkeypatch.context() as patch:
        patch.setattr(training,'sample_latents',lambda *a,**k:(_ for _ in ()).throw(AssertionError('H0 draws')))
        score,terms=training.loss(p,b,[r],[g],np.random.default_rng(1),current_likelihood_only=True,analytic_current_cover=True)
    assert torch.isfinite(score) and terms['analytic_current_cover'].tolist()==[1.]
    parameters={k:v.detach().numpy()[0] for k,v in p.items()}
    rows,_,_=draw_conditioned_current(r,parameters,r['positive_history'],g,128,np.random.default_rng(41))
    counts=np.array([len(row.hidden_positions) for row in rows])
    assert np.any(counts==0) and np.any(counts>0) # Unobserved people remain uncertain.


@pytest.mark.parametrize('availability',['stale','invalid','unavailable'])
def test_missing_current_evidence_with_past_positives_remains_MC(monkeypatch,availability):
    r,g=synthetic([0.,1.],[[[0.,0.,0.]],[]],valid=[True,availability!='invalid'],
                  available=[True,availability!='unavailable'])
    assert not training.current_cover_certificate(r,g)[0]
    p,b=problem([r],{0:[0.,0.]});calls=[];original=training.sample_latents
    def capture(*args,**kwargs):calls.append(kwargs);return original(*args,**kwargs)
    monkeypatch.setattr(training,'sample_latents',capture)
    score,terms=training.loss(p,b,[r],[g],np.random.default_rng(17),current_likelihood_only=True,analytic_current_cover=True)
    assert torch.isfinite(score) and len(calls)==1 and calls[0]['include_lineage'] is False
    assert terms['analytic_current_cover'].tolist()==[0.] and terms['proposals'][0]>0


def test_full_DAG_stronger_than_observed_count_and_tiny_stale_rounding_never_shortcut(monkeypatch):
    r,g=synthetic([0.,1.],[[[0.,0.,0.]],[[10.,0.,0.]]])
    assert r['minimum_count']==1 and g.minimum_current_count==2
    assert not training.current_cover_certificate(r,g)[0] # c>=weakL still fails fulld==c.
    tiny,graph=synthetic([0.,5e-17],[[[1.,0.,0.]],[]]);tiny['velocity'][0,0]=4.
    assert not training.current_cover_certificate(tiny,graph)[0]
    current=np.float64(1.)+np.float64(5e-17)*np.float64(2.4)
    assert current-1.>3*5e-17 # Actual unique-stale floating counterexample.
    p,b=problem([tiny],{0:[1.,0.]})
    def must_sample(*a,**k):raise RuntimeError('UNCERTIFIED_MC_REQUIRED')
    monkeypatch.setattr(training,'sample_latents',must_sample)
    with pytest.raises(RuntimeError,match='UNCERTIFIED_MC_REQUIRED'):
        training.loss(p,b,[tiny],[graph],np.random.default_rng(13),current_likelihood_only=True,analytic_current_cover=True)


@pytest.mark.parametrize('bad',['clock','graph','source_ref','source_age'])
def test_certificate_rejects_false_clock_graph_or_current_source_binding(bad):
    r,g=synthetic([0.,1.],[[[0.,0.,0.]],[[0.,0.,0.]]]);r=copy.deepcopy(r)
    if bad=='clock':r['timestamp']+=.1
    elif bad=='graph':_,g=synthetic([0.,1.],[[[1.,0.,0.]],[[1.,0.,0.]]])
    elif bad=='source_ref':r['source_last_event_indices'][:]=0
    else:r['age'][0]=0.
    with pytest.raises(ValueError):training.current_cover_certificate(r,g)


def test_certificate_never_reads_privileged_targets():
    r,g=synthetic([0.,1.],[[[0.,0.,0.]],[[0.,0.,0.]]])
    class Forbidden:
        def __array__(self,*a,**k):raise AssertionError('privileged population entered certificate')
    r['population_current_positions']=Forbidden();r['population_future_positions']=Forbidden()
    assert training.current_cover_certificate(r,g)[0]


def test_mixed_batch_keeps_all_padding_and_original_weighted_mean(monkeypatch):
    covered,g0=synthetic([0.,1.],[[[0.,0.,0.],[1.,0.,0.]],[[0.,0.,0.],[1.,0.,0.]]])
    stale,g1=synthetic([0.,1.],[[[0.,0.,0.]],[]]);records=[covered,stale];graphs=[g0,g1]
    p,b=problem(records,{1:[0.,0.]});q=clone(p);called=[];original=training.sample_latents
    def capture(rows,parameters,states,*args,**kwargs):
        assert rows==[stale] and parameters['means'].shape[:2]==(1,4)
        assert parameters['spatial_logits'].shape==(1,3) and torch.isneginf(parameters['spatial_logits'][0,1:]).all()
        called.append(len(rows));return original(rows,parameters,states,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(training,'sample_latents',capture)
        a=np.random.default_rng(19)
        score,terms=training.loss(p,b,records,graphs,a,current_likelihood_only=True,analytic_current_cover=True)
    assert called==[1] and terms['analytic_current_cover'].tolist()==[1.,0.]
    index=torch.tensor([1]);subp=training._leading_subset(q,index,2);subbatch=training._leading_subset(b,index,2)
    subbatch['weight']=torch.ones(1,dtype=torch.float64);c=np.random.default_rng(19)
    previous,_=OLD.loss(subp,subbatch,[stale],[g1],c,current_likelihood_only=True)
    first=set_nll(q,canonical_batch(b,records,graphs),True,True,3.)[0]
    direct=(first*b['weight'][0]+previous*b['weight'][1])/2
    torch.testing.assert_close(score,direct,rtol=0,atol=2e-13)
    torch.testing.assert_close(torch.autograd.grad(score,p[LOGITS])[0],torch.autograd.grad(direct,q[LOGITS])[0],rtol=0,atol=2e-13)
    assert a.bit_generator.state==c.bit_generator.state


@pytest.fixture(scope='module')
def actual():
    saved=json.loads((OUT/'mixed_count_optimizer_smoke_89d5.json').read_text())
    wanted={(r['root'],r['anchor']) for r in saved['rows']}
    records=[r for r in load_records('train_pilot_native_v2') if (r['root_id'],r['anchor_index']) in wanted]
    assert len(records)==3
    attach_histories(records,'train_pilot_conditioned');attach_metadata(records,'train_pilot_conditioned');attach_targets(records,'train_campaign_20261023')
    checkpoint=torch.load(OUT/'experiments/pilot_t_empty_history_affine/best.pt',map_location='cpu')
    model=HumanGraphPredictor(**checkpoint['model_config']).eval();model.load_state_dict(checkpoint['model']);model.requires_grad_(False)
    batch=training.make_batch(records,'cpu');graphs=[AssociationGraph.from_history(r['positive_history']) for r in records]
    ctx=torch.tensor(np.array([legal_context(r,r['positive_history'],g) for r,g in zip(records,graphs)]))
    parameters=model(batch['x'],batch['graph'],batch['mask'],batch['global_features'],batch['global_graph'],count_context=ctx)
    parameters=augment_torch(parameters,records,graphs,.5)
    return records,batch,graphs,{k:v.detach().double() for k,v in parameters.items()}


def test_defaultFalse_actual_scalar_terms_gradient_and_rng_bitwise885(actual):
    records,batch,graphs,parameters=actual;p=clone(parameters);q=clone(parameters)
    a=np.random.default_rng(413);b=np.random.default_rng(413)
    first,terms=training.loss(p,batch,records,graphs,a,current_likelihood_only=True,analytic_current_cover=False)
    second,other=OLD.loss(q,batch,records,graphs,b,current_likelihood_only=True)
    assert torch.equal(first,second)
    assert torch.equal(torch.autograd.grad(first,p[LOGITS])[0],torch.autograd.grad(second,q[LOGITS])[0])
    assert a.bit_generator.state==b.bit_generator.state and terms.keys()==other.keys()
    for key in terms:
        if torch.is_tensor(terms[key]):assert torch.equal(terms[key],other[key])
        else:assert np.array_equal(terms[key],other[key])


def test_real_mixed_control_only_draws_uncertified_rows(actual,monkeypatch):
    records,batch,graphs,parameters=actual;covered=[training.current_cover_certificate(r,g)[0] for r,g in zip(records,graphs)]
    calls=[];original=training.sample_latents
    def capture(rows,*a,**k):calls.append(rows);return original(rows,*a,**k)
    monkeypatch.setattr(training,'sample_latents',capture)
    score,terms=training.loss(clone(parameters),batch,records,graphs,np.random.default_rng(413),
        current_likelihood_only=True,analytic_current_cover=True)
    assert torch.isfinite(score) and terms['analytic_current_cover'].tolist()==list(map(float,covered))
    assert [r for rows in calls for r in rows]==[r for r,yes in zip(records,covered) if not yes]


def test_cover_option_keeps_supported_target_and_frozen_base_guards(actual,monkeypatch):
    records,batch,graphs,parameters=actual
    monkeypatch.setattr(training,'sample_latents',lambda *a,**k:(_ for _ in ()).throw(AssertionError('sampling before guards')))
    bad=dict(batch,set_valid=torch.zeros(len(records)))
    with pytest.raises(ValueError):training.loss(clone(parameters),bad,records,graphs,np.random.default_rng(1),
        current_likelihood_only=True,analytic_current_cover=True)
    p=clone(parameters);p['rate'].requires_grad_(True)
    with pytest.raises(ValueError):training.loss(p,batch,records,graphs,np.random.default_rng(1),
        current_likelihood_only=True,analytic_current_cover=True)
    for flag in (True,1,None):
        with pytest.raises(ValueError):training.loss(clone(parameters),batch,records,graphs,np.random.default_rng(1),analytic_current_cover=flag)
