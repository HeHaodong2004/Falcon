"""Exact preparation/draw oracle against pre-optimization component code."""
import gzip
import importlib.util
import itertools
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

from eventwm_graph.l2_two_stage.fast_count import PreparedCountSampler
from eventwm_graph.l2_two_stage.mixture_count_sampler import PreparedMixtureCountSampler,_prepare_pair

ROOT=Path(__file__).resolve().parents[2]
ARTIFACT=ROOT/'docs/l2/two_stage_20261023/reviews/mixed_family_integration_untrained_20261024/paired_profile'
NAME='eventwm_graph.l2_two_stage._reviewer_c_original_mixture_count'
spec=importlib.util.spec_from_file_location(NAME,ARTIFACT/'sources/mixture_count_sampler.py')
REFERENCE=importlib.util.module_from_spec(spec);sys.modules[NAME]=REFERENCE;spec.loader.exec_module(REFERENCE)


def component_equal(a,b):
    assert a.rate==b.rate and a.minimum==b.minimum and a.total==b.total
    assert a.log_normalizer==b.log_normalizer
    for name in ('suffix','probabilities'):assert getattr(a,name).tobytes()==getattr(b,name).tobytes()
    assert len(a.prefixes)==len(b.prefixes)
    for x,y in zip(a.prefixes,b.prefixes):
        assert (x.lower,x.upper,x.reference_only)==(y.lower,y.upper,y.reference_only)
        assert x.tail_probability==y.tail_probability or np.isnan(x.tail_probability) and np.isnan(y.tail_probability)
        assert x.conditional_cdf.tobytes()==y.conditional_cdf.tobytes()
        assert x.log_pmf.tobytes()==y.log_pmf.tobytes()
    r0=np.random.default_rng(731);r1=np.random.default_rng(731)
    a0,a1=a.sample(97,r0);b0,b1=b.sample(97,r1)
    assert a0.tobytes()==b0.tobytes() and a1.tobytes()==b1.tobytes()
    assert r0.bit_generator.state==r1.bit_generator.state


def mixture_equal(a,b):
    assert a.log_normalizer==b.log_normalizer
    for name in ('rates','log_count_probabilities','log_component_normalizers',
                 'component_log_probabilities','active_components'):
        assert getattr(a,name).tobytes()==getattr(b,name).tobytes()
    for j in a.active_components:component_equal(a.components[j],b.components[j])
    r0=np.random.default_rng(413);r1=np.random.default_rng(413)
    a0,a1=a.sample(129,r0);b0,b1=b.sample(129,r1)
    assert a0.tobytes()==b0.tobytes() and a1.tobytes()==b1.tobytes()
    assert r0.bit_generator.state==r1.bit_generator.state


def test_every_saved_real_component_has_bitwise_batched_tables_and_draws():
    with gzip.open(ARTIFACT/'count_preparation_fixtures.pkl.gz','rb') as source:fixtures=pickle.load(source)
    assert len(fixtures)==102
    for fixture in fixtures:
        values=fixture['parameters'];rates=np.array([values['rate'],values['rate']*.71+.1])
        actual=_prepare_pair(values['logits'],values['age'],rates,values['minimum'],values['certain'],values['total'])
        for rate,component in zip(rates,actual):
            reference=PreparedCountSampler(**dict(values,rate=rate))
            component_equal(component,reference)


@pytest.mark.parametrize('rates',[[1e-200,.03],[1e-12,1000.],[.03,.7],[5.,100.],[.7,.7]])
@pytest.mark.parametrize('total',[None,5,12])
def test_mixture_preparation_matches_old_code_including_tails_and_fixed_total(rates,total):
    for minimum in (0,1,3,5):
        values=dict(logits=np.array([-.9,.2,1.5,-2.]),age=np.array([.2,0.,2.,.8]),
                    rates=rates,count_logits=[.7,-.2],minimum=minimum,
                    certain=[False,False,False,True],total=total)
        try:old=REFERENCE.PreparedMixtureCountSampler(**values)
        except RuntimeError:
            with pytest.raises(RuntimeError):PreparedMixtureCountSampler(**values)
            continue
        actual=PreparedMixtureCountSampler(**values)
        mixture_equal(actual,old)


@pytest.mark.parametrize('weights',[[.3,-.1],[0.,-900.],[-np.inf,.2]])
def test_lower_floor_reweighting_is_fresh_old_mixture_bitwise(weights):
    values=dict(logits=[-.8,.3,1.1],age=[1.,0.,.9],rates=[.03,9.],count_logits=[.6,.2],minimum=6)
    prepared=PreparedMixtureCountSampler(**values)
    for minimum in range(7):
        actual=prepared.recondition(minimum,weights)
        expected=REFERENCE.PreparedMixtureCountSampler(**dict(values,minimum=minimum,count_logits=weights))
        mixture_equal(actual,expected)
        for component in actual.components:
            if component is not None:
                with pytest.raises(ValueError):component.suffix.setflags(write=True)


def test_reconditioning_rejects_incompatible_constraints_and_component_activation():
    prepared=PreparedMixtureCountSampler([.2],[1.],[.3,2.],[0.,0.],3)
    for minimum in (-1,4,1.5):
        with pytest.raises(ValueError):prepared.recondition(minimum,[0.,0.])
    for weights in ([0.],[np.nan,0.],[np.inf,0.],[-np.inf,-np.inf]):
        with pytest.raises(ValueError):prepared.recondition(2,weights)
    fixed=PreparedMixtureCountSampler([.2],[1.],[.3,2.],[0.,0.],1,total=3)
    with pytest.raises(ValueError):fixed.recondition(1,[0.,0.])
    inactive=PreparedMixtureCountSampler([.2],[1.],[.3,2.],[0.,-np.inf],1)
    with pytest.raises(ValueError):inactive.recondition(0,[0.,0.])


def test_large_grid_fallback_and_single_active_old_rng_are_unchanged():
    values=dict(logits=[-.5,.4],age=[1.,1.],rates=[1.,70000.],count_logits=[.2,-.3],minimum=3)
    mixture_equal(PreparedMixtureCountSampler(**values),REFERENCE.PreparedMixtureCountSampler(**values))
    values.update(rates=[.03,1.],count_logits=[-np.inf,.7])
    mixture_equal(PreparedMixtureCountSampler(**values),REFERENCE.PreparedMixtureCountSampler(**values))
