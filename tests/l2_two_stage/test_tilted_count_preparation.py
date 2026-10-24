"""Bitwise pool/draw oracle against the frozen, independently reviewed sampler."""
import gzip
import importlib.util
import itertools
from pathlib import Path
import pickle
import sys

import numpy as np

from eventwm_graph.l2_two_stage.fast_count import PreparedCountSampler
from eventwm_graph.l2_two_stage.tilted_count_sampler import PreparedTiltedCountSampler,_prepare_base_tables

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'output/l2_two_stage_20261023/frozen_sources/0c1f362d4dbb69f31eafca3c7c9fa24220ae21bc7eafedc66eaf6c5727f39d12/eventwm_graph/l2_two_stage/tilted_count_sampler.py'
NAME='eventwm_graph.l2_two_stage._reviewer_c_tilt_a96_reference'
spec=importlib.util.spec_from_file_location(NAME,SOURCE);OLD=importlib.util.module_from_spec(spec)
sys.modules[NAME]=OLD;spec.loader.exec_module(OLD)


def equal_component(a,b):
    assert (a.rate,a.minimum,a.total,a.log_normalizer)==(b.rate,b.minimum,b.total,b.log_normalizer)
    for name in ('suffix','probabilities'):assert getattr(a,name).tobytes()==getattr(b,name).tobytes()
    assert len(a.prefixes)==len(b.prefixes)
    for x,y in zip(a.prefixes,b.prefixes):
        assert (x.lower,x.upper,x.reference_only)==(y.lower,y.upper,y.reference_only)
        assert x.tail_probability==y.tail_probability or np.isnan(x.tail_probability) and np.isnan(y.tail_probability)
        assert x.conditional_cdf.tobytes()==y.conditional_cdf.tobytes()
        assert x.log_pmf.tobytes()==y.log_pmf.tobytes()
    r0=np.random.default_rng(519);r1=np.random.default_rng(519)
    a0,a1=a.sample(97,r0);b0,b1=b.sample(97,r1)
    assert a0.tobytes()==b0.tobytes() and a1.tobytes()==b1.tobytes()
    assert r0.bit_generator.state==r1.bit_generator.state


def test_saved_actual_component_parameters_share_exact_SF_and_PMF_pool():
    path=ROOT/'docs/l2/two_stage_20261023/reviews/mixed_family_integration_untrained_20261024/paired_profile/count_preparation_fixtures.pkl.gz'
    with gzip.open(path,'rb') as source:fixtures=pickle.load(source)
    assert len(fixtures)==102
    for fixture in fixtures:
        p=fixture['parameters'];certain=np.asarray(p['age'])==0
        if p['certain'] is not None:certain|=p['certain']
        actual=_prepare_base_tables(np.asarray(p['logits'],float),np.asarray(p['age'],float),p['rate'],7,6,certain)
        expected=(PreparedCountSampler(p['logits'],p['age'],p['rate'],7,certain=certain),
                  PreparedCountSampler(p['logits'],p['age'],p['rate'],0,certain=certain,total=6))
        for a,b in zip(actual,expected):equal_component(a,b)


def test_synthetic_extremes_and_complete_tilted_rng_match_original():
    cases=0;failures=0
    for rate,n,minimum,shift in itertools.product((1e-200,1e-12,.03,.7,5.,100.,1000.,70000.),(0,3,8),(0,2,6),(0,1)):
        logits=np.linspace(-1.,1.,n);age=np.ones(n)
        if n:age[0]=0.
        values=dict(logits=logits,age=age,rate=rate,minimum=minimum,
                    tilt_logits=[-.7,.1,.6,-.5,.3,1.2,-.1,.2],shift=shift)
        try:old=OLD.PreparedTiltedCountSampler(**values)
        except RuntimeError:
            try:PreparedTiltedCountSampler(**values)
            except RuntimeError:failures+=1;continue
            raise AssertionError('numerically unsupported reference produced candidate worlds')
        current=PreparedTiltedCountSampler(**values)
        assert old.log_normalizer==current.log_normalizer
        assert old.class_totals.tobytes()==current.class_totals.tobytes()
        assert old.class_log_probabilities.tobytes()==current.class_log_probabilities.tobytes()
        for a,b in zip(current.samplers,old.samplers):equal_component(a,b)
        r0=np.random.default_rng(731);r1=np.random.default_rng(731)
        a,b=current.sample(137,r0);c,d=old.sample(137,r1)
        assert a.tobytes()==c.tobytes() and b.tobytes()==d.tobytes()
        assert r0.bit_generator.state==r1.bit_generator.state
        cases+=1
    assert cases>100 and failures>0
