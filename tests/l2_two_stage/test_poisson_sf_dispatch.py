"""Exact installed Poisson SF backend and unchanged count-family support."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import scipy
from scipy.stats import poisson
from scipy.stats._distn_infrastructure import rv_discrete,rv_generic

from eventwm_graph.l2_two_stage import tilted_count_sampler as sampler


ROOT=Path(__file__).resolve().parents[2]
ORACLE=ROOT/'docs/l2/two_stage_20261023/reviews/poisson_dispatch_readonly_20261024/result.json'
OLD_PATH=ROOT/'output/l2_two_stage_20261023/frozen_sources/2982829029cdcb66b72338960a92ac0aef0bb7bd129b084c1d5e0fe125cc78f5/eventwm_graph/l2_two_stage/tilted_count_sampler.py'
NAME='eventwm_graph.l2_two_stage._reviewer_c_sf_298_reference'
spec=importlib.util.spec_from_file_location(NAME,OLD_PATH);OLD=importlib.util.module_from_spec(spec)
sys.modules[NAME]=OLD;spec.loader.exec_module(OLD)


def test_installed_backend_is_bound_to_the_5234078_decision_oracle():
    assert hashlib.sha256(ORACLE.read_bytes()).hexdigest()=='0dc08ed410c4d26c50af48fcdafb9a32f9b932e83a1f11f750296b738dd2f2d9'
    oracle=json.loads(ORACLE.read_text())
    assert oracle['total_decisions']==5234078 and oracle['issues']==[]
    assert scipy.__version__==oracle['scipy']=='1.13.1'
    for label,owner,name in [('poisson_sf',type(poisson),'_sf'),('generic_logsf',rv_generic,'_logsf'),
                             ('public_discrete_logsf',rv_discrete,'logsf')]:
        method=getattr(owner,name);source=oracle['sources'][label]
        assert inspect.getsource(method)==source['source']
        assert hashlib.sha256(Path(inspect.getfile(method)).read_bytes()).hexdigest()==source['sha256']


@pytest.mark.parametrize('rate',[1e-200,1e-12,.03,.11536016315221786,.7,1.9608168601989746,5.,100.,1000.])
def test_actual_constructed_grid_keeps_negative_support_underflow_and_all_integer_values(rate):
    captured=[]
    original=sampler._prepare_prefix
    def prefix(value,lower,shared=None):
        captured.append(shared[0].copy())
        return original(value,lower,shared)
    with patch.object(sampler,'_prepare_prefix',prefix):
        current=sampler._prepare_base_tables(np.empty(0),np.empty(0),rate,1,0,np.empty(0,bool))
    assert len(captured)==1
    grid=captured[0]
    expected=poisson.logsf(np.arange(-1,len(grid)-1),rate)
    assert grid.tobytes()==expected.tobytes()
    assert grid[0].view(np.uint64)==0 # positive0.0, not log(pdtrc(-1,rate))
    old=OLD._prepare_base_tables(np.empty(0),np.empty(0),rate,1,0,np.empty(0,bool))
    for a,b in zip(current,old):
        assert a.log_normalizer==b.log_normalizer
        assert a.suffix.tobytes()==b.suffix.tobytes()
        assert a.probabilities.tobytes()==b.probabilities.tobytes()


def test_full_unbounded_overflow_allocation_and_post_rng_are_unchanged():
    values=dict(logits=[-.5,.3],age=[1e-9,1.],rate=100.,minimum=0,
                tilt_logits=[-.7,.1,.6,-.5,.3,1.2,-.1,.2],shift=1)
    a=sampler.PreparedTiltedCountSampler(**values);b=OLD.PreparedTiltedCountSampler(**values)
    r0=np.random.default_rng(731);r1=np.random.default_rng(731)
    p0,u0=a.sample(1024,r0);p1,u1=b.sample(1024,r1)
    assert p0.tobytes()==p1.tobytes() and u0.tobytes()==u1.tobytes()
    assert r0.bit_generator.state==r1.bit_generator.state
    assert np.any(u0>7) and np.max(u0)>100
    # A fixed physical remainder total still cancels the tilt; the existing
    # fixed-total branch and its allocation RNG are untouched.
    x=a.recondition(0,1,total=256);y=b.recondition(0,1,total=256)
    q0,n0=x.sample(41,r0);q1,n1=y.sample(41,r1)
    assert q0.tobytes()==q1.tobytes() and n0.tobytes()==n1.tobytes()
    assert np.all(q0.sum(1)+n0==256) and r0.bit_generator.state==r1.bit_generator.state


@pytest.mark.parametrize('rate',[0.,-1.,np.inf,np.nan])
def test_existing_positive_rate_validation_is_unchanged(rate):
    for constructor in (sampler.PreparedTiltedCountSampler,OLD.PreparedTiltedCountSampler):
        with pytest.raises(ValueError):constructor([],[],rate,0,np.zeros(8))


def test_large_grid_keeps_existing_reference_fallback():
    # upper>65536 must continue through existing constructors, not a truncated
    # direct grid or a special-function approximation.
    values=dict(logits=[],age=[],rate=70000.,minimum=1,tilt_logits=np.arange(8)/10)
    a=sampler.PreparedTiltedCountSampler(**values);b=OLD.PreparedTiltedCountSampler(**values)
    r0=np.random.default_rng(49);r1=np.random.default_rng(49)
    p0,u0=a.sample(41,r0);p1,u1=b.sample(41,r1)
    assert p0.tobytes()==p1.tobytes() and u0.tobytes()==u1.tobytes()
    assert r0.bit_generator.state==r1.bit_generator.state
