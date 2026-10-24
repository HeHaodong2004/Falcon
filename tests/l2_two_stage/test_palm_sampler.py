"""Independent analytic population law with a legally mandatory residual person."""
import math
import numpy as np
from scipy.special import digamma
from scipy.stats import poisson
from eventwm_graph.l2_two_stage.association import AssociationGraph
from eventwm_graph.l2_two_stage.conditioned import draw_conditioned_current,pad_current
from eventwm_graph.l2_two_stage.positive_history import PositiveSetHistory
from eventwm_graph.l2_two_stage.palm_sampler import required_rectangle
from eventwm_graph.l2_two_stage.vector_sampler import PreparedCurrentSampler


def fixture():
    # Two anonymous positive observations100m apart cannot be the same person.
    # The current source is pinned at100m; a residual person must witness x=0.
    h=PositiveSetHistory('','','',1,np.array([0.,1.]),np.array([0,1,2]),
        np.array([[0.,0.,0.],[100.,0.,0.]]),np.ones(2,bool),np.ones(2,bool))
    g=AssociationGraph.from_history(h)
    f=dict(keys=['anonymous'],timestamp=1.,age=np.array([0.]),velocity=np.zeros((1,2)),
        minimum_count=1,source_last_event_indices=np.array([1]),robotxy=np.zeros(2))
    p=dict(mode_logits=np.zeros(4),presence_logits=np.zeros(1),rate=1.2,
        means=np.zeros((1,4,5,2)),scales=np.ones((1,4,5,2)),
        spatial_logits=np.zeros(1),spatial_mean=np.zeros((1,2)),spatial_scale=np.full((1,2),5.))
    return h,g,f,p


def test_exact_palm_matches_analytic_unbounded_count_and_normalizer():
    h,g,f,p=fixture();rng=np.random.default_rng(20261023)
    rows,events,stats=draw_conditioned_current(f,p,h,g,16000,rng)
    assert stats['proposal_count_offset']==1
    positions,present=pad_current(rows,1)
    assert g.feasible(positions,present).all()
    # Analytic thinning: U~Poisson(lambda) conditioned on at least one atom
    # inside the old observation's3m reachable disk. No count truncation.
    disk=1-math.exp(-9/50);rate=p['rate'];normalizer=1-math.exp(-rate*disk)
    counts=np.array([len(row.hidden_positions) for row in rows])
    for u in range(1,8):
        expected=poisson.pmf(u,rate)*(1-(1-disk)**u)/normalizer
        assert abs(np.mean(counts==u)-expected)<.015
    logz=digamma(len(rows))-digamma(stats['proposals_examined_through_last_accept'])+stats['log_normalizer_multiplier']
    assert abs(logz-math.log(normalizer/(1-math.exp(-rate))))<.025
    # The distinguished point cannot stay in a special latent slot: it must be
    # uniformly shuffled with original PPP marks before associations/motion.
    many=[row for row in rows if len(row.hidden_positions)==2]
    first=np.mean([np.linalg.norm(row.hidden_positions[0])<=3 for row in many])
    last=np.mean([np.linalg.norm(row.hidden_positions[-1])<=3 for row in many])
    assert abs(first-last)<.04


def test_certificate_never_uses_a_residual_necessity_when_a_source_can_cover_it():
    h,g,f,p=fixture()
    assert required_rectangle(g,np.array([[100.,0.]]),np.array([0.])).event==0
    # A legal source support disk overlapping the historic witness invalidates
    # this optimization; it does not invalidate that feasible model world.
    assert required_rectangle(g,np.array([[0.,0.],[100.,0.]]),np.array([1.,0.])) is None


def test_no_certificate_preserves_original_sampler_and_random_stream():
    h,g,f,p=fixture();f=dict(f,age=np.array([100.]))
    raw=np.array([[100.,0.]])
    a=PreparedCurrentSampler(f,p,raw,2,graph=g)
    b=PreparedCurrentSampler(f,p,raw,2)
    assert a.count_offset==0
    aa=a.draw(128,np.random.default_rng(18));bb=b.draw(128,np.random.default_rng(18))
    for key in aa.__dataclass_fields__:np.testing.assert_array_equal(getattr(aa,key),getattr(bb,key))
