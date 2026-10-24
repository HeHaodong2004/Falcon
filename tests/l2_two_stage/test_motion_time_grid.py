"""Actual native forecast reuse and worst-case knot interpolation counterexample."""
from types import SimpleNamespace
from pathlib import Path
import os
import numpy as np
import pytest
import torch
from unittest.mock import patch
from eventwm_graph.l2_two_stage.time_grid import prediction_grid,interpolate_positions
from eventwm_graph.l2_two_stage.conditioned import HistoryConditionedWorldModel
from eventwm_graph.l2_two_stage.context import OnlineContext
from eventwm_graph.l2_two_stage.features import extract
from eventwm_graph.l2_two_stage.data import jobs_for,read,selected_branch
from eventwm_graph.l2_two_stage.common import OUT


def test_grid_keeps_size_endpoints_knots_including_near_endpoint():
    for horizon in (.01,.5,np.nextafter(.5,np.inf),.75,1.,1.9,2.,3.):
        times=prediction_grid(horizon)
        assert times.shape==(64,) and times[0]==0 and times[-1]==horizon
        assert np.all(np.diff(times)>0)
        for knot in (.5,1.,2.):
            if knot<horizon:assert knot in times


def test_linear_reuse_cannot_hide_the_old_point071m_kink_error():
    def path(t):return np.stack((2.999*t-5.998*np.maximum(t-.5,0),np.zeros_like(t)),-1)[None,:,None]
    old=np.linspace(0,3,64);oldbank=SimpleNamespace(times=old,positions=path(old))
    with pytest.raises(ValueError,match='omits'):interpolate_positions(oldbank,[.5])
    estimate=np.interp(.5,old,oldbank.positions[0,:,0,0])
    assert abs(estimate-path(np.array([.5]))[0,0,0,0])>.071
    times=prediction_grid(3.);bank=SimpleNamespace(times=times,positions=path(times))
    np.testing.assert_allclose(interpolate_positions(bank,[.49,.5,.51]),path(np.array([.49,.5,.51])),atol=1e-14,rtol=0)
    with pytest.raises(ValueError,match='extrapolation'):interpolate_positions(bank,[3.001])


def test_real_public_bank_reuses_same_worlds_at_arbitrary_times_without_inference():
    torch.set_num_threads(1)
    folder=Path(jobs_for('dev')[0][0]);legal=read(folder/'legal_root.json.gz')
    context=OnlineContext.from_raw(legal['history'],legal['partial_graph'],legal['history_actions']).with_association()
    checkpoint=Path(os.environ.get('EVENTWM_STAGE1_TEST_CHECKPOINT',str(OUT/'experiments/full_held_o_guided_campaign/best.pt')))
    model=HistoryConditionedWorldModel(checkpoint)
    bank=model.predict(context,context.history[-1],seed=731)
    features=extract(context);features['positive_history']=context._positive_history
    requested=np.array([0.,.001,.123,.5,.6,1.01,1.5,2.17,3.])
    exact=model.predict_features(features,times=requested,seed=731,association_graph=context._association_graph)
    np.testing.assert_array_equal(bank.present,exact.present)
    np.testing.assert_array_equal(bank.association.event_person,exact.association.event_person)
    np.testing.assert_allclose(bank.positions_at(requested),exact.positions,atol=2e-13,rtol=0)
    assert not bank.positions_at(requested).flags.writeable
    # No model object participates when downstream callers reuse the bank.
    del model
    np.testing.assert_allclose(bank.positions_at(requested),exact.positions,atol=2e-13,rtol=0)


def test_mixed_requests_copy_exact_endpoints_and_knots_without_ulp_regrouping():
    # All segment speeds <=0.3m/s. Recomputing -.2+(.1-(-.2)) yields
    # .10000000000000003, while the actual stored endpoint is exactly .1.
    times=np.array([0.,.5,1.,2.,3.])
    positions=np.array([[-.5,0.],[-.4,0.],[-.3,0.],[-.2,0.],[.1,0.]])[None,:,None]
    bank=SimpleNamespace(times=times,positions=positions)
    exact=interpolate_positions(bank,[3.,.5,0.,3.])
    mixed=interpolate_positions(bank,[3.,.25,.5,0.,3.,1.75])
    np.testing.assert_array_equal(mixed[:,[0,2,3,4]],exact)
    assert mixed[:,[0,2,3,4]].tobytes()==exact.tobytes()
    assert not mixed.flags.writeable
    with pytest.raises(ValueError):mixed.setflags(write=True)


def test_native_bank_exact_requests_work_but_missing_knot_offgrid_refuses():
    times=np.array([0.,.048,.104,.152,.208,.504,1.008,2.016,3.024])
    values=np.stack((times*.25,times*-.5),axis=-1)[None,:,None]
    bank=SimpleNamespace(times=times,positions=values)
    requested=times[[8,1,0,5,8]]
    np.testing.assert_array_equal(interpolate_positions(bank,requested),values[:,[8,1,0,5,8]])
    with pytest.raises(ValueError,match='omits'):interpolate_positions(bank,[.5])
    # The original diagnostic bank and its requested ordering were not changed.
    np.testing.assert_array_equal(bank.times,times)


@pytest.mark.parametrize('bad_times',[
    np.array([]),np.array([0.]),np.array([0.,1.,.5,2.,3.]),
    np.array([0.,.5,.5,1.,2.,3.]),np.array([0.,.5,np.nan,2.,3.]),
    np.array([0.,.5,1.,2.,np.inf]),np.array([0.,.5+1j,1.,2.,3.]),
])
def test_malformed_stored_grids_fail_even_for_exact_or_empty_requests(bad_times):
    bank=SimpleNamespace(times=bad_times,positions=np.zeros((2,len(bad_times),1,2)))
    with pytest.raises(ValueError):interpolate_positions(bank,[0.])
    with pytest.raises(ValueError):interpolate_positions(bank,[])


@pytest.mark.parametrize('requested',[
    np.array([.5+1.j]),np.array([[.5]]),np.array([np.nan]),np.array([np.inf]),
    np.array([-np.finfo(float).eps]),np.array([np.nextafter(3.,np.inf)]),
])
def test_invalid_requested_times_never_coerce_or_extrapolate(requested):
    times=prediction_grid(3.)
    bank=SimpleNamespace(times=times,positions=np.zeros((2,len(times),1,2)))
    with pytest.raises(ValueError):interpolate_positions(bank,requested)


def test_nonfinite_or_wrong_shape_trajectories_fail_closed():
    times=prediction_grid(3.)
    for value in (np.nan,np.inf,-np.inf):
        positions=np.zeros((2,len(times),1,2));positions[1,4,0,1]=value
        # Reject the invalid full trajectory even if this request misses it.
        with pytest.raises(ValueError):
            interpolate_positions(SimpleNamespace(times=times,positions=positions),[0.])
    for positions in (np.zeros((2,len(times),2)),np.zeros((2,len(times)-1,1,2)),
                      np.zeros((2,len(times),1,3)),np.zeros((2,len(times),1,2),complex)):
        with pytest.raises(ValueError):
            interpolate_positions(SimpleNamespace(times=times,positions=positions),[0.])


def test_unsigned_decreasing_positions_are_promoted_before_interpolation():
    # A physically valid <=2m/s path used to wrap uint8(2)-uint8(3) to255.
    bank=SimpleNamespace(times=np.array([0.,.5,1.,2.,3.]),
        positions=np.array([[3,0],[2,0],[1,0],[1,0],[0,0]],np.uint8)[None,:,None])
    np.testing.assert_array_equal(interpolate_positions(bank,[.25]),[[[[2.5,0.]]]])


def test_minimal_horizon_grids_and_unrepresentable_grids_fail_honestly():
    np.testing.assert_array_equal(prediction_grid(.5,2),[0.,.5])
    np.testing.assert_array_equal(prediction_grid(3.,5),[0.,.5,1.,2.,3.])
    for horizon,count in ((3.,4),(1.,2),(.5,1),(.5,True),(.5,64.)):
        with pytest.raises(ValueError):prediction_grid(horizon,count)
    for horizon in (0.,-.1,np.nan,np.inf,3.1):
        with pytest.raises(ValueError):prediction_grid(horizon)
    #64 distinct positive float64 times cannot fit into the smallest subnormal
    # interval. Refuse it, rather than lowering count or returning duplicates.
    with pytest.raises((ValueError,RuntimeError)):prediction_grid(np.nextafter(0.,1.))


@pytest.fixture(scope='module')
def actual_reuse_cases():
    torch.set_num_threads(1)
    mapping={row['root_id']:Path(folder) for folder,row in jobs_for('dev')}
    selections=[(next(iter(mapping)),0),
        ('4210ef67367c610897b7a406eb87c1061c1ca7e9f4f859911de95737bd3fc844',1),
        ('ffccd1e26a606a0c92cbd17fef7b7043fe25f44d26008e6a4be24f3964debc3b',117)]
    model=HistoryConditionedWorldModel(Path(os.environ.get('EVENTWM_STAGE1_TEST_CHECKPOINT',str(OUT/'experiments/full_held_o_guided_campaign/best.pt'))))
    results=[]
    for root,anchor in selections:
        legal=read(mapping[root]/'legal_root.json.gz');branch=selected_branch(mapping[root])[2]
        context=OnlineContext.from_raw(legal['history'],legal['partial_graph'],legal['history_actions']).with_association()
        for node in branch[:anchor]:context=context.ingest(node['observation'],node['action'])
        bank=model.predict(context,context.history[-1],seed=819,worlds=32,horizon=3.)
        features=extract(context);features['positive_history']=context._positive_history
        native=np.array([0.]+[node['observation']['timestamp']-bank.timestamp for node in branch[anchor:]
                              if 0.<node['observation']['timestamp']-bank.timestamp<=3.])
        assert len(native)>=2
        results.append((model,context,features,bank,native))
    return results


def test_actual_sparse_dense_and_gap_banks_preserve_default_shape_and_latent_samples(actual_reuse_cases):
    for model,context,features,bank,native in actual_reuse_cases:
        assert bank.times.shape==(64,) and bank.positions.shape[:2]==(32,64)
        assert bank.times[0]==0. and bank.times[-1]==3.
        assert all(knot in bank.times for knot in (.5,1.,2.))
        assert not bank.times.flags.writeable and not bank.positions.flags.writeable
        for times in (native,np.unique(np.r_[0.,.123,.499,.5,.501,.999,1.,1.001,1.999,2.,2.001,2.713,3.])):
            direct=model.predict_features(features,times=times,seed=819,worlds=32,
                                          association_graph=context._association_graph)
            np.testing.assert_array_equal(bank.present,direct.present)
            np.testing.assert_array_equal(bank.weights,direct.weights)
            np.testing.assert_array_equal(bank.source_event_indices,direct.source_event_indices)
            np.testing.assert_array_equal(bank.previously_observed,direct.previously_observed)
            assert bank.track_keys==direct.track_keys
            assert bank.association.kernel==direct.association.kernel
            assert dict(bank.sampling)==dict(direct.sampling)
            for field in ('event_person','matching','row_order','first_edge','physical_seen','auxiliary_log_probability'):
                np.testing.assert_array_equal(getattr(bank.association,field),getattr(direct.association,field))
            np.testing.assert_allclose(bank.positions_at(times),direct.positions,atol=2e-13,rtol=0)


def test_actual_request_order_duplicates_readonly_and_no_resampling(actual_reuse_cases):
    for model,context,features,bank,_ in actual_reuse_cases:
        requested=np.array([3.,.123,.5,0.,2.,.123,1.,3.]);before=bank.positions.tobytes()
        sorted_unique=np.unique(requested)
        direct=model.predict_features(features,times=sorted_unique,seed=819,worlds=32,
                                      association_graph=context._association_graph)
        expected=direct.positions[:,np.searchsorted(sorted_unique,requested)]
        with patch.object(model.network,'forward',side_effect=AssertionError('bank reuse called NN')), \
             patch('numpy.random.default_rng',side_effect=AssertionError('bank reuse resampled')):
            reused=bank.positions_at(requested)
        np.testing.assert_allclose(reused,expected,atol=2e-13,rtol=0)
        assert reused[:,[0,2,3,4,6,7]].tobytes()==bank.positions_at([3.,.5,0.,2.,1.,3.]).tobytes()
        assert before==bank.positions.tobytes()
        requested[:]=-999.
        assert not reused.flags.writeable
        with pytest.raises(ValueError):reused.setflags(write=True)
        empty=bank.positions_at([])
        assert empty.shape==(32,0,bank.positions.shape[2],2) and not empty.flags.writeable


def test_public_shorter_horizons_keep_the_same_worlds_and_64_time_slots(actual_reuse_cases):
    for model,context,_,bank,_ in actual_reuse_cases:
        for horizon in (.5,1.,2.):
            shorter=model.predict(context,context.history[-1],seed=819,worlds=32,horizon=horizon)
            assert shorter.times.shape==(64,) and shorter.positions.shape[:2]==(32,64)
            assert shorter.times[0]==0. and shorter.times[-1]==horizon
            np.testing.assert_array_equal(shorter.present,bank.present)
            np.testing.assert_array_equal(shorter.association.event_person,bank.association.event_person)
            np.testing.assert_array_equal(shorter.association.auxiliary_log_probability,bank.association.auxiliary_log_probability)
            np.testing.assert_allclose(bank.positions_at(shorter.times),shorter.positions,atol=2e-13,rtol=0)
