"""Differential proof against the frozen pre-optimization world-loop oracle."""
import copy
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from eventwm_graph.l2_two_stage import conditioned
from eventwm_graph.l2_two_stage.common import OUT
from eventwm_graph.l2_two_stage.context import OnlineContext
from eventwm_graph.l2_two_stage.data import jobs_for,read,selected_branch
from eventwm_graph.l2_two_stage.neural import KNOTS
from eventwm_graph.l2_two_stage.speed import bound_numpy
from eventwm_graph.l2_two_stage.time_grid import prediction_grid


def reference_future(features,parameters,rows,times,rng):
    # Exact function body from frozen conditioned.py c8066a2553a48d7c...
    n=len(features['keys']);k=len(rows);current,present=conditioned.pad_current(rows,n)
    mode=np.array([row.mode for row in rows]);initial=np.array([row.initial_coefficients for row in rows]).reshape(k,n,2)
    noise=rng.normal(size=(k,n,4,2));common=rng.normal(size=(k,1,4,2))
    coefficient=parameters['means'][:n][np.arange(n)[None],mode[:,None],1:]+parameters['scales'][:n][np.arange(n)[None],mode[:,None],1:]*(noise*.979795897+common*.2)
    velocity=bound_numpy(features['velocity'][None,:,None]+initial[:,:,None]+coefficient,3.)
    starts=np.array(KNOTS[:-1]);ends=np.array(KNOTS[1:])
    basis=np.minimum(np.maximum(times[:,None]-starts,0),ends-starts);basis[:,-1]=np.maximum(times-starts[-1],0)
    positions=np.zeros((k,len(times),current.shape[1],2),np.float64)
    increments=(basis@velocity.transpose(2,0,1,3).reshape(4,-1)).reshape(len(times),k,n,2).transpose(1,0,2,3)
    positions[:,:,:n]=current[:,None,:n]+increments
    for w,row in enumerate(rows):
        components=row.hidden_components;count=len(components)
        raw_velocity=parameters['spatial_velocity'][components]+parameters['spatial_velocity_scale'][components,None]*rng.normal(size=(count,2))
        positions[w,:,n:n+count]=row.hidden_positions[None]+times[:,None,None]*bound_numpy(raw_velocity,3.)[None]
    return positions,present


def rng_json(rng):
    return json.dumps(rng.bit_generator.state,sort_keys=True,default=lambda value:value.tolist())


@pytest.mark.parametrize('worlds',[32,40])
@pytest.mark.parametrize('sources',[0,3])
@pytest.mark.parametrize('all_empty',[False,True])
@pytest.mark.parametrize('generator',[np.random.PCG64,np.random.MT19937,np.random.Philox])
def test_actual_slot_order_padding_ghosts_and_rng_stream(worlds,sources,all_empty,generator):
    setup=np.random.default_rng(913)
    features=dict(keys=tuple(map(str,range(sources))),velocity=setup.normal(size=(sources,2)))
    parameters=dict(means=setup.normal(size=(sources,4,5,2)),scales=setup.random((sources,4,5,2))+.01,
        spatial_velocity=setup.normal(size=(4,2)),spatial_velocity_scale=setup.random(4)+.01)
    rows=[]
    for w in range(worlds):
        count=0 if all_empty else (0,3,1,0,4,2)[w%6]
        rows.append(conditioned.CurrentProposal(w%4,np.full(sources,w%3!=0,bool),setup.normal(size=(sources,2)),
            setup.integers(0,4,count,dtype=np.int32 if w%2 else np.int64),
            setup.normal(size=(count,2)).astype(np.float32 if w%2 else np.float64),setup.normal(size=(sources,2))))
    times=prediction_grid(3.)
    old_rng=np.random.Generator(generator(73));new_rng=np.random.Generator(generator(73))
    old=reference_future(features,parameters,rows,times,old_rng)
    new=conditioned.construct_future(features,parameters,rows,times,new_rng)
    assert old[0].tobytes()==new[0].tobytes() and old[1].tobytes()==new[1].tobytes()
    assert rng_json(old_rng)==rng_json(new_rng)
    np.testing.assert_array_equal(old_rng.bit_generator.random_raw(128),new_rng.bit_generator.random_raw(128))
    for w,row in enumerate(rows):
        count=len(row.hidden_positions)
        np.testing.assert_array_equal(new[0][w,0,:sources],row.source_positions)
        np.testing.assert_array_equal(new[0][w,0,sources:sources+count],row.hidden_positions)
        assert not new[0][w,:,sources+count:].any() and not new[1][w,sources+count:].any()
        np.testing.assert_array_equal(new[1][w,:sources],row.source_present)
    # Native timing does not change draw count or the actual motion at t=0.
    native=np.array([0.,.048,.504,1.008,2.016,3.024])
    old_rng=np.random.Generator(generator(91));new_rng=np.random.Generator(generator(91))
    old=reference_future(features,parameters,rows,native,old_rng)
    new=conditioned.construct_future(features,parameters,rows,native,new_rng)
    assert old[0].tobytes()==new[0].tobytes() and old[1].tobytes()==new[1].tobytes()
    assert rng_json(old_rng)==rng_json(new_rng)


@pytest.fixture(scope='module')
def native_contexts():
    torch.set_num_threads(1)
    selections=[('ffccd1e26a606a0c92cbd17fef7b7043fe25f44d26008e6a4be24f3964debc3b',117),
        ('e4445c0e408b3377da67ca62919dcbe3ba68d1554adc876c750dd4653beb2c50',41),
        ('b4436f4b2eb9d2b260123cac159611aa893ab882b2f8c53e13646d7bfd7c806b',10)]
    folders={row['root_id']:Path(folder) for folder,row in jobs_for('dev')}
    wanted={root for root,_ in selections}
    for folder,row in jobs_for('train_campaign_20261023'):
        if row['root_id'] in wanted:folders[row['root_id']]=Path(folder)
    results=[]
    for root,anchor in selections:
        legal=read(folders[root]/'legal_root.json.gz');branch=selected_branch(folders[root])[2]
        context=OnlineContext.from_raw(legal['history'],legal['partial_graph'],legal['history_actions']).with_association()
        for node in branch[:anchor-1]:context=context.ingest(node['observation'],node['action'])
        results.append((context,branch[anchor-1]))
    return results


@pytest.mark.parametrize('case',[0,1,2])
@pytest.mark.parametrize('worlds',[32,40])
def test_real_complete_world_ancestry_and_post_future_rng_are_bitwise(native_contexts,case,worlds,monkeypatch):
    checkpoint=Path(os.environ.get('EVENTWM_STAGE1_TEST_CHECKPOINT',str(OUT/'experiments/full_refit_o_guided_campaign/last.pt')))
    model=conditioned.HistoryConditionedWorldModel(checkpoint)
    context,node=native_contexts[case];candidate=conditioned.construct_future;states=[]
    def capture(function):
        def wrapper(features,parameters,rows,times,rng):
            result=function(features,parameters,rows,times,rng);states.append(copy.deepcopy(rng.bit_generator.state));return result
        return wrapper
    monkeypatch.setattr(conditioned,'construct_future',capture(reference_future))
    before=model.predict(context,node['observation'],executed_action=node['action'],seed=731015,worlds=worlds)
    monkeypatch.setattr(conditioned,'construct_future',capture(candidate))
    after=model.predict(context,node['observation'],executed_action=node['action'],seed=731015,worlds=worlds)
    assert states[0]==states[1]
    for field in ('positions','present','times','weights','source_event_indices','previously_observed'):
        assert getattr(before,field).tobytes()==getattr(after,field).tobytes()
    for field in ('matching','event_person','physical_seen','row_order','first_edge','auxiliary_log_probability'):
        assert getattr(before.association,field).tobytes()==getattr(after.association,field).tobytes()
    assert before.track_keys==after.track_keys and before.association.kernel==after.association.kernel
    assert dict(before.sampling)==dict(after.sampling)
