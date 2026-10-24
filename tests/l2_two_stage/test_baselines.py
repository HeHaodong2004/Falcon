"""Fair baseline fitting and complete legal-interface checks on actual data."""
from pathlib import Path
import numpy as np
import pytest
from eventwm_graph.l2_two_stage.baselines import KinematicPredictor
from eventwm_graph.l2_two_stage.context import OnlineContext
from eventwm_graph.l2_two_stage.data import jobs_for,read,load_records
from eventwm_graph.l2_two_stage.evaluate import Kinematic
from eventwm_graph.l2_two_stage.features import extract


@pytest.mark.parametrize('graph',[False,True])
def test_streamed_fit_matches_literal_root_weighted_reference(graph):
    records=load_records('train_pilot_native_v2')[:80]
    model=Kinematic(graph=graph).fit(records);x=[];y=[];weights=[]
    for r in records:
        for i in range(len(r['keys'])):
            for t in np.flatnonzero(r['target_mask'][i]):
                seconds=r['times'][t]
                if seconds<.4:continue
                x.append(np.r_[1,r['x'][i],r['graph'][i].mean(0)])
                y.append((r['target'][i,t]-r['origin'][i]-r['velocity'][i]*seconds)/seconds)
                weights.append(r['weight']/max(1,r['target_mask'].sum()))
    x=np.array(x);y=np.array(y);w=np.array(weights)
    if graph:
        coefficient=np.linalg.solve((x*w[:,None]).T@x+np.eye(x.shape[-1])*.1,(x*w[:,None]).T@y)
        np.testing.assert_allclose(model.coefficients,coefficient,atol=2e-6,rtol=2e-5)
        y-=x@coefficient
    np.testing.assert_allclose(model.diffusion,max(.05,np.sqrt(np.average((y**2).mean(-1),weights=w))),rtol=1e-6)


@pytest.mark.parametrize('memory,graph',[(False,False),(True,False),(True,True)])
@pytest.mark.parametrize('worlds',[32,40])
def test_online_baseline_matches_exact_prepared_feature_forecast(memory,graph,worlds):
    legal=read(Path(jobs_for('dev')[1][0])/'legal_root.json.gz')
    context=OnlineContext.from_raw(legal['history'],legal['partial_graph'],legal['history_actions'])
    f=extract(context);model=Kinematic(memory,graph).fit(load_records('train_pilot_native_v2')[:80])
    parameters=dict(schema='legal_kinematic_baseline_1',memory=memory,graph=graph,diffusion=model.diffusion,
                    coefficients=None if model.coefficients is None else model.coefficients.tolist())
    public=KinematicPredictor(parameters).predict(context,legal['history'][-1],seed=193,worlds=worlds)
    expected,supported=model.predict(f,public.times,seed=193,worlds=worlds)
    assert expected.shape==(worlds,64,len(f['keys']),2)
    np.testing.assert_array_equal(public.positions,expected)
    np.testing.assert_array_equal(public.supported,supported)
    with pytest.raises(ValueError):public.positions.setflags(write=True)
    assert 'UNATTRIBUTED_POPULATION_UNMODELED' in public.support


@pytest.mark.parametrize('worlds',[True,1,0,-1,2.5])
def test_invalid_evaluation_budget_fails_before_any_data_read(worlds):
    from eventwm_graph.l2_two_stage.evaluate import run
    with pytest.raises(ValueError,match='integer iid worlds'):
        run('invalid-budget-must-not-create-artifact',split='nonexistent',worlds=worlds)
