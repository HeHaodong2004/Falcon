"""Current-score proposal normalization, without checkpoint scoring."""
import math
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
from scipy.special import digamma,polygamma
from scipy.stats import poisson
import torch

from eventwm_graph.l2_two_stage import evaluate_current_population as evaluation
from eventwm_graph.l2_two_stage.conditioned import CurrentProposal,HistoryConditionedLaw,PopulationSamplingFailure
from eventwm_graph.l2_two_stage.association import AssociationGraph
from eventwm_graph.l2_two_stage.positive_history import PositiveSetHistory


def test_normalizer_includes_exact_palm_multiplier_and_keeps_ordinary_path():
    ordinary=dict(proposals_examined_through_last_accept=80)
    palm=dict(ordinary,log_normalizer_multiplier=math.log(2.5))
    expected=float(digamma(32)-digamma(80))
    assert evaluation.current_log_normalizer(32,ordinary)==expected
    assert evaluation.current_log_normalizer(32,dict(ordinary,log_normalizer_multiplier=0.))==expected
    assert evaluation.current_log_normalizer(32,palm)==math.log(2.5)+expected
    # An exact proposal factor is a log term, not an extra random normalizer.
    assert evaluation.nll_mc_uncertainty([dict(root_id='r',scene_id='s')])['number_of_independent_log_estimators']==1


def test_invalid_sampler_normalization_cannot_become_a_scored_density():
    for stats in [dict(proposals_examined_through_last_accept=31),
                  dict(proposals_examined_through_last_accept=32.5),
                  dict(proposals_examined_through_last_accept=80,log_normalizer_multiplier=math.nan),
                  dict(proposals_examined_through_last_accept=80,log_normalizer_multiplier=math.inf)]:
        try:evaluation.current_log_normalizer(32,stats)
        except ValueError:continue
        raise AssertionError('invalid sampler normalization was accepted: '+repr(stats))


def test_model_row_adds_palm_factor_only_to_current_density():
    history=PositiveSetHistory('r','s','q',0,np.array([0.]),np.array([0,0]),
        np.empty((0,3)),np.array([True]),np.array([True]))
    record={key:np.zeros(1) for key in evaluation.FEATURE_FIELDS}
    record.update(keys=(),age=np.empty(0),timestamp=0.,minimum_count=0,positive_history=history,
        source_last_event_indices=np.empty(0,int),robotxy=np.zeros(2),
        population_current_positions=np.empty((0,2)),root_id='r',anchor_index=0)
    graph=SimpleNamespace(minimum_current_count=0,feasible=lambda positions,present:np.ones(len(positions),bool))
    model=SimpleNamespace(parameters=lambda features:{},history_fraction=.5,
        network=SimpleNamespace(compulsory_unique_episode=True))
    atom=CurrentProposal(0,np.zeros(0,bool),np.empty((0,2)),np.empty(0,int),np.empty((0,2)),np.empty((0,2)))
    stats=dict(proposals_examined_through_last_accept=80,log_normalizer_multiplier=math.log(2.5))
    batch=dict(mask=torch.zeros((1,1)),origin=torch.zeros((1,1,2)))
    for worlds in (32,40):
        with patch.object(evaluation,'augment_numpy',side_effect=lambda parameters,*args:parameters), \
             patch.object(evaluation,'draw_conditioned_current',return_value=([atom]*worlds,None,stats)) as draw, \
             patch.object(evaluation,'make_batch',return_value=batch), \
             patch.object(evaluation,'set_nll',return_value=torch.tensor([2.],dtype=torch.float64)), \
             patch.object(evaluation,'estimate_current_count_nll',return_value=(.75,dict(status='OK'))) as count_estimator:
            row=evaluation.model_row(model,record,graph,worlds,100000,1000000)
        expected=math.log(2.5)+float(digamma(worlds)-digamma(80))
        assert row['current_set_nll']==2.+expected
        assert row['current_normalizer_log_estimate']==expected
        assert row['current_normalizer_log_multiplier']==math.log(2.5)
        assert row['current_count_nll']==.75
        assert row['count_rps_finite_bank']==0. and row['occupancy_brier_finite_bank']==0.
        assert count_estimator.call_count==1 and count_estimator.call_args.args[6]==worlds
        assert draw.call_args.args[4]==worlds


def poisson_fixture():
    history=PositiveSetHistory('r','s','q',0,np.array([0.]),np.array([0,0]),
        np.empty((0,3)),np.array([True]),np.array([True]))
    features=dict(keys=(),age=np.empty(0),velocity=np.empty((0,2)),robotxy=np.zeros(2),
        timestamp=0.,minimum_count=0,source_last_event_indices=np.empty(0,int))
    parameters=dict(presence_logits=np.empty(0),rate=np.array(2.),mode_logits=np.zeros(4),
        spatial_logits=np.zeros(1),spatial_mean=np.zeros((1,2)),spatial_scale=np.ones((1,2)),
        means=np.empty((0,4,1,2)),scales=np.empty((0,4,1,2)))
    return parameters,features,history,AssociationGraph.from_history(history)


def test_both_estimators_score_same_analytic_poisson_probability():
    parameters,features,history,graph=poisson_fixture();truth=3;exact=-float(poisson.logpmf(truth,2.))
    for successes in (32,40):
        value,stats=evaluation.estimate_current_count_nll(parameters,features,history,graph,True,truth,successes,803,100000)
        assert abs(value-exact)<1e-12 and stats['status']=='OK'
        assert stats['normalizer_waiting']==stats['count_waiting']==successes
        assert stats['number_of_independent_log_estimators']==2
        assert stats['variance_upper_bound']==float(2*polygamma(1,successes))
    # Here all populations are feasible: public count-stratified acceptance is
    # one in both streams, whereas the retained direct estimator still waits
    # for the Poisson count event. Its ensemble mean targets the same score.
    rng=np.random.default_rng(803);values=[];repetitions=2048
    for _ in range(repetitions):
        direct,_=evaluation.negative_binomial_count_nll(lambda size:rng.poisson(2.,size),truth,32,100000)
        values.append(direct)
    assert abs(np.mean(values)-exact)<6*math.sqrt(float(polygamma(1,32))/repetitions)


def test_public_count_statuses_and_negative_estimate_are_preserved():
    parameters,features,history,graph=poisson_fixture()
    arguments=(parameters,features,history,graph,True,3,32,803,100000)
    with patch.object(HistoryConditionedLaw,'estimate_count_log_probability',return_value=dict(
            status='UNBIASED_LOG_SCORE_ESTIMATE',log_probability=.2)) as api:
        value,stats=evaluation.estimate_current_count_nll(*arguments)
    assert value==-.2 and stats['status']=='OK'
    assert api.call_args.kwargs==dict(successes=32,seed=803,max_proposals=100000)
    with patch.object(HistoryConditionedLaw,'estimate_count_log_probability',return_value=dict(
            status='ZERO_PROPOSAL_COUNT_SUPPORT',log_probability=-math.inf)):
        value,stats=evaluation.estimate_current_count_nll(*arguments)
    assert value is None and stats['status']=='FAIL_ZERO_CURRENT_COUNT_SUPPORT'
    assert stats['log_probability']=='-infinity' # JSON-safe failure, not a clipped score.
    with patch.object(HistoryConditionedLaw,'estimate_count_log_probability',side_effect=PopulationSamplingFailure('cap')):
        value,stats=evaluation.estimate_current_count_nll(*arguments)
    assert value is None and stats['status']=='INCONCLUSIVE_COUNT_SAMPLING_CAP'


def test_density_and_count_mc_bounds_use_different_log_stream_counts():
    rows=[dict(root_id='r',scene_id='s')]
    density=evaluation.nll_mc_uncertainty(rows,independent_log_estimators=1)
    count=evaluation.nll_mc_uncertainty(rows,independent_log_estimators=2)
    assert count['per_anchor_variance_upper_bound']==2*density['per_anchor_variance_upper_bound']
    assert count['aggregate_variance_upper_bound']==2*density['aggregate_variance_upper_bound']
    assert count['number_of_independent_log_estimators']==2


def test_forty_world_finite_and_unbiased_scores_keep_distinct_denominators():
    finite,unbiased=evaluation.count_rps(np.r_[np.zeros(20),np.full(20,2)],1)
    assert finite==.5 and unbiased==1.-20/39
    finite,unbiased=evaluation.occupancy_scores(np.r_[np.zeros(20),np.ones(20)][:,None],np.ones(1))
    assert finite==.25 and abs(unbiased-20*19/(40*39))<1e-15


def test_world_budget_rejected_before_any_data_or_checkpoint_access():
    for worlds in (True,False,0,1,2,31,33,39,41,32.,40.,'40',None):
        with patch.object(evaluation.torch,'load') as load,patch.object(evaluation,'load_records') as records:
            try:evaluation.run('unused','unused.pt',worlds=worlds)
            except ValueError:pass
            else:raise AssertionError('invalid or undeclared world budget accepted: '+repr(worlds))
        assert not load.called and not records.called


def test_run_propagates_declared_budget_to_both_variants_and_all_mc_bounds():
    """Tiny wiring fixture only; no real-checkpoint/data acceptance claim."""
    history=PositiveSetHistory('r','s','q',0,np.array([0.]),np.array([0,0]),
        np.empty((0,3)),np.array([True]),np.array([True]))
    record=dict(root_id='r',scene_id='s',anchor_index=0,timestamp=0.,positive_history=history,
        population_future_valid=np.zeros(8,bool))
    model_result=dict(status='OK',**{key:1. for key in evaluation.KEYS})
    control={key:dict(status='OK',**{metric:2. for metric in evaluation.KEYS if metric!='current_set_nll'})
        for key in ('constant_unconditional','constant_detection_overlay_residual')}
    saved=dict(conditioned_config={},config=dict(split='fit',sidecar='fit',full_fit=True,data_manifest_sha256='identity'))
    with tempfile.TemporaryDirectory() as directory:
        temporary=Path(directory)
        for worlds in (32,40):
            with patch.object(evaluation,'OUT',temporary),patch.object(evaluation,'DOC',temporary), \
                 patch.object(evaluation.torch,'set_num_threads'),patch.object(evaluation.torch,'set_num_interop_threads'), \
                 patch.object(evaluation.torch,'load',return_value=saved),patch.object(evaluation,'sha',return_value='identity'), \
                 patch.object(evaluation,'fit_priors',return_value=({}, {'fitting'}, {'fitting_scene'})), \
                 patch.object(evaluation,'load_records',return_value=[record]), \
                 patch.object(evaluation,'attach_histories',return_value={}),patch.object(evaluation,'attach_metadata',return_value={}), \
                 patch.object(evaluation,'attach_targets',return_value={}),patch.object(evaluation,'HistoryConditionedWorldModel'), \
                 patch.object(evaluation.AssociationGraph,'from_history',return_value=object()), \
                 patch.object(evaluation,'model_row',return_value=model_result) as model_row, \
                 patch.object(evaluation,'baseline_rows',return_value=control) as baseline_rows:
                arguments={} if worlds==32 else dict(worlds=worlds)
                result=evaluation.run('worlds_'+str(worlds),'unused.pt',**arguments)
            assert model_row.call_args.args[3]==baseline_rows.call_args.args[3]==worlds
            assert model_row.call_args.args[4:]==(100000,1000000)
            assert result['anchors']==result['future_censored_anchors_retained']==1
            assert result['roots']==result['scenes']==1 and not result['model_failures']
            protocol=result['protocol']
            assert protocol['worlds']==protocol['count_successes_per_stream']==worlds
            assert f'digamma({worlds})' in protocol['density_estimator']
            assert f'trigamma({worlds})' in protocol['monte_carlo_uncertainty']
            assert f'and{worlds}successes' in protocol['count_nll_estimator']
            assert protocol['reference_scores'].startswith(f'finite{worlds}bank ')
            if worlds==32:
                assert protocol['density_estimator']=='q set NLL + exact sampler log_normalizer_multiplier + digamma(32)-digamma(proposals examined through32nd final acceptance); absent multiplier means ordinary rejection with log multiplier zero'
            for cohort in ('all_current','valid_empty'):
                summary=result['summaries'][cohort]['model']
                for metric,multiplier in (('current_set_nll',1),('current_count_nll',2)):
                    bound=summary[metric]['monte_carlo_uncertainty_bound']
                    assert bound['successes']==worlds and bound['per_anchor_variance_upper_bound']==float(multiplier*polygamma(1,worlds))
                for baseline in control:
                    bound=result['paired_baseline_comparisons'][cohort][baseline]['current_count_nll']['candidate_monte_carlo_uncertainty_bound']
                    assert bound['successes']==worlds and bound['per_anchor_variance_upper_bound']==float(2*polygamma(1,worlds))
            rows=[json.loads(line) for line in (temporary/'evaluation'/('worlds_'+str(worlds))/'current_rows.jsonl').read_text().splitlines()]
            assert len(rows)==3 and {row['variant'] for row in rows}=={'model',*control}


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(function) for function in (
        test_normalizer_includes_exact_palm_multiplier_and_keeps_ordinary_path,
        test_invalid_sampler_normalization_cannot_become_a_scored_density,
        test_model_row_adds_palm_factor_only_to_current_density,
        test_both_estimators_score_same_analytic_poisson_probability,
        test_public_count_statuses_and_negative_estimate_are_preserved,
        test_density_and_count_mc_bounds_use_different_log_stream_counts,
        test_forty_world_finite_and_unbiased_scores_keep_distinct_denominators,
        test_world_budget_rejected_before_any_data_or_checkpoint_access,
        test_run_propagates_declared_budget_to_both_variants_and_all_mc_bounds))


if __name__=='__main__':unittest.main()
