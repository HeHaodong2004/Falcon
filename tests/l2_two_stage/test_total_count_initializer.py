"""Use the actual independently audited fitting-only initializer, including its schema."""
import copy
import hashlib
import json

import numpy as np
import pytest

from eventwm_graph.l2_two_stage.common import OUT
from eventwm_graph.l2_two_stage.fit_conditioned import tilt_initial_bias


@pytest.fixture
def initializer():
    path=OUT/'evaluation/pilot_fitting_only_GLOBAL_total_count_tilt_initialization_20261024/initialization.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='bad7a12d4c9518e0ced6c5b8843302092d2d95319add6364c43e3af9dfcdeaa7'
    return json.loads(path.read_text())


def test_actual_initializer_parser_matches_independently_replayed_histogram_ratio(initializer):
    expected=np.log(initializer['smoothed_empirical_probabilities8'])-np.asarray(initializer['average_unconditioned_base_log_probabilities8'])
    expected-=expected.max()
    biases=tilt_initial_bias(initializer,7)
    np.testing.assert_allclose(biases,expected,rtol=0,atol=1e-14)
    assert biases.shape==(8,) and np.isfinite(biases).all()
    assert initializer['fitting_Roots']==1657 and initializer['excluded_validation_Roots']==391
    assert initializer['no_validation_or_dev_parameter_fit']
    biases[:]=0
    assert initializer['log_tilt_weights8'][-1]<-3


@pytest.mark.parametrize('change',[{'schema':'wrong'}, {'log_tilt_weights8':[0]*7},
    {'log_tilt_weights8':[True]*8}, {'log_tilt_weights8':[0]*7+[float('nan')]},
    {'log_tilt_weights8':[0]*7+[1e100]}])
def test_initializer_rejects_schema_shape_or_numeric_corruption(initializer,change):
    bad=copy.deepcopy(initializer);bad.update(change)
    with pytest.raises(ValueError):tilt_initial_bias(bad,7)


def test_initializer_does_not_guess_missing_or_old_proposed_field(initializer):
    bad=copy.deepcopy(initializer);bad['log_total_count_potentials']=bad.pop('log_tilt_weights8')
    with pytest.raises(KeyError):tilt_initial_bias(bad,7)
