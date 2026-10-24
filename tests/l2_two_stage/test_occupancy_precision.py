"""Regression for the independently audited native-prior binning defect."""
import numpy as np
from eventwm_graph.l2_two_stage.evaluate import occupancy


def test_float32_input_does_not_round_across_an_actual_grid_boundary():
    positions=np.array([[[-1e-9,0.]]],np.float32)
    present=np.ones((1,1),bool);robot=np.zeros(2,np.float32)
    probability=occupancy(positions,present,robot)
    assert probability[5*12+6]==1.
    assert probability[6*12+6]==0.
    np.testing.assert_array_equal(probability,occupancy(positions.astype(np.float64),present,robot.astype(np.float64)))


def test_population_outside_bin_and_duplicate_occupancy_remain_accounted():
    positions=np.array([[[0.,0.],[0.,0.],[100.,0.]]],np.float64)
    probability=occupancy(positions,np.ones((1,3),bool),np.zeros(2))
    assert probability[6*12+6]==1. and probability[-1]==1.
    assert probability.sum()==2.
