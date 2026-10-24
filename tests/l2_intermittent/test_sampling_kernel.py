"""Full-path coupling and RNG continuation against the frozen NumPy sampler."""
import unittest
import numpy as np
from eventwm_graph.l2_intermittent.dynamics import (
    Component,DEFAULT,sample_joint_paths,_sample_joint_paths_reference)


def component(n,seed,start):
    r=np.random.default_rng(seed)
    return Component(1.,r.normal(size=(n,4)),np.zeros((n,4,4)),np.arange(n),
        np.full(n,start),np.full(n,start),np.zeros((n,2)),r.random(n)<.5,
        r.normal(size=(n,2)),r.uniform(.001,.3,n),start+r.uniform(.001,.4,n),
        np.zeros(n,dtype=np.int64),seed,np.array([1.]))


class CompiledSamplingTests(unittest.TestCase):
    def assert_coupled(self,worlds,rel,start,cv=None):
        a=np.random.default_rng(73);b=np.random.default_rng(73)
        fast=sample_joint_paths(worlds,rel,start,DEFAULT,a,cv)
        frozen=_sample_joint_paths_reference(worlds,rel,start,DEFAULT,b,cv)
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)
        for i,(x,y) in enumerate(zip(fast,frozen)):
            if i<2:np.testing.assert_allclose(x,y,rtol=0,atol=1e-10,equal_nan=True)
            else:np.testing.assert_array_equal(x,y)
        # A continued consumer must receive exactly the same remaining stream.
        np.testing.assert_array_equal(a.normal(size=50),b.normal(size=50))

    def test_default_and_off_grid_paths_padding_modes_and_world_counts(self):
        start=1.23456789
        grid=np.unique(np.r_[np.arange(63)*.048,3.])
        for worlds in (8,16,32,64):
            sources=[component(n,221+i,start) for i,n in enumerate((0,1,3,6))]
            self.assert_coupled([sources[i%4] for i in range(worlds)],grid,start)
        self.assert_coupled([component(6,17,start)]*16,np.array([0.,.013,.054,.231,.5,1.003,2.231,3.]),start)

    def test_endpoint_tolerance_events_and_very_short_actual_intervals(self):
        start=.013;grid=np.array([0.,1e-14,5e-13,2e-12,.000013,.048,.5,1.,3.])
        worlds=[component(3,221+i,start) for i in range(8)]
        for i,c in enumerate(worlds):
            c.next_switch[:]=start+np.array([0.,.048+(-.5 if i%2 else .5)*1e-12,.5])
        self.assert_coupled(worlds,grid,start)

    def test_zero_horizon_empty_population_and_cv_controls(self):
        start=0.;sources=[component(0,221,start)]*16
        self.assert_coupled(sources,np.array([0.]),start)
        self.assert_coupled(sources,np.array([0.,.048,3.]),start)
        for cv in (0.,.16,.64):
            self.assert_coupled([component(3,221,start)]*8,np.array([0.,.013,.048,1.,3.]),start,cv)


if __name__=='__main__':unittest.main()
