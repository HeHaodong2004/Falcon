"""Offline ancestry must never improve a score by changing the scored person."""
from types import SimpleNamespace
import unittest
import numpy as np
from eventwm_graph.l2_intermittent.evaluate import extend_ancestry, lineage_slot, score


class AncestryAccountingTests(unittest.TestCase):
    def setUp(self):
        self.component=SimpleNamespace(origin=0,slots=np.array([9,2]),weight=1.,
            last_seen=np.array([0.,1.]),last_position=np.array([[10.,0.],[0.,0.]]),
            birth=np.array([0.,-1.]))
        self.belief=SimpleNamespace(timestamp=1.,components=(self.component,))
        self.times=np.array([1.,4.]);self.truth=np.zeros((2,1,2))
        self.ancestry={(0,9):(0,0.,0.)}

    def test_later_older_slot_cannot_replace_first_anchor(self):
        extended=dict(self.ancestry)
        extend_ancestry(self.belief,extended,self.times,self.truth)
        self.assertEqual(int(self.component.slots[lineage_slot(self.component,extended,0)]),9)
        component=self.component
        class Sampler:
            def predict_shared(self,b,horizon,num_worlds,seed,times):
                positions=np.zeros((num_worlds,len(times),2,2));positions[:,:,0,0]=10.
                return SimpleNamespace(positions=positions,velocities=np.zeros_like(positions),
                    stopped=np.ones((num_worlds,len(times),2),bool),times=1.+np.asarray(times),
                    endpoints=[[component]*num_worlds])
        meta=dict(root_id='synthetic_anchor',scene_id='synthetic',model='fixture',cohort='native',decision='later')
        before=score(Sampler(),self.belief,self.ancestry,[0],{0:0.},self.times,self.truth,meta,41)
        after=score(Sampler(),self.belief,extended,[0],{0:0.},self.times,self.truth,meta,41)
        self.assertEqual([r['tracked_es'] for r in before],[10.]*4)
        self.assertEqual([r['tracked_es'] for r in after],[10.]*4)

    def test_missing_preferred_cannot_be_rescued_by_duplicate(self):
        c=SimpleNamespace(origin=0,slots=np.array([2]),birth=np.array([-1.]))
        mapping={**self.ancestry,(0,2):(0,-1.,1.)}
        self.assertIsNone(lineage_slot(c,mapping,0))

    def test_simultaneous_distinct_first_anchors_are_unsupported(self):
        mapping={**self.ancestry,(0,2):(0,-1.,0.)}
        self.assertIsNone(lineage_slot(self.component,mapping,0))

    def test_recycled_slot_cannot_inherit_physical_ancestry(self):
        c=SimpleNamespace(origin=0,slots=np.array([9]),birth=np.array([.5]))
        self.assertIsNone(lineage_slot(c,self.ancestry,0))

    def test_first_explainer_must_be_common_to_every_positive_fork(self):
        a=SimpleNamespace(origin=0,slots=np.array([7]),weight=.5,last_seen=np.array([1.]),
            last_position=np.array([[0.,0.]]),birth=np.array([0.]))
        b=SimpleNamespace(origin=0,slots=np.array([7,8]),weight=.5,last_seen=np.array([.5,1.]),
            last_position=np.array([[10.,0.],[0.,0.]]),birth=np.array([0.,1.]))
        table={};extend_ancestry(SimpleNamespace(timestamp=1.,components=(a,b)),table,self.times,self.truth)
        self.assertIsNone(lineage_slot(a,table,0));self.assertIsNone(lineage_slot(b,table,0))

    def test_valid_common_first_explainer_is_supported(self):
        a=SimpleNamespace(origin=0,slots=np.array([7]),weight=.5,last_seen=np.array([1.]),
            last_position=np.array([[0.,0.]]),birth=np.array([0.]))
        table={};extend_ancestry(SimpleNamespace(timestamp=1.,components=(a,a)),table,self.times,self.truth)
        self.assertEqual(lineage_slot(a,table,0),0)


if __name__=='__main__':unittest.main()
