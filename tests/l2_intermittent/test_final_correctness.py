"""Independent exhaustive references and original audit regression witnesses."""
import itertools
import math
import unittest
import numpy as np
from scipy.special import logsumexp
from eventwm_graph.l2_intermittent.association import prepare, sample_assignment, association_marginals
from eventwm_graph.l2_intermittent.model import IntermittentWorldModel, detections, sealed
from eventwm_graph.l2_intermittent.observations import condition_observation_event
from eventwm_graph.l2_partial.state import Hypothesis, ObservedGraph
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState


def exhaustive(ll,ranks,pd,miss,bd,den,hmiss):
    n,d=ll.shape;rows=[];vol=[0.,math.log(2.),math.log(math.pi)]
    for assignment in itertools.product(range(-1,n),repeat=d):
        used=[i for i in assignment if i>=0]
        if len(set(used))!=len(used):continue
        nb=assignment.count(-1)
        if nb>len(den):continue
        lp=sum(math.log(miss[i]) for i in range(n) if i not in used)+sum(math.log(x) for x in den[:nb])+hmiss[nb]
        dimension=2*nb
        for z,i in enumerate(assignment):
            if i<0:lp+=bd[z]+vol[2]
            else:lp+=ll[i,z]+pd[z]+vol[ranks[i]];dimension+=ranks[i]
        if math.isfinite(lp):rows.append((assignment,dimension,lp))
    if not rows:return [],[],math.inf
    dimension=min(r[1] for r in rows);rows=[r for r in rows if r[1]==dimension]
    normalizer=logsumexp([r[2] for r in rows]);weights=np.exp([r[2]-normalizer for r in rows])
    return rows,weights,normalizer


class AssociationReferences(unittest.TestCase):
    def test_exact_complete_posterior_against_enumeration(self):
        rng=np.random.default_rng(913)
        for n,d in ((0,2),(2,3),(4,3),(3,0)):
            ll=rng.normal(size=(n,d));rank=np.full(n,2,np.int64);pd=np.log(np.full(d,.8))
            miss=rng.uniform(.1,.8,n);bd=rng.normal(size=d);den=[2.4,1.8,1.1][:d];hm=np.log(np.linspace(.3,.9,len(den)+1))
            rows,w,total=exhaustive(ll,rank,pd,miss,bd,den,hm)
            positive,lmiss,table=prepare(ll,rank,pd,miss,bd,den,hm)
            self.assertAlmostEqual(table[1][0,0],total,places=11)
            self.assertAlmostEqual(table[2],float(-sum(p*math.log(p) for p in w)),places=10)
            marginal,birth=association_marginals(positive,rank,lmiss,table[0],table[1])
            ref=np.zeros((n,d));refbirth=np.zeros(d)
            for row,p in zip(rows,w):
                for z,i in enumerate(row[0]):
                    if i<0:refbirth[z]+=p
                    else:ref[i,z]+=p
            np.testing.assert_allclose(marginal,ref,atol=2e-12)
            np.testing.assert_allclose(birth,refbirth,atol=2e-12)

    def test_all_24_symmetric_assignments_survive_before_carrier_sampling(self):
        ll=np.zeros((4,3));ranks=np.full(4,2,np.int64)
        p,l,t=prepare(ll,ranks,np.zeros(3),np.ones(4),np.full(3,-np.inf),[],[0.])
        self.assertEqual(t[3],24)
        self.assertAlmostEqual(t[2],math.log(24))
        marginal,birth=association_marginals(p,ranks,l,t[0],t[1])
        np.testing.assert_allclose(marginal.sum(1),.75,atol=1e-12)
        rng=np.random.default_rng(7);seen=set()
        for _ in range(1024):seen.add(tuple(sample_assignment(p,ranks,l,t[0],t[1],rng.random(4))[0]))
        self.assertEqual(len(seen),24)

    def test_permuted_centers_have_equal_exact_mass_and_marginals(self):
        ll=np.array([[.4,-1.,2.],[-.5,.7,-.8],[1.,2.,3.]])
        rank=np.full(3,2,np.int64);pd=np.zeros(3);miss=np.array([.2,.5,.7]);bd=np.array([-.4,-1.,-.2])
        reference=None
        for perm in itertools.permutations(range(3)):
            ix=np.array(perm);p,l,t=prepare(ll[:,ix],rank,pd,miss,bd[ix],[1.3,.9,.4],[0.,-.3,-.6,-.8])
            m,b=association_marginals(p,rank,l,t[0],t[1]);value=(t[1][0,0],m[:,np.argsort(ix)])
            if reference is None:reference=value
            self.assertAlmostEqual(value[0],reference[0]);np.testing.assert_allclose(value[1],reference[1],atol=1e-12)

    def test_mixed_rank_assignments_use_total_small_ball_dimension(self):
        # One line-Gaussian track can explain either center; the other is born.
        p,l,t=prepare(np.zeros((1,2)),np.array([1]),np.zeros(2),np.ones(1),np.zeros(2),[1.],[0.,0.])
        self.assertEqual(t[0][0,0],3);self.assertEqual(t[3],2)
        m,b=association_marginals(p,np.array([1]),l,t[0],t[1])
        np.testing.assert_allclose(m,.5);np.testing.assert_allclose(b,.5)

    def test_hidden_atom_dominates_diffuse_tracked_density(self):
        p,l,t=prepare(np.zeros((1,1)),np.array([2]),np.zeros(1),np.ones(1),np.zeros(1),[1.],[0.,0.],birth_rank=0)
        self.assertEqual(t[0][0,0],0)
        m,b=association_marginals(p,np.array([2]),l,t[0],t[1])
        np.testing.assert_array_equal(m,0.);np.testing.assert_array_equal(b,1.)


class AuditWitnesses(unittest.TestCase):
    def setUp(self):
        self.model=IntermittentWorldModel(max_components=8)
        g=ObservedGraph('audit-witness',0,.2,0.,{})
        h=Hypothesis(1.,np.array([[2.,1.,.5,0.]]),np.diag([.04,.04,.1,.1])[None],np.array([0.]),np.array([0]),np.array([.5,.5]),birth_time=np.array([0.]))
        self.b=self.model.from_legacy(StructuralBeliefState(g,[h],0.,np.zeros(4),model_version='test'),17)

    def ob(self,t,x=2.2):
        return dict(timestamp=t,robot_pose=[0.,0.,0.,0.],static_updates=[],tracks=[dict(position=[x,0.,1.],visible_now=True,last_seen_time=t,valid=True)])

    def test_current_positive_timestamp_rejected(self):
        ob=self.ob(.1);ob['tracks'][0]['last_seen_time']=0.
        with self.assertRaisesRegex(ValueError,'timestamp'):self.model.observe(self.b,ob)

    def test_unavailable_final_initialization_accepts_same_time_positive(self):
        unavailable={**self.ob(.1),'tracks':[],'human_sensor_available':False}
        graph=dict(timestamp=.1,resolution=.2,height=0.,revision=0,cells=[])
        b=self.model.initialize([unavailable],graph)
        self.assertEqual(b.receipt,'')
        post=self.model.observe(b,self.ob(.1))
        self.assertTrue(any(c.n for c in post.components))
        self.assertEqual(post.fingerprint(),self.model.observe(post,self.ob(.1)).fingerprint())

    def test_prefix_repeated_evidence_does_not_condition_twice(self):
        ob=self.ob(.1);graph=dict(timestamp=.1,resolution=.2,height=0.,revision=0,cells=[])
        a=self.model.initialize([ob],graph)
        b=self.model.initialize([ob,dict(ob)],graph)
        self.assertEqual(a.fingerprint(),b.fingerprint())
        with self.assertRaisesRegex(ValueError,'conflicting'):self.model.initialize([ob,self.ob(.1,2.3)],graph)

    def test_duplicate_prefix_compatibility_is_explicit_and_persistent(self):
        ob=self.ob(.1);ob['tracks'].append(dict(ob['tracks'][0]))
        graph=dict(timestamp=.1,resolution=.2,height=0.,revision=0,cells=[])
        b=self.model.initialize([ob],graph)
        self.assertTrue(b.diagnostics['bootstrap_duplicate_fallback'])
        self.assertEqual(b.fingerprint(),self.model.observe(b,ob).fingerprint())
        b=self.model.propagate(b,.2)
        self.assertTrue(b.diagnostics['bootstrap_duplicate_fallback'])
        f=self.model.predict_shared(b,.2,8,1,times=[0.,.2])
        self.assertTrue(self.model.successor_at(f,1).diagnostics['bootstrap_duplicate_fallback'])

    def test_permutations_and_anonymous_ids_do_not_change_update(self):
        ob=self.ob(.1);ob['tracks'].append(dict(position=[2.5,0.,1.2],visible_now=True,last_seen_time=.1,valid=True,track_id=789))
        a=self.model.observe(self.b,ob)
        ob['tracks']=list(reversed(ob['tracks']));ob['tracks'][1]['track_id']='new-anonymous-id'
        b=self.model.observe(self.b,ob)
        self.assertEqual(a.fingerprint(),b.fingerprint())

    def test_allocator_survives_total_slot_support_loss_and_serialization(self):
        cs=[]
        for c in self.b.components:
            kw={k:v[:0] for k,v in vars(c).items() if isinstance(v,np.ndarray) and k not in ('hidden_pmf','hidden_mean','hidden_covariance')}
            cs.append(sealed(c.fork(**kw)))
        lost=self.b.fork(components=tuple(cs))
        restored=self.model.deserialize_actual_belief(self.model.serialize_actual_belief(lost))
        updated=self.model.observe(restored,self.ob(.1))
        self.assertTrue(all(0 not in c.slots for c in updated.components))
        self.assertGreater(updated.next_slot,restored.next_slot)

    def test_prediction_provenance_and_legal_positions_are_distinct(self):
        post=self.model.observe(self.b,self.ob(.1));stale=self.model.propagate(post,.6)
        tracks=self.model.get_previously_observed_tracks(stale)
        for track in tracks:
            self.assertEqual(track['state_provenance'],'propagated_prediction')
            for measurement in track['last_observed_measurements']:
                np.testing.assert_allclose(measurement['position'],[2.2,1.])
        fresh=self.model.get_previously_observed_tracks(post)
        self.assertTrue(all(t['state_provenance']=='legal_observation' or t['age']>0 for t in fresh))

    def test_invalid_event_probabilities_rejected(self):
        f=self.model.predict_shared(self.b,.2,8,1,times=[0.,.2])
        event=dict(successor_supported=True,future_identity=f.identity(),index=1,robot_pose=np.zeros(4),region=None)
        for none in ([np.nan]*8,[1.1]*8,[-.1]*8,[.5]*7):
            with self.assertRaises(ValueError):condition_observation_event(self.model,f,dict(event,per_world_none=none),'none')
        with self.assertRaises(ValueError):condition_observation_event(self.model,f,dict(event,per_world_none=np.zeros(8)),'none')


if __name__=='__main__':unittest.main()
