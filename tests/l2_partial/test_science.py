import json,unittest
import numpy as np
from eventwm_graph.l2_partial.model import GraphBeliefWorldModel
from eventwm_graph.l2_partial.state import *
from eventwm_graph.l2_partial.probability import *
from eventwm_graph.l2_partial.queries import integration_features

def graph():return ObservedGraph('test',1,.2,0.,{(i,j):(1,0.) for i in range(-30,31) for j in range(-20,21)})
def belief(means=((2.,0.,1.,0.),),weights=None,hidden=1):
    hs=[]
    for i,m in enumerate(means):
        pmf=np.zeros(5);pmf[hidden]=1
        hs.append(Hypothesis(1/len(means),np.asarray(m,dtype=float)[None],np.diag([.4,.4,.5,.5])[None],np.array([0.]),np.array([0]),pmf))
    return BeliefState(graph(),hs,0.,np.zeros(4)).validate()
def observation(t,positions=(),**kw):
    return {'timestamp':t,'robot_pose':[0.,0.,0.,0.],'static_updates':[],
        'tracks':[{'position':[x,0.,z],'last_seen_time':t,'visible_now':True,'valid':True,'track_id':99+i} for i,(x,z) in enumerate(positions)],**kw}
def query(steps=12,turn=False):
    poses=[[0.,0.,0.,0.]]
    for i in range(steps):poses.append([0. if turn else (i+1)*25/120,0.,0.,(i+1)*10/120 if turn else 0.])
    return {'query_id':'turn' if turn else 'forward','executor_version':EXECUTOR_VERSION,'compiled':{'planned_poses':poses,'actions':['left' if turn else 'forward']*steps,'graph_revision':1},'waypoints':[poses[-1][:3]],'final_yaw':poses[-1][3]}

class ProbabilityTests(unittest.TestCase):
    def test_gaussian_known_posterior(self):
        m=np.array([1.,2.,3.,4.]);C=np.eye(4);C[0,2]=C[2,0]=.5
        (mean,cov),lp=condition_position(m,C,[2.,2.])
        np.testing.assert_allclose(mean,[2,2,3.5,4]);self.assertAlmostEqual(cov[2,2],.75);self.assertTrue(np.all(cov[:2]==0));self.assertTrue(np.isfinite(lp))
    def test_semigroup(self):
        h=belief().hypotheses[0].fork(intent=np.array([[.2,-.5]]),diffusion=np.array([.7]),mode=np.array([2]),relaxation=.7)
        a=advance(advance(h,.73),1.17);b=advance(h,1.90)
        np.testing.assert_allclose(a.mean,b.mean,atol=1e-12);np.testing.assert_allclose(a.covariance,b.covariance,atol=1e-12)
    def test_hidden_transfer_fixed_population(self):
        p=np.array([0.,0.,1.]);q,rate=consume_hidden(p);np.testing.assert_array_equal(q,[0,1,0]);self.assertEqual(rate,2.)
    def test_exact_observations_and_count(self):
        wm=GraphBeliefWorldModel(kind='memory_cv');b=wm.observe(belief(),observation(.1,[(2.1,0.)]))
        b.validate()
        for h in b.hypotheses:
            self.assertTrue(any(np.linalg.norm(x[:2]-[2.1,0])<1e-10 for x in h.mean));self.assertAlmostEqual(h.n+h.expected_hidden,2.)
    def test_no_identity_privilege(self):
        wm=GraphBeliefWorldModel(kind='memory_cv');o=observation(.1,[(2.1,0.)]);b=belief()
        a=wm.observe(b,o);o['tracks'][0]['track_id']=-123;o['tracks'][0]['estimated_velocity']=[999,0,999]
        c=wm.observe(b,o);self.assertEqual(a.to_dict(),c.to_dict())
    def test_missing_is_not_empty_and_reappears(self):
        wm=GraphBeliefWorldModel(kind='memory_cv');b=belief()
        c=wm.observe(b,observation(4.));self.assertEqual(c.hypotheses[0].n,1);self.assertAlmostEqual(c.hypotheses[0].expected_hidden,1.)
        d=wm.observe(c,observation(4.1,[(6.1,0.)]));self.assertTrue(all(h.n+h.expected_hidden==2 for h in d.hypotheses))
    def test_certified_empty_point_oracle(self):
        wm=GraphBeliefWorldModel(kind='memory_cv');b=belief(means=((0,0,0,0),(4,0,0,0)),hidden=0)
        b=b.fork(hypotheses=[h.fork(covariance=np.zeros((1,4,4))) for h in b.hypotheses])
        c=wm.observe(b,observation(0,certified_visible_empty_boxes=[[[-1,-1],[1,1]]]))
        self.assertEqual(len(c.hypotheses),1);self.assertAlmostEqual(c.hypotheses[0].weight,1);self.assertEqual(c.hypotheses[0].mean[0,0],4)
        d=wm.observe(b,observation(0,occluded_boxes=[[[-1,-1],[1,1]]]))
        self.assertAlmostEqual(d.hypotheses[0].weight,.5)
    def test_uncertain_empty_not_false_certainty(self):
        wm=GraphBeliefWorldModel(kind='memory_cv')
        b=belief(means=((0,0,0,0),));c=wm.observe(b,observation(0,certified_visible_empty_boxes=[[[-1,-1],[1,1]]]))
        c.validate();self.assertLess(c.evidence['empty_evidence_absolute_error'],.03)
        for h in c.hypotheses:self.assertFalse(np.all(np.abs(h.mean[0,:2])<=1.))
    def test_ambiguous_one_to_one(self):
        b=belief();h=b.hypotheses[0];h=h.fork(mean=np.array([[-.2,0,0,0],[.2,0,0,0]]),covariance=np.tile(np.eye(4),(2,1,1)),last_seen=np.zeros(2),slot_ids=np.arange(2),hidden_pmf=np.array([1.,0.,0.]))
        wm=GraphBeliefWorldModel(kind='memory_cv');c=wm.observe(b.fork(hypotheses=[h]),observation(.1,[[-.1,0],[.1,0]]));self.assertGreater(len(c.hypotheses),1)
        for hh in c.hypotheses:self.assertEqual(hh.n,2);np.testing.assert_allclose(np.sort(hh.mean[:,0]),[-.1,.1],atol=1e-12)
    def test_unexpected_detection_residual(self):
        wm=GraphBeliefWorldModel(kind='memory_cv');b=belief(hidden=0);h=b.hypotheses[0].fork(covariance=np.zeros((1,4,4)))
        c=wm.observe(b.fork(hypotheses=[h]),observation(0,[[100.,100.]]));self.assertEqual(c.unresolved_mass,1.);self.assertTrue(c.evidence['unsupported_observation'])
    def test_serialization(self):
        wm=GraphBeliefWorldModel();b=wm.observe(belief(),observation(.4,[[2.3,.2]]));copy=BeliefState.from_dict(json.loads(json.dumps(b.to_dict())))
        self.assertEqual(b.to_dict(),copy.to_dict());f=wm.predict_shared(b,1.,seed=8);g=wm.predict_shared(copy,1.,seed=8);np.testing.assert_equal(f.positions,g.positions)
    def test_terminal_absorption(self):
        wm=GraphBeliefWorldModel();b=belief().fork(terminal=True);self.assertIs(wm.observe(b,observation(1.)),b)
        f=wm.predict_shared(b,3.);p=wm.evaluate_queries(b,[query()],f).predictions[0];self.assertEqual(p.termination_probabilities['absorbed'],1);self.assertEqual(p.evaluated_interval,(0,0))
    def test_shared_futures_query_permutation(self):
        wm=GraphBeliefWorldModel();b=belief();f=wm.predict_shared(b,3.,seed=2);saved=f.positions.copy()
        a=wm.evaluate_queries(b,[query(),query(turn=True)],f);c=wm.evaluate_queries(b,[query(turn=True),query()],f)
        self.assertAlmostEqual(a.predictions[0].collision_probability,c.predictions[1].collision_probability);np.testing.assert_equal(f.positions,saved)
    def test_incomplete_cursor_and_recursion(self):
        wm=GraphBeliefWorldModel();b=belief();q=query(24);f=wm.predict_shared(b,.48);p=wm.evaluate_queries(b,[q],f).predictions[0]
        self.assertFalse(p.query_complete);self.assertEqual(p.remaining_query_cursor['action_index'],10)
        child=next(x.successor for x in p.observation_branches if not x.successor.terminal);child.validate()
        r=wm.continue_rollout(child,q,horizon=.48);self.assertEqual(r.remaining_query_cursor['action_index'],20)
    def test_branch_total_probability(self):
        wm=GraphBeliefWorldModel();b=belief();f=wm.predict_shared(b,1.);p=wm.evaluate_queries(b,[query()],f).predictions[0]
        self.assertAlmostEqual(sum(x.probability for x in p.observation_branches),1.)
        idx=np.searchsorted(f.times,p.evaluated_interval[1]);prior=sum(w*positions[:h.n].sum(0) for w,positions,h in zip(f.weights,f.positions[:,idx],f.worlds))
        posterior=sum(br.probability*sum(h.weight*h.mean[:,:2].sum(0) for h in br.successor.hypotheses) for br in p.observation_branches)
        np.testing.assert_allclose(prior,posterior,atol=1e-10)
        terminal=sum(x.probability for x in p.observation_branches if x.successor.terminal);self.assertAlmostEqual(terminal,p.collision_probability)
    def test_survival_same_likelihood(self):
        wm=GraphBeliefWorldModel(kind='memory_cv');b=belief(means=((1,0,0,0),(5,0,0,0)))
        c=wm.observe(b,observation(0),{'world_likelihood':[.1,.9]});self.assertAlmostEqual(c.hypotheses[0].weight,.1)
    def test_graph_versions_and_unknown(self):
        g=graph();self.assertEqual(g.get([10000,10000])[0],0)
        row={'cell':[0,0],'state':2,'height':0.,'source':'observed_static_wall_inflated_025m'};h=g.update([row])
        self.assertEqual(g.get([0,0])[0],1);self.assertEqual(h.get([0,0])[0],2);self.assertNotEqual(h.namespace,g.namespace)
    def test_bounded_cache_and_adapter(self):
        wm=GraphBeliefWorldModel();b=belief();f=wm.predict_shared(b,.1)
        for i in range(270):
            q=query(1);q['query_id']=str(i);batch=wm.evaluate_queries(b,[q],f,refined_queries=0)
        self.assertLessEqual(len(wm.graph_cache),256);self.assertEqual(integration_features(batch).shape,(1,8))

class ReviewRegressionTests(unittest.TestCase):
    def test_prepared_serialization(self):
        wm=GraphBeliefWorldModel();b=wm.prepare(belief());copy=BeliefState.from_dict(json.loads(json.dumps(b.to_dict())))
        np.testing.assert_equal(wm.predict_shared(b,1.,seed=3).positions,wm.predict_shared(copy,1.,seed=3).positions)
    def test_rare_mode_preserved(self):
        wm=GraphBeliefWorldModel(kind='memory_cv');b=belief(means=((1,0,0,0),(8,0,0,0)))
        b=b.fork(hypotheses=[h.fork(weight=w) for h,w in zip(b.hypotheses,[.01,.99])]);f=wm.predict_shared(b,1.,num_worlds=16)
        rare=[w for w,h in zip(f.weights,f.worlds) if h.slot_ids[0]==0 and h.mean[0,0]<4]
        self.assertAlmostEqual(sum(rare),.01);self.assertAlmostEqual(f.weights.sum()+f.unresolved_mass,1)
    def test_unrelated_latent_preserved(self):
        wm=GraphBeliefWorldModel();b=belief();h=b.hypotheses[0]
        h=h.fork(mean=np.array([[2,0,0,0],[10,0,0,0.]]),covariance=np.zeros((2,4,4)),last_seen=np.zeros(2),slot_ids=np.arange(2),
                 intent=np.array([[0,0],[1,0.]]),diffusion=np.ones(2)*.2,mode=np.array([0,2]),relaxation=.7)
        c=wm.observe(b.fork(hypotheses=[h]),observation(0,[[2,0]]))
        self.assertTrue(all(np.all(hh.intent[1]==[1,0]) for hh in c.hypotheses if hh.n==2))
    def test_impossible_survival(self):
        wm=GraphBeliefWorldModel();b=wm.observe(belief(),observation(0),{'world_likelihood':[0.]})
        self.assertEqual(b.unresolved_mass,1.);self.assertTrue(b.evidence['unsupported_survival'])
    def test_empty_world_structural_zeros(self):
        wm=GraphBeliefWorldModel();b=belief();h=b.hypotheses[0].fork(mean=np.empty((0,4)),covariance=np.empty((0,4,4)),last_seen=np.empty(0),slot_ids=np.empty(0,int),hidden_pmf=np.array([1.]))
        b=b.fork(hypotheses=[h]);p=wm.evaluate_queries(b,[query()],wm.predict_shared(b,1.)).predictions[0]
        self.assertEqual(p.collision_probability,0.);self.assertEqual(p.psc_probability,0.);self.assertEqual(p.diagnostics['hidden_appearance_probability'],0.)
    def test_wrong_yaw_rejected(self):
        wm=GraphBeliefWorldModel();b=belief().fork(robot_pose=np.array([0,0,0,np.pi]))
        with self.assertRaises(ValueError):wm.evaluate_queries(b,[query()],wm.predict_shared(b,1.))
    def test_sparse_clock_rejected(self):
        wm=GraphBeliefWorldModel();b=belief();f=wm.predict_shared(b,.5,times=[0,.25,.5])
        with self.assertRaises(ValueError):wm.evaluate_queries(b,[query()],f)
    def test_terminal_time_distribution(self):
        from eventwm_graph.l2_partial.queries import NativeEventModel
        coeff=np.zeros((3,12));coeff[:,0]=100;wm=GraphBeliefWorldModel(event_model=NativeEventModel(coeff));b=belief()
        p=wm.evaluate_queries(b,[query()],wm.predict_shared(b,1.)).predictions[0]
        terminal=next(x.successor for x in p.observation_branches if x.successor.terminal)
        self.assertAlmostEqual(np.dot(terminal.terminal_event['times'],terminal.terminal_event['probabilities']),.048,places=8)
        self.assertAlmostEqual(terminal.robot_pose[0],25/120,places=8);self.assertIsNone(terminal.query_cursor)
    def test_cache_invalidation_changes_support(self):
        wm=GraphBeliefWorldModel();b=belief();q=query();p=wm.evaluate_queries(b,[q],wm.predict_shared(b,1.)).predictions[0];self.assertEqual(p.blockage_probability,0.)
        g=b.graph.update([{'cell':[4,0],'state':2,'height':0.,'source':'wall'}]);c=b.fork(graph=g)
        r=wm.evaluate_queries(c,[q],wm.predict_shared(c,1.)).predictions[0];self.assertEqual(r.blockage_probability,1.)
    def test_branch_second_moment_regularization_declared(self):
        wm=GraphBeliefWorldModel();b=belief();f=wm.predict_shared(b,1.);p=wm.evaluate_queries(b,[query()],f).predictions[0];idx=p.diagnostics['endpoint_index']
        prior=sum(w*(f.positions[i,idx,:h.n]**2).sum() for i,(w,h) in enumerate(zip(f.weights,f.worlds)))
        post=sum(br.probability*sum(h.weight*((h.mean[:,:2]**2).sum()+np.trace(h.covariance[:,:2,:2],axis1=1,axis2=2).sum()) for h in br.successor.hypotheses) for br in p.observation_branches)
        self.assertAlmostEqual(post-prior,2*.005**2,places=9)

    def test_wrong_branch_future_rejected(self):
        wm=GraphBeliefWorldModel();b=belief();f=wm.predict_shared(b,.2)
        h=b.hypotheses[0];c=b.fork(hypotheses=[h.fork(mean=h.mean+np.array([50,0,0,0]))])
        with self.assertRaises(ValueError):wm.evaluate_queries(c,[query(2)],f)
    def test_exact_height_and_unsupported_recovery(self):
        wm=GraphBeliefWorldModel(kind='memory_cv');b=belief(hidden=0)
        b=b.fork(hypotheses=[b.hypotheses[0].fork(covariance=np.zeros((1,4,4)))])
        o=observation(0,[[100.,100.]]);o['tracks'][0]['position'][1]=1.234
        c=wm.observe(b,o);f=wm.predict_shared(c,.1)
        self.assertEqual(c.unresolved_mass,1.);self.assertEqual(f.weights.sum(),0.)
        np.testing.assert_allclose(f.positions[:,0,0],np.tile([100.,100.],(16,1)))
        self.assertEqual(f.worlds[0].height[0],1.234)
    def test_omitted_components_are_residual(self):
        wm=GraphBeliefWorldModel(kind='memory_cv');b=belief(means=tuple((i,0,0,0) for i in range(20)),hidden=0)
        f=wm.predict_shared(b,1.,num_worlds=4)
        self.assertAlmostEqual(f.weights.sum()+f.unresolved_mass,1.);self.assertAlmostEqual(f.unresolved_mass,.8)
    def test_timeout_at_native_limit(self):
        wm=GraphBeliefWorldModel();b=belief();q=query(121,turn=True)
        b=b.fork(robot_pose=np.array(q['compiled']['planned_poses'][120]),query_cursor={'query_id':q['query_id'],'action_index':120})
        p=wm.evaluate_queries(b,[q],wm.predict_shared(b,.1)).predictions[0]
        self.assertEqual(p.termination_probabilities['query_timeout'],1.);self.assertFalse(p.query_complete)
    def test_neural_gradient_finite_at_coincident_people(self):
        import torch
        from eventwm_graph.l2_partial.neural import GraphDynamics
        net=GraphDynamics();x=torch.zeros((1,2,4));g=torch.zeros((1,2,18))
        targets,q,logits=net(x,torch.zeros((1,2)),g,torch.ones((1,2)));(targets.sum()+q.sum()+logits.sum()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_cached_observation_is_idempotent(self):
        wm=GraphBeliefWorldModel();b=belief();o=observation(.1,[[2.1,0]])
        c=wm.observe(b,o);self.assertIs(wm.observe(c,o),c)
    def test_duplicate_empty_certificate_is_idempotent(self):
        wm=GraphBeliefWorldModel(kind='memory_cv');b=belief();box=[[-1,-1],[1,1]]
        a=wm.observe(b,observation(.1,certified_visible_empty_boxes=[box]))
        c=wm.observe(b,observation(.1,certified_visible_empty_boxes=[box,box]))
        self.assertEqual(a.to_dict(),c.to_dict())

    def test_privileged_extras_cannot_change_online_prediction(self):
        wm=GraphBeliefWorldModel();b=belief();o=observation(.1,[[2.1,0]])
        a=wm.observe(b,o);o['privileged_future']={'humans':[[999,999]]};o['full_scene_geometry']={'all_free':True}
        c=wm.observe(b,o);self.assertEqual(a.to_dict(),c.to_dict())
        fa=wm.predict_shared(a,.2,seed=9);fc=wm.predict_shared(c,.2,seed=9)
        np.testing.assert_equal(fa.positions,fc.positions)
        self.assertEqual(wm.evaluate_queries(a,[query(2)],fa).predictions[0].collision_probability,wm.evaluate_queries(c,[query(2)],fc).predictions[0].collision_probability)
    def test_graph_overlay_flatten_preserves_legal_support(self):
        original=graph();g=original
        for i in range(20):g=g.update([{'cell':[100+i,100],'state':1,'height':0.,'source':'test_legal_certificate'}])
        self.assertLessEqual(g.depth,8);self.assertEqual(original.get([100,100])[0],0)
        for i in range(20):self.assertEqual(g.get([100+i,100])[0],1)

if __name__=='__main__':unittest.main()
