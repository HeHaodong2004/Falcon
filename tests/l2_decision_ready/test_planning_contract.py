"""Decision interface and actual-belief counterexamples; no dataset required."""
import copy
import unittest
import threading
from unittest.mock import patch
import numpy as np
from scripts.l2_partial_structural_api_example import build_model, legal_fixture, observation, queries
from eventwm_graph.l2_partial.planning import (
    packet_from_batch, online_planning_call, graph_decision_features,
    pessimistic_scores, CandidateStatus, BranchStatus,
)
from eventwm_graph.l2_partial.structural_budget import UnsupportedConditioning
from eventwm_graph.l2_partial.structural_budget import ACTIVE_DEADLINE,Deadline
from eventwm_graph.l2_partial.structural_state import freeze_hypotheses


class PlanningContractTests(unittest.TestCase):
    def setUp(self):
        self.model,_ = build_model()
        history,graph = legal_fixture()
        self.b = self.model.initialize(history,graph)
        self.q = queries(self.b.graph.version)

    def evaluate(self,b=None,qs=None,horizon=.288,refined=None):
        b=self.b if b is None else b; qs=self.q if qs is None else qs
        f=self.model.predict_shared(b,horizon,16,27)
        batch=self.model.evaluate_queries(b,qs,f,budget_ms=5000.,refined_query_indices=refined)
        return f,batch,packet_from_batch(b,qs,batch,pointgoal=[3.,0.,0.])

    def test_packet_shares_world_and_graph_storage(self):
        f,batch,p=self.evaluate()
        self.assertIs(p.graph_reference,self.b.graph)
        for name in ('positions','velocities','weights','times'):
            self.assertIs(getattr(p.shared_future,name),getattr(f,name))
            self.assertFalse(getattr(p.shared_future,name).flags.writeable)
        with self.assertRaises(TypeError):p.graph_reference.cells[(0,0)]=(2,0.)
        with self.assertRaises(TypeError):p.uncertainty_diagnostics['x']=1

    def test_preserves_interval_and_paired_world_responses(self):
        f,batch,p=self.evaluate()
        for c,pred in zip(p.candidates,batch.predictions):
            self.assertEqual(c.collision_risk_interval,pred.collision_interval)
            self.assertAlmostEqual(float(f.weights@c.per_world_collision),c.collision_risk_interval[0])
            self.assertAlmostEqual(float(f.weights@c.per_world_social),c.social_risk_interval[0])
            self.assertFalse(c.per_world_social.flags.writeable)

    def test_signed_goal_progress_differs_from_route_travel(self):
        _,_,p=self.evaluate()
        self.assertGreater(p.candidates[0].goal_progress.mean_interval[0],0.)
        self.assertEqual(p.candidates[1].goal_progress.mean_interval,(0.,0.))
        self.assertEqual(p.candidates[1].route_progress.mean_interval,(0.,0.))

    def test_unresolved_bucket_is_not_a_supported_observation(self):
        b=self.b.fork(hypotheses=[h.fork(weight=h.weight*.8) for h in self.b.hypotheses],unresolved_mass=.2)
        _,batch,p=self.evaluate(b=b)
        for c in p.candidates:
            self.assertNotIn('unresolved_physical_outcome',[x.kind for x in c.observation_branches])
            self.assertAlmostEqual(sum(x.lower_mass for x in c.observation_branches),c.represented_mass)
            self.assertAlmostEqual(c.duration.lower_weights.sum()+c.duration.unresolved_mass,1.)
            self.assertGreater(c.collision_risk_interval[1]-c.collision_risk_interval[0],.19)
        self.assertTrue(any(x.observation['kind']=='unresolved_physical_outcome' for x in batch.predictions[0].observation_branches))

    def test_censored_query_preserves_endpoint_orientation_cursor(self):
        _,_,p=self.evaluate(horizon=.096)
        for c in p.candidates:
            self.assertEqual(c.status,CandidateStatus.CENSORED)
            self.assertEqual(c.remaining_cursor['action_index'],2)
            self.assertEqual(c.evaluated_interval,(0.,.096))
            self.assertEqual(c.geometry.evaluated_survival_endpoint,tuple(c.geometry.planned_poses[2]))
            for br in c.observation_branches:self.assertFalse(br.successor.can_select_new_query)

    def test_selected_rows_deferred_is_distinct_from_impossible(self):
        f,batch,p=self.evaluate(refined=(1,))
        self.assertEqual(p.candidates[0].branch_status,BranchStatus.DEFERRED)
        self.assertEqual(p.candidates[0].observation_branches,())
        self.assertEqual(p.candidates[1].branch_status,BranchStatus.AVAILABLE)
        other=self.model.evaluate_queries(self.b,self.q,f,budget_ms=5000.,refined_queries=0)
        for a,b in zip(batch.predictions,other.predictions):
            self.assertEqual(a.collision_interval,b.collision_interval)
            self.assertEqual(a.psc_interval,b.psc_interval)

    def test_candidate_reordering_preserves_common_world_consequences(self):
        f,batch,p=self.evaluate()
        reversed_batch=self.model.evaluate_queries(self.b,self.q[::-1],f,budget_ms=5000.)
        for a,b in zip(batch.predictions,reversed_batch.predictions[::-1]):
            self.assertEqual(a.collision_interval,b.collision_interval)
            np.testing.assert_array_equal(a.world_social_probability,b.world_social_probability)

    def test_no_new_graph_connectivity_in_successors(self):
        _,_,p=self.evaluate()
        for c in p.candidates:
            for br in c.observation_branches:
                self.assertIs(br.successor.resolve().graph,self.b.graph)
                self.assertFalse(br.static_graph_delta_supported)
                self.assertFalse(br.certified_human_empty)

    def test_unknown_support_is_not_a_safe_action(self):
        qs=copy.deepcopy(self.q)
        qs[0]['compiled']['planned_poses'][-1][0]=100.
        _,_,p=self.evaluate(qs=qs)
        self.assertEqual(p.candidates[0].status,CandidateStatus.UNSUPPORTED)
        self.assertEqual(pessimistic_scores(p)[0],-np.inf)
        self.assertFalse(graph_decision_features(p).decision_supported[0])
        self.assertEqual(self.b.graph.get((500,0))[0],0)

    def test_actual_and_imagined_beliefs_remain_separate(self):
        _,_,p=self.evaluate()
        br=next(br for br in p.candidates[0].observation_branches if br.kind=='survived_detection')
        hypothetical=br.successor.resolve()
        with self.assertRaises(UnsupportedConditioning):
            self.model.observe(hypothetical,observation(hypothetical.timestamp,2.037,hypothetical.robot_pose))
        actual=self.model.observe(self.b,observation(hypothetical.timestamp,2.037,hypothetical.robot_pose))
        self.assertIsNone(actual.carrier)
        for h in actual.hypotheses:
            if h.weight>0:self.assertTrue(np.any(np.isclose(h.mean[:,0],2.037)))
        with self.assertRaises(ValueError):
            online_planning_call(self.model,hypothetical,observation(.3,2.),self.q)

    def test_timeout_publishes_no_partial_actual_update(self):
        fingerprint=self.b.fingerprint()
        result=online_planning_call(self.model,self.b,observation(.048,2.01),self.q,budget_ms=0.)
        self.assertEqual(result.status,'PENDING_OBSERVATION')
        self.assertIsNone(result.packet);self.assertIs(result.actual_belief,self.b)
        self.assertEqual(fingerprint,self.b.fingerprint())

    def test_terminal_absorption_and_no_new_risk(self):
        b=self.b.fork(terminal=True)
        _,_,p=self.evaluate(b=b)
        for c in p.candidates:
            self.assertEqual(c.status,CandidateStatus.ABSORBED)
            self.assertEqual(c.collision_risk_interval,(0.,0.))
            self.assertFalse(c.observation_branches[0].successor.can_select_new_query)

    def test_horizon_zero_does_not_emit_observations(self):
        _,_,p=self.evaluate(horizon=0.)
        for c in p.candidates:
            self.assertEqual(c.collision_risk_interval,(0.,0.))
            self.assertEqual(c.branch_status,BranchStatus.NO_EXPOSURE)

    def test_packet_rejects_geometry_mutation(self):
        _,batch,_=self.evaluate()
        changed=copy.deepcopy(self.q);changed[0]['compiled']['planned_poses'][-1][0]+=.01
        with self.assertRaises(ValueError):packet_from_batch(self.b,changed,batch)

    def test_packet_rejects_requested_geometry_and_primitive_mutation(self):
        _,batch,_=self.evaluate()
        for key in ('yaw','waypoint','action'):
            changed=copy.deepcopy(self.q)
            if key=='yaw':changed[0]['final_yaw']+=.1
            elif key=='waypoint':changed[0]['waypoints'][0][0]+=.1
            else:changed[0]['compiled']['actions'][0]='STOP'
            with self.subTest(key=key),self.assertRaises(ValueError):packet_from_batch(self.b,changed,batch)

    def test_unknown_and_stop_primitives_are_refused(self):
        for action in ('STOP','teleport'):
            changed=copy.deepcopy(self.q);changed[0]['compiled']['actions'][0]=action
            with self.subTest(action=action),self.assertRaises(ValueError):self.evaluate(qs=changed)

    def test_residual_never_populates_impossible_termination_causes(self):
        b=self.b.fork(hypotheses=[h.fork(weight=h.weight*.8) for h in self.b.hypotheses],unresolved_mass=.2)
        for horizon,survivor in ((.288,'query_complete'),(.096,'horizon_censored')):
            _,_,p=self.evaluate(b=b,horizon=horizon)
            for c in p.candidates:
                self.assertNotIn('unresolved',c.termination_lower_masses)
                for cause,interval in c.termination_intervals.items():
                    if cause not in ('human_collision',survivor):self.assertEqual(interval,(0.,0.))

    def test_executor_limit_ends_reference_but_allows_another_high_level_query(self):
        from eventwm_graph.l2_partial.state import ObservedGraph
        b=self.b.fork(graph=ObservedGraph('long-known-strip',1,.2,0.,{(x,z):(1,0.) for x in range(-15,146) for z in range(-10,11)}))
        qs=queries(b.graph.version,steps=121)[:1]
        _,_,p=self.evaluate(b=b,qs=qs,horizon=6.)
        c=p.candidates[0]
        self.assertEqual(c.status,CandidateStatus.EXECUTOR_LIMIT)
        self.assertEqual(c.remaining_cursor['action_index'],120)
        for br in c.observation_branches:
            if not br.successor.terminal:
                self.assertTrue(br.successor.can_select_new_query)
                self.assertFalse(br.successor.can_continue_reference)

    def test_known_analytic_imagination_is_not_an_actual_execution_prior(self):
        f,_,_=self.evaluate()
        imagined=self.model.successor_at(f,1,f.weights,np.asarray(self.q[0]['compiled']['planned_poses'][1]))
        self.assertIsNone(imagined.carrier)
        self.assertEqual(imagined.evidence['belief_role'],'imagined_analytic')
        with self.assertRaises(ValueError):online_planning_call(self.model,imagined,observation(.096,2.),self.q)

    def test_outer_deadline_cannot_be_erased_by_unbounded_adapter(self):
        token=ACTIVE_DEADLINE.set(Deadline.create(0.))
        try:
            outer=ACTIVE_DEADLINE.get()
            result=online_planning_call(self.model,self.b,observation(.048,2.),self.q,budget_ms=None)
            self.assertTrue(result.pending_observation)
            self.assertIs(ACTIVE_DEADLINE.get(),outer)
        finally:ACTIVE_DEADLINE.reset(token)

    def test_lock_wait_is_part_of_adapter_deadline(self):
        acquired=threading.Event();release=threading.Event()
        def holder():
            with self.model._instance_lock:
                acquired.set();release.wait(2.)
        thread=threading.Thread(target=holder);thread.start();acquired.wait(2.)
        try:
            result=online_planning_call(self.model,self.b,observation(.048,2.),self.q,budget_ms=1.)
            self.assertEqual(result.timeout_stage,'planning:instance_lock')
            self.assertIsNone(result.packet)
        finally:release.set();thread.join()

    def test_late_timeout_also_keeps_actual_observation_pending(self):
        from eventwm_graph.l2_partial.structural_budget import InferenceBudgetExceeded
        before=self.b.fingerprint()
        def expire(*args,**kwargs):
            deadline=ACTIVE_DEADLINE.get()
            raise InferenceBudgetExceeded('injected:forecast',deadline.budget_ms+1,deadline.budget_ms,True)
        with patch.object(self.model,'predict_shared',side_effect=expire):
            result=online_planning_call(self.model,self.b,observation(.048,2.01),self.q,budget_ms=5000.)
        self.assertTrue(result.pending_observation);self.assertIsNone(result.packet)
        self.assertEqual(before,self.b.fingerprint())

    def test_new_paired_outputs_match_reference_kernels(self):
        slow,_=build_model(optimized=False)
        f,batch,p=self.evaluate()
        other_future=slow.predict_shared(self.b,.288,16,27)
        other=slow.evaluate_queries(self.b,self.q,other_future,budget_ms=5000.)
        for a,b in zip(batch.predictions,other.predictions):
            np.testing.assert_allclose(a.world_collision_probability,b.world_collision_probability,atol=2e-8,rtol=2e-9)
            np.testing.assert_allclose(a.world_social_probability,b.world_social_probability,atol=2e-8,rtol=2e-9)

    def test_stage0_topology_join_checks_arrived_support(self):
        from eventwm_graph.l2_partial.planning import bind_stage0_context
        _,_,p=self.evaluate();_,graph=legal_fixture()
        graph['global_nodes']={'n':{'cell':[0,0],'position':[.1,0.,.1],'kind':'frontier_viewpoint'}}
        context=bind_stage0_context(graph,p,pointgoal=[3.,0.,0.])
        self.assertEqual(context.graph_version,p.graph_version)
        self.assertIn('n',context.global_nodes)
        changed=copy.deepcopy(graph);changed['timestamp']=99.
        with self.assertRaises(ValueError):bind_stage0_context(changed,p,pointgoal=[3.,0.,0.])
        changed=copy.deepcopy(graph);changed['cells'][0]['state']=2
        with self.assertRaises(ValueError):bind_stage0_context(changed,p,pointgoal=[3.,0.,0.])

    def test_template_snapshot_shares_arrays_but_never_live_mutable_storage(self):
        h=self.b.hypotheses[0];templates=freeze_hypotheses([h,h.fork(weight=.2)],weight=0.)
        self.assertIs(templates[0].mean,templates[1].mean)
        self.assertIsNot(templates[0].mean,h.mean)
        before=templates[0].mean.copy();h.mean[0,0]+=1.
        np.testing.assert_array_equal(templates[0].mean,before)

    def test_readonly_view_does_not_hide_a_mutable_owner(self):
        h=self.b.hypotheses[0];owner=h.mean.copy();view=owner.view();view.setflags(write=False)
        h=h.fork(mean=view);templates=freeze_hypotheses([h])
        before=templates[0].mean.copy();owner[0,0]+=100.
        np.testing.assert_array_equal(templates[0].mean,before)

    def test_single_template_freezer_also_owns_readonly_view_storage(self):
        from eventwm_graph.l2_partial.structural_state import freeze_hypothesis
        h=self.b.hypotheses[0];owner=h.mean.copy();view=owner.view();view.setflags(write=False)
        sealed=freeze_hypothesis(h.fork(mean=view));snapshot=freeze_hypotheses([sealed])
        before=snapshot[0].mean.copy();owner[0,0]+=100.
        np.testing.assert_array_equal(snapshot[0].mean,before)

    def test_malformed_or_future_static_updates_fail_atomically(self):
        base={'cell':[50,50],'state':1,'height':0.,'source':'synthetic_observed_certificate','timestamp':.048}
        for changes in ({'timestamp':99.},{'timestamp':0.},{'state':0},{'state':3},{'height':float('nan')},{'cell':[1.5,2]},{'source':''}):
            with self.subTest(changes=changes):
                row=observation(.048,2.01);row['static_updates']=[{**base,**changes}]
                before=self.b.fingerprint()
                with self.assertRaises(ValueError):self.model.observe(self.b,row)
                self.assertEqual(before,self.b.fingerprint())

    def test_actual_replay_tracker_changes_and_graph_only_delta_do_not_duplicate_sensor(self):
        row=observation(.048,2.01);post=self.model.observe(self.b,row)
        changed=copy.deepcopy(row);changed['tracks'][0]['track_id']='new-source-id'
        changed['static_updates']=[{'cell':[50,50],'state':2,'height':0.,'source':'synthetic_wall','timestamp':.048}]
        again=self.model.observe(post,changed)
        self.assertEqual(post.unresolved_mass,again.unresolved_mass)
        for h,hh in zip(post.hypotheses,again.hypotheses):np.testing.assert_array_equal(h.mean,hh.mean)


if __name__=='__main__':unittest.main()
