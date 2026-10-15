from dataclasses import replace
from types import MethodType
import unittest

import torch
from torch import nn

from eventforecastwm.model import energy_score, trajectory_log_probability
from eventforecastwm.model_agent_scale import AgentScaleWorldModel
from eventforecastwm.model_posterior import PosteriorForecastWorldModel
from eventforecastwm.performance_v2.physical import PhysicalWorldModel, arrival_forward, VARIANTS
from test_forecast_model import legal


class TestPhysicalV2(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(58)

    def test_missing_packets_hold_encoding_and_preserve_gradients(self):
        gru=nn.GRU(13,16,batch_first=True)
        gru.forward=MethodType(arrival_forward,gru)
        observed=torch.randn(2,3,13)
        observed[...,6]=1;observed[...,10]=1
        observed[...,7]=0;observed[...,8]=torch.tensor([-.2,-.1,0.])
        original,_=gru(observed)
        # Interleave absent packets; the timestamps of real packets are unchanged.
        missing=torch.zeros(2,7,13);missing[:,[0,2,4]]=observed
        output,hidden=gru(missing)
        self.assertTrue(torch.equal(original[:,2],output[:,-1]))
        self.assertTrue(torch.equal(output[:,4],output[:,5]))
        self.assertTrue(torch.equal(output[:,5],output[:,6]))
        output.square().sum().backward()
        self.assertGreater(float(gru.weight_ih_l0.grad.norm()),0.)
        # A retained stale observation is also not a new arrival.
        stale=missing.clone();stale[:,5:]=missing[:,4:5]
        stale[:,5:,7]=torch.tensor([.1,.2]);stale[:,5:,8]=torch.tensor([.1,.2])
        self.assertTrue(torch.equal(output,gru(stale)[0]))
        absent,_=gru(torch.zeros(1,5,13))
        self.assertTrue(torch.equal(absent,torch.zeros_like(absent)))

    def test_capacity_composition_sampling_and_query_independence(self):
        baseline=PosteriorForecastWorldModel(noise='integrated')
        for variant in VARIANTS:
            model=PhysicalWorldModel(variant=variant,noise='integrated')
            self.assertEqual(model.parameter_counts(),dict(baseline.parameter_counts(),recipe=model.recipe))
            model.core.acceleration[-1].weight.data.normal_(0,.02)
            b=model.encode_history(legal())
            whole=model.imagine(b,horizon_s=3.)
            left=model.imagine(b,candidate_query=torch.randn(1,9,10),horizon_s=1.)
            right=model.imagine(left.next_belief,candidate_query=torch.full((1,7,10),float('nan')),horizon_s=2.)
            self.assertTrue(torch.equal(whole.positions[:,:,4:],right.positions),variant)
            self.assertTrue(torch.equal(whole.covariance[:,:,4:],right.covariance),variant)
            g1=torch.Generator().manual_seed(71);g2=torch.Generator().manual_seed(71)
            expected=whole.sample(4,g1)[0];a,_,state=left.sample(4,g2);z=right.sample(4,g2,state)[0]
            self.assertTrue(torch.equal(expected,torch.cat([a,z],2)),variant)
            self.assertLessEqual(float(whole.velocities.norm(dim=-1).max()),3.00001)

    def test_distribution_adapters_preserve_v2_means_and_observe(self):
        for variant in VARIANTS:
            model=PhysicalWorldModel(variant=variant,noise='integrated')
            model.core.acceleration[-1].weight.data.normal_(0,.02)
            converted=AgentScaleWorldModel(mean_variant=variant,conditional=True,age_aware=True)
            mismatch=converted.load_state_dict(model.state_dict(),strict=False)
            self.assertEqual(set(mismatch.missing_keys),{'raw_decay_rate','scale_values','scale_logits','scale_context.weight','scale_context.bias'})
            self.assertFalse(mismatch.unexpected_keys)
            expected=model.imagine(model.encode_history(legal()),horizon_s=1.).positions
            f=converted.imagine(converted.encode_history(legal()),horizon_s=1.)
            self.assertTrue(torch.equal(expected,f.positions),variant)
            observed=f.positions[:,0,-1].detach()+.2
            updated=converted.observe(f.next_belief,dict(positions=observed,observed=torch.ones(1,1,dtype=torch.bool),
                acquisition_times=torch.ones(1,1),timestamp_s=1.))
            self.assertLess(float((updated.core.positions-observed[:,None]).square().mean()),
                            float((f.next_belief.core.positions-observed[:,None]).square().mean()))

    def test_finite_useful_motion_gradients_and_causal_encoder(self):
        for variant in VARIANTS:
            model=PhysicalWorldModel(variant=variant,noise='integrated')
            f=model.imagine(model.encode_history(legal()),horizon_s=1.)
            truth=f.positions[:,0].detach()+torch.tensor([.2,.1]);mask=torch.ones(1,4,1,dtype=torch.bool)
            lp,count=trajectory_log_probability(f,truth,mask);es,_=energy_score(f,truth,mask)
            (es.mean()-.05*(lp/count).mean()).backward()
            grad=model.core.acceleration[-1].weight.grad
            self.assertTrue(torch.isfinite(grad).all(),variant);self.assertGreater(float(grad.norm()),0.,variant)
        gru=nn.GRU(13,16,batch_first=True);gru.forward=MethodType(arrival_forward,gru)
        x=torch.zeros(1,5,13);x[...,6]=1;x[...,10]=1;x[...,8]=torch.arange(5)/10
        y=x.clone();y[:,3:,:4]=1000
        self.assertTrue(torch.equal(gru(x)[0][:,:3],gru(y)[0][:,:3]))

    def test_occupancy_auxiliary_marginalizes_unsupported_agents(self):
        from eventforecastwm.performance_v2.train_occupancy import loss
        model=PosteriorForecastWorldModel(noise='integrated')
        data=legal();data=replace(data,humans=data.humans.repeat(1,2,1,1),human_padding=data.human_padding.repeat(1,2))
        truth=model.imagine(model.encode_history(data),horizon_s=1.).positions[:,0].detach()+.1
        mask=torch.ones(1,4,2,dtype=torch.bool);mask[:,:,1]=False
        batch=dict(base=data,geometry=torch.zeros(1,2,32,32),geometry_support=torch.zeros(1,32,32,dtype=torch.bool),
            history_timestamps=None,track_acquisition_times=None,sensor_config=None,robot_geometry=None,
            targets=dict(track_positions=truth,track_valid=mask))
        torch.manual_seed(31);value,metrics=loss(model,batch,horizon=1.)
        batch['targets']['track_positions']=truth.clone();batch['targets']['track_positions'][:,:,1]=10000
        torch.manual_seed(31);poisoned,other=loss(model,batch,horizon=1.)
        self.assertTrue(torch.equal(value,poisoned))
        self.assertEqual(metrics['occupancy_bce'],other['occupancy_bce'])
        value.backward()
        self.assertTrue(torch.isfinite(model.core.acceleration[-1].weight.grad).all())

    def test_fresh_position_fixed_but_estimated_velocity_can_be_revised(self):
        model=PhysicalWorldModel(variant='posterior_velocity_v2',noise='integrated')
        initial=model.encode_history(legal())
        model.core.posterior_head[-1].bias.data.fill_(.5)
        corrected=model.encode_history(legal())
        self.assertTrue(torch.equal(initial.core.positions,corrected.core.positions))
        self.assertGreater(float((initial.core.velocities-corrected.core.velocities).abs().max()),.1)
        loss=(corrected.core.velocities-initial.core.velocities.detach()).square().mean()
        loss.backward()
        self.assertGreater(float(model.core.posterior_head[-1].weight.grad.norm()),0.)

    def test_vectorized_conditional_density_matches_joint_likelihood(self):
        from eventforecastwm.performance_v2.conditional_coverage import sample_agent_density, masked_joint
        from eventforecastwm.distribution import trajectory_log_probability as log_probability
        from eventforecastwm.sampling import joint_samples
        model=AgentScaleWorldModel(mean_variant='posterior_v1',conditional=True,age_aware=True)
        data=legal();data=replace(data,humans=data.humans.repeat(2,3,1,1),
                                 human_padding=data.human_padding.repeat(2,3),
                                 robot=data.robot.repeat(2,1,1),history_padding=data.history_padding.repeat(2,1),
                                 query=data.query.repeat(2,1,1))
        f=model.imagine(model.encode_history(data),horizon_s=1.)
        paths,_=joint_samples(f,5,torch.Generator().manual_seed(91))
        complete=torch.tensor([[True,False,True],[True,True,False]])
        mask=complete[:,None].expand(-1,4,-1)
        density=sample_agent_density(f,paths,mask,chunk=3)
        actual=masked_joint(density,f.weights,complete)
        expected=torch.stack([log_probability(f,paths[:,i],mask)[0] for i in range(5)],1)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-5,rtol=1e-6))

    def test_old_window_sampling_preserves_expected_root_objective(self):
        from eventforecastwm.performance_v2.train_sampling import sampling_slots
        model=PosteriorForecastWorldModel(noise='integrated');prepared=model.core.prepare_history(legal())
        windows=[dict(_prepared=replace(prepared,last_observed_times=torch.full_like(prepared.last_observed_times,-age)),
                      value=value) for age,value in ((0.,2.),(.5,7.),(2.,20.))]
        slots,multiplicity=sampling_slots(windows)
        self.assertEqual(multiplicity,[1,1,4])
        expected=sum(w['value'] for w in windows)/len(windows)
        weighted=sum(w['value']*w['_sampling_importance'] for w in slots)/len(slots)
        self.assertAlmostEqual(expected,weighted)
        self.assertAlmostEqual(sum(w['_sampling_importance'] for w in slots)/len(slots),1.)
        self.assertTrue(all('_sampling_importance' not in w for w in windows))

    def test_anonymous_readout_uses_only_legal_context_and_preserves_grid_axes(self):
        from eventforecastwm.performance_v2.occupancy_prior import predict
        model=PosteriorForecastWorldModel(noise='integrated');data=legal()
        prepared=model.core.prepare_history(data)
        batch=dict(base=data,prepared_history=prepared,geometry=torch.zeros(1,2,32,32),
                   geometry_support=torch.ones(1,32,32,dtype=torch.bool),targets={'poison':float('nan')})
        batch['geometry'][:,1,3,7]=1;batch['geometry'][:,0,3,7]=1
        exposure=torch.ones(16,4,20,32,32)*100
        positive=torch.zeros_like(exposure);positive[:,3]=20;positive[:,1]=10
        tables=dict(positive=positive,exposure=exposure,prior=torch.full((20,32,32),.01))
        first=predict(tables,batch,strength=100.)
        self.assertAlmostEqual(float(first[0,0,3,7]),.105,places=6)
        self.assertAlmostEqual(float(first[0,0,7,3]),.055,places=6)
        batch['targets']={'changed_future':torch.randn(99,7)}
        batch['base']=replace(data,query=torch.full_like(data.query,float('nan')))
        self.assertTrue(torch.equal(first,predict(tables,batch,strength=100.)))
        map_only=predict(tables,batch,history_context=False)
        batch['prepared_history']=replace(prepared,valid=torch.zeros_like(prepared.valid))
        self.assertTrue(torch.equal(map_only,predict(tables,batch,history_context=False)))

    def test_actual_arrival_adapter_preserves_frame_time_and_legal_identity(self):
        import math
        from eventforecastwm.performance_v2.recurrent_audit import arrived_observation
        model=PosteriorForecastWorldModel(noise='integrated');data=legal()
        prepared=model.core.prepare_history(data);core=model.imagine(model.encode_history(data),horizon_s=1.).next_belief.core
        new_prepared=replace(prepared,positions=torch.tensor([[[999.,999.],[1.,0.]]]),
            valid=torch.ones(1,2,dtype=torch.bool),last_observed_times=torch.zeros(1,2,dtype=torch.float64))
        old=dict(metadata=dict(track_ids=['legal'],timestamp=0.,origin=[0.,0.,0.],yaw=0.))
        new=dict(metadata=dict(track_ids=['different_ID','legal'],timestamp=.9,origin=[1.,0.,2.],yaw=math.pi/2),
                 _prepared=new_prepared,targets={'forbidden':float('nan')})
        observation=arrived_observation(old,new,core)
        self.assertTrue(observation['observed'][0,0])
        self.assertTrue(torch.allclose(observation['positions'],torch.tensor([[[1.,1.]]]),atol=1e-6))
        self.assertAlmostEqual(float(observation['acquisition_times'][0,0]),.9)
        new['_prepared']=replace(new_prepared,last_observed_times=torch.full((1,2),-.3,dtype=torch.float64))
        self.assertFalse(arrived_observation(old,new,core)['observed'].any())
        new['metadata']['track_ids']=['different_ID','new_ID']
        self.assertFalse(arrived_observation(old,new,core)['observed'].any())

    def test_confirmation_membership_excludes_all_parent_physical_identities(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from eventbeliefwm.data import sha256
        from eventforecastwm.performance_v2.confirmation_population import select
        def block(name,split,scene,episode=None):
            row=dict(root_id=name,episode_id=episode or name,fixture_id='f'+name,trajectory_id='t'+name,
                scene=scene,category='synthetic',dataset='/test',branch_id='b0',query_id='q'+name,
                sr_valid=True,sr_label=1,observed_duration_s=3.,human_collision_time_valid=False,human_collision_time_s=0.)
            return dict(root=dict(root_id=name,physical_scene_id=scene,seal_file=name+'.seal'),split=split,rows=[row],
                archive_path='/test/'+name,seal_sha256='test',complete_sha256='test',evidence={})
        with TemporaryDirectory() as temporary:
            path=Path(temporary);parent=dict(blocks=[block('old','train','training_scene')])
            parent_path=path/'parent.json';parent_path.write_text(json.dumps(parent))
            blocks=list(parent['blocks'])
            for bank in ('natural_reference','unseen_evaluation'):
                scene='training_scene' if bank=='natural_reference' else 'unseen_scene'
                blocks.extend(block(bank+str(i),bank,scene) for i in range(64))
                blocks.append(block(bank+'overlap',bank,scene,episode='old'))
            snapshot=path/'snapshot.json';snapshot.write_text(json.dumps(dict(blocks=blocks)))
            (path/'snapshot_identity.json').write_text(json.dumps(dict(sha256=sha256(snapshot),parent_snapshot_sha256=sha256(parent_path))))
            out=path/'membership';select(dict(run=str(out),snapshot=str(snapshot),parent_snapshot=str(parent_path)))
            receipt=json.loads((out/'membership.json').read_text())
            self.assertEqual(receipt['count_per_bank'],64)
            self.assertEqual(sum(map(len,receipt['excluded_physical_overlap'].values())),2)
            self.assertTrue(all('overlap' not in name for rows in receipt['banks'].values() for name in rows))

    def test_hybrid_keeps_physical_means_and_both_forecast_clocks(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from eventforecastwm.benchmark import load_model
        from eventforecastwm.performance_v2.hybrid import OccupancyWorldModel
        physical=AgentScaleWorldModel(mean_variant='posterior_v1',age_aware=True)
        model=OccupancyWorldModel(physical.recipe);model.physical.load_state_dict(physical.state_dict())
        for category in range(4):
            model.anonymous_probabilities[category]=category*.1+torch.arange(1,21)[:,None,None]*.001
        data=legal();geometry=dict(features=torch.ones(1,2,32,32),support=torch.ones(1,32,32,dtype=torch.bool))
        belief=model.encode_history(data,geometry);whole=model.imagine(belief,horizon_s=3.)
        expected=physical.imagine(physical.encode_history(data),horizon_s=3.)
        self.assertTrue(torch.equal(expected.positions,whole.positions))
        left=model.imagine(belief,torch.full((1,5),float('nan')),horizon_s=1.)
        right=model.imagine(left.next_belief,horizon_s=2.)
        self.assertTrue(torch.equal(whole.positions[:,:,4:],right.positions))
        self.assertTrue(torch.equal(whole.anonymous_occupancy[:,4:],right.anonymous_occupancy))
        self.assertAlmostEqual(float(left.anonymous_occupancy[0,-1,0,0]),.304,places=6)
        g1=torch.Generator().manual_seed(4);g2=torch.Generator().manual_seed(4)
        all_paths=whole.sample(3,g1)[0];prefix,_,sample=left.sample(3,g2);suffix=right.sample(3,g2,sample)[0]
        self.assertTrue(torch.equal(all_paths,torch.cat([prefix,suffix],2)))
        observation=dict(positions=left.positions[:,0,-1]+.2,observed=torch.ones(1,1,dtype=torch.bool),
                         acquisition_times=torch.ones(1,1),timestamp_s=1.)
        updated=model.observe(left.next_belief,observation)
        self.assertTrue(torch.equal(model.imagine(updated,horizon_s=2.).anonymous_occupancy,right.anonymous_occupancy))
        beyond=model.imagine(whole.next_belief,horizon_s=3.)
        self.assertTrue(beyond.anonymous_occupancy_supported[:,:8].all())
        self.assertFalse(beyond.anonymous_occupancy_supported[:,8:].any())
        self.assertTrue(torch.equal(beyond.anonymous_occupancy[:,8:],torch.zeros_like(beyond.anonymous_occupancy[:,8:])))
        with TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';torch.save(dict(recipe=model.recipe,model=model.state_dict()),path)
            restored=load_model(path).eval();actual=restored.imagine(restored.encode_history(data,geometry),horizon_s=3.)
            self.assertTrue(torch.equal(actual.positions,whole.positions))
            self.assertTrue(torch.equal(actual.anonymous_occupancy,whole.anonymous_occupancy))


if __name__=='__main__':unittest.main()
