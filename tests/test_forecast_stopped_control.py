from dataclasses import fields,replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from eventforecastwm.model_agent_scale import AgentScaleWorldModel, trajectory_log_probability, occupancy
from eventforecastwm.performance_v2.stopped_control import (
    stopped_forecast, branch_density, joint_log_probability, mixed_paths, mixed_occupancy)
from test_forecast_model import legal


class TestStoppedControl(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(72)
        model=AgentScaleWorldModel(age_aware=True)
        self.belief=model.encode_history(legal());self.f=model.imagine(self.belief,horizon_s=1.)
        self.stopped=stopped_forecast(self.f,self.belief)

    def test_exact_branch_limits_and_target_mask(self):
        f=self.f;truth=f.positions[:,0].detach()+.1;mask=torch.ones_like(truth[...,0],dtype=torch.bool)
        density,count=branch_density(f,self.stopped,truth,mask)
        for index,forecast in enumerate((f,self.stopped)):
            probability=torch.zeros(1,1,2);probability[...,index]=1
            expected,_=trajectory_log_probability(forecast,truth,mask)
            actual=joint_log_probability(density,count,f.weights,probability)
            self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.allclose(mixed_occupancy(f,self.stopped,probability),occupancy(forecast),atol=1e-7))
        masked=torch.zeros_like(mask)
        d,c=branch_density(f,self.stopped,truth*1000,masked)
        self.assertTrue(torch.allclose(joint_log_probability(d,c,f.weights,torch.full((1,1,2),.5)),torch.zeros(1),atol=1e-6))

    def test_stopped_branch_and_noise_are_persistent(self):
        f=self.f;paths,_,_=f.sample(8,torch.Generator().manual_seed(9))
        mixed=mixed_paths(paths,f,self.stopped,torch.tensor([[[.5,.5]]]),torch.Generator().manual_seed(5))
        moving=f.positions[:,:,None].expand(-1,-1,8,-1,-1,-1).reshape_as(paths)
        stop=self.stopped.positions[:,:,None].expand(-1,-1,8,-1,-1,-1).reshape_as(paths)
        delta=mixed-paths;shift=stop-moving
        keep=(delta.abs().sum((2,3,4))<1e-6)
        change=((delta-shift).abs().sum((2,3,4))<1e-5)
        self.assertTrue((keep|change).all());self.assertTrue(keep.any());self.assertTrue(change.any())
        self.assertTrue(torch.equal(self.stopped.positions[:,:,0],self.stopped.positions[:,:,-1]))
        self.assertTrue(torch.equal(self.stopped.positions[:,0,0],self.belief.core.last_observed_positions))

    def test_independent_agent_branches_mix_before_union(self):
        f=self.f
        # Duplicate a legal agent to expose the distinction from a scene branch.
        fields={k:v.repeat_interleave(2,dim) for k,v,dim in (
            ('positions',f.positions,3),('velocities',f.velocities,3),('valid',f.valid,1),
            ('covariance',f.covariance,3),('agent_scale_probabilities',f.agent_scale_probabilities,2))}
        pair=replace(f,**fields);stopped=replace(pair,positions=pair.positions+3.)
        p=torch.full((1,2,2),.5)
        actual=mixed_occupancy(pair,stopped,p)
        from eventforecastwm.performance_v2.stopped_control import agent_cell_mass
        a=agent_cell_mass(pair);b=agent_cell_mass(stopped)
        expected=((1-(1-.5*a-.5*b).prod(3))*pair.weights[:,:,None,None,None]).sum(1)
        scene_mixture=.5*occupancy(pair)+.5*occupancy(stopped)
        # Alternative subtraction grouping differs by one float32 ULP.
        self.assertTrue(torch.allclose(actual,expected,atol=2e-7))
        self.assertGreater(float((actual-scene_mixture).abs().max()),1e-4)

    def test_density_build_resumes_verified_chunks_exactly(self):
        from eventforecastwm.performance_v2 import stopped_control as control
        model=AgentScaleWorldModel(age_aware=True).eval();base=legal();ids=[f'root{i:03d}' for i in range(128)]
        class Guard:
            def __init__(self,fail=None):self.calls=0;self.fail=fail
            def check(self):
                self.calls+=1
                if self.calls==self.fail:raise RuntimeError('Research stopped: test pause')
        def batch(windows,device):
            size=len(windows)
            data=replace(base,**{f.name:getattr(base,f.name).repeat(size,*([1]*(getattr(base,f.name).ndim-1)))
                for f in fields(base) if isinstance(getattr(base,f.name),torch.Tensor)})
            return dict(base=data,targets=dict(track_positions=torch.zeros(size,12,1,2),track_valid=torch.ones(size,12,1,dtype=torch.bool)))
        def roots(manifest,selected,guard):return [dict(root_id=rid,windows=[dict(root_id=rid)]) for rid in selected]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol=root/'protocol.json';protocol.write_text(json.dumps(dict(subsets={'full':ids},development=[])))
            interrupted=root/'interrupted';direct=root/'direct';interrupted.mkdir();direct.mkdir()
            cfg=dict(run=str(interrupted),protocol=str(protocol),manifest='test',prepared_cache='test',fit_scale='full',steps=2)
            identity=dict(experiment='synthetic exact cache recovery')
            with patch.object(control,'load_roots',side_effect=roots) as loader,patch.object(control,'prepare'),\
                 patch.object(control,'prepared_batch',side_effect=batch),patch.object(control,'encode',side_effect=lambda m,b:m.encode_history(b['base'])):
                with self.assertRaisesRegex(RuntimeError,'test pause'):control.fit(cfg,model,Guard(fail=2),identity)
                self.assertTrue((interrupted/'density_parts/000000.pt').exists())
                self.assertFalse((interrupted/'densities.pt').exists());loader.reset_mock()
                resumed,_=control.fit(cfg,model,Guard(),identity)
                self.assertEqual(loader.call_count,1);self.assertEqual(loader.call_args.args[1],ids[64:])
                expected,_=control.fit(dict(cfg,run=str(direct)),model,Guard(),identity)
            a=torch.load(interrupted/'densities.pt');b=torch.load(direct/'densities.pt')
            self.assertEqual(a['root_ids'],b['root_ids'])
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a['tensors'],b['tensors'])))
            self.assertTrue(all(torch.equal(value,expected.state_dict()[name]) for name,value in resumed.state_dict().items()))


if __name__=='__main__':unittest.main()
