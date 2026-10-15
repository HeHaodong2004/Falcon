import unittest
import torch
from eventforecastwm.model_student import StudentWorldModel
from eventforecastwm.distribution import trajectory_log_probability
from test_forecast_model import legal


class TestStudentDamped(unittest.TestCase):
    def test_persistent_joint_scale_and_actual_update(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        model=StudentWorldModel(damped=True,age_aware=True);belief=model.encode_history(legal())
        full=model.imagine(belief,horizon_s=3.);left=model.imagine(belief,horizon_s=1.)
        right=model.imagine(left.next_belief,horizon_s=2.)
        a=torch.Generator().manual_seed(1);b=torch.Generator().manual_seed(1)
        paths,_,_=full.sample(3,a);p,_,state=left.sample(3,b);q,_,_=right.sample(3,b,state)
        self.assertTrue(torch.equal(paths,torch.cat([p,q],2)))
        target=full.positions[:,0].detach()+.2;mask=torch.ones(1,12,1,dtype=torch.bool)
        lp,count=trajectory_log_probability(full,target,mask);(-lp/count).mean().backward()
        self.assertGreater(float(model.raw_decay_rate.grad.abs()),0.)
        self.assertGreater(float(model.raw_df.grad.abs()),0.)
        before=left.next_belief;after=model.observe(before,dict(source='actual_arrived',
            positions=before.core.positions.mean(1).detach()+.1,observed=torch.ones(1,1,dtype=torch.bool),
            acquisition_times=torch.ones(1,1),timestamp_s=1.))
        self.assertTrue((after.concentration>before.concentration).all())
        self.assertTrue(torch.isfinite(model.imagine(after,horizon_s=.25).positions).all())


if __name__=='__main__':unittest.main()
