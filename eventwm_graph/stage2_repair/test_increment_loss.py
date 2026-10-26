import unittest
import numpy as np
import torch
from .increment_loss import micro_dice


class IncrementLoss(unittest.TestCase):
    def test_hard_predictions_match_independent_weighted_confusion(self):
        truth=torch.tensor([0,0,0,1,1,1,2,2,2]);prediction=torch.tensor([0,1,2,0,1,2,0,1,2]);weight=torch.tensor([1.,3.,2.,5.,4.,7.,6.,8.,9.])
        r=(prediction>0).double();b=(prediction==2).double();w=weight.double();cm=np.zeros((3,3))
        for y,p,v in zip(truth,prediction,weight):cm[int(y),int(p)]+=float(v)
        correct=cm[1,1]+cm[2,2];true=cm[1:,:].sum();predicted=cm[:,1:].sum();iou=correct/(true+predicted-correct)
        self.assertAlmostEqual(float(micro_dice(r,b,truth,w)),2*(1-2*iou/(1+iou)),places=7)

    def test_empty_target_penalizes_false_increment_and_has_gradient(self):
        logits=torch.tensor([-2.,1.],requires_grad=True);r=logits.sigmoid();y=torch.zeros(2,dtype=torch.long);loss=micro_dice(r,torch.full((2,),.5),y,torch.ones(2));loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all());self.assertTrue((logits.grad>0).all())
        self.assertEqual(float(micro_dice(torch.zeros(2),torch.zeros(2),y,torch.ones(2))),0.)


if __name__=='__main__':unittest.main()
