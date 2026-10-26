"""Differentiable typed-increment overlap, with original inclusion weights."""
import torch


def micro_dice(revelation,blocked,y,weight):
    """Twice one-minus typed F1, same scale as sum of two class Dice losses.

    Its hard-probability value is monotonic in the existing label-aware IoU.
    UNKNOWN true negatives do not improve overlap. Wrong FREE/BLOCKED labels
    contribute to both the true and predicted increment sizes.
    """
    correct=(revelation*torch.where(y==2,blocked,1-blocked)*(y>0)*weight).sum()
    size=((revelation+(y>0).float())*weight).sum()
    return 2*(1-(2*correct+1.e-7)/(size+1.e-7))
