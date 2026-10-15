"""Separate masks for frozen SR semantics and privileged successor supervision."""
import torch
from torch.nn import functional as F


def sr_loss(logits, target, valid):
    if ((target[valid] != 0) & (target[valid] != 1)).any():
        raise ValueError('Unmasked SR must be zero or one')
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid].float()) if valid.any() else logits.sum()*0


def outcome_loss(log_probabilities, target, valid):
    if ((target[valid] < 0) | (target[valid] > 2)).any():
        raise ValueError('Unmasked outcome must be class 0, 1 or 2')
    return F.nll_loss(log_probabilities[valid],target[valid].long()) if valid.any() else log_probabilities.sum()*0


def physical_loss(probability, target, mask):
    """Mean of active horizon means. No arbitrary positive weighting changes priors."""
    losses = []
    for h in range(probability.shape[1]):
        active = mask[:, h].bool()
        if active.any():
            losses.append(F.binary_cross_entropy(probability[active, h], target[active, h].float()))
    return torch.stack(losses).mean() if losses else probability.sum()*0
