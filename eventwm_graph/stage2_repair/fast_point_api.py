"""Explicit FP16 linear approximation for frozen priors; F32 logit arithmetic.

Weights, map/query features, candidate cells and graph outputs are unchanged.
This is a numerical approximation, requiring matched reconstruction measurement.
"""
import torch
from torch.nn import functional as F
from .range_evidence import EvidenceNet
from .range_spatial import SpatialEvidenceNet
from .device_pipeline import ColumnPreparedPipelinePredictor


class FastPointEvidence(EvidenceNet):
    def raw(self,x,extra):
        # Preserve the interface's original F32 height estimates exactly.
        # FP16 prior GEMMs changed a few heights with query batch size.
        with torch.cuda.amp.autocast(enabled=False):base=self.point.prior(x.float())
        with torch.cuda.amp.autocast():
            # The legal-range model always supplies zero recency inputs here.
            # Avoid allocating32 zero columns for every original candidate.
            first=self.point.correction[0]
            hidden=F.linear(x,first.weight[:,:88],first.bias)
            for layer in self.point.correction[1:]:hidden=layer(hidden)
            delta=self.head(torch.cat([x,extra],1))
        base=base.float();hidden=hidden.float();delta=delta.float()
        r=base[:,1:3].logsumexp(-1)-base[:,0]+hidden[:,0]+delta[:,0]
        b=base[:,2]-base[:,1]+(hidden[:,1]if self.point.train_occupancy else 0)+delta[:,1]
        return r,b,base[:,3]


class FastPointSpatial(SpatialEvidenceNet):
    def __init__(self):
        super().__init__();self.point=FastPointEvidence()
        for p in self.point.parameters():p.requires_grad_(False)


class FastPointPredictor(ColumnPreparedPipelinePredictor):
    model_class=FastPointSpatial
    point_precision='F32 prior/heights; FP16 time/range corrections and spatial encoder/head; F32 logit math; TF32off'
    def predict_segments(self,**kwargs):
        results=super().predict_segments(**kwargs)
        for d in results:d['provenance']['frozen_point_precision']='Original F32 prior/heights; FP16 point corrections; F32 logit arithmetic; no zero recency columns'
        return results
