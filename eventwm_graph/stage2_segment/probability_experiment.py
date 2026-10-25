"""Unpromoted GPU probability experiment with tolerance-bounded floats.

The same 1e-4 logit near-tie guard and complete CPU reference blocks remain.
Only softmax execution moves to CUDA; no model, decision or domain changes.
"""
import time
import numpy as np
import torch
from numba import set_num_threads
from .api import SegmentPredictor
from .cuda import critical_blocks
from .fused import transform
from eventwm_graph.stage2.common import SCHEMA as GRAPH_SCHEMA


def forward_probabilities(execution,x,context,keys,baseline):
    start=time.perf_counter();values=np.empty((len(keys),4),np.float32);p=np.empty((len(keys),3),np.float32)
    with torch.inference_mode(),torch.cuda.amp.autocast(enabled=False),torch.cpu.amp.autocast(enabled=False):
        for lo in range(0,len(keys),65536):
            hi=min(lo+65536,len(keys));data=x[lo:hi]
            if not isinstance(data,torch.Tensor):data=torch.from_numpy(data).to(execution.device)
            logits=execution.net(data);probability=torch.softmax(logits[:,:3],dim=-1)
            values[lo:hi]=logits.cpu().numpy();p[lo:hi]=probability.cpu().numpy()
        returned=time.perf_counter();blocks=critical_blocks(values)
        for block in blocks:
            lo=int(block)*8192;hi=min(lo+8192,len(keys))
            exact,_=transform(context,keys[lo:hi],end_only=x.shape[1]==80)
            original=execution.reference(torch.from_numpy(exact))
            values[lo:hi]=original.numpy();p[lo:hi]=torch.softmax(original[:,:3],dim=-1).numpy()
    reference_done=time.perf_counter();height=np.add(baseline,values[:,3],dtype=np.float32)
    if not np.isfinite(values).all() or not np.isfinite(p).all() or not np.isfinite(height).all():raise ValueError('nonfinite neural probability output')
    execution.last_profile.update(neural_transfer_forward_probability_return_seconds=returned-start,
        near_tie_reference_blocks=len(blocks),near_tie_reference_seconds=reference_done-returned,
        probability_final_validation_seconds=time.perf_counter()-reference_done,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(execution.device),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(execution.device),
        probability_backend='experimental_cuda_softmax_with_original_reference_guard')
    return p,height


class GpuProbabilityPredictor(SegmentPredictor):
    def _predict(self,graph,trajectory,prepared):
        if self.execution!='optimized' or self.cuda is None or self.feature_count not in (96,104):
            return super()._predict(graph,trajectory,prepared)
        set_num_threads(self.feature_threads);self.cuda.last_profile.clear()
        start=time.perf_counter();context=self.feature_class(graph,trajectory,prepared);spatial=time.perf_counter()
        keys=context.candidates('both');candidates=time.perf_counter()
        x,baseline=transform(context,keys,end_only=self.feature_count==96,cuda=self.cuda,device_output=True)
        features=time.perf_counter();transform_profile=dict(context.last_transform_profile)
        p,height=forward_probabilities(self.cuda,x,context,keys,baseline);inferred=time.perf_counter()
        if len(context.path)==1:p.fill(0);p[:,0]=1
        states=p.argmax(-1).astype(np.uint8);now=float(context.path[-1,0]);selected=np.flatnonzero(states)
        updates=[dict(cell=keys[i].tolist(),state=int(states[i]),height=float(height[i]),timestamp=now,source='predicted_stage2_research') for i in selected]
        delta=dict(schema=GRAPH_SCHEMA,predictor_schema='trajectory_conditioned_graphdelta_quality_2',predicted=True,
            base_graph_identity=prepared.identity,base_timestamp=float(graph['timestamp']),timestamp=now,robot_pose=context.path[-1,1:].tolist(),
            static_updates=updates,candidate_cells=keys,probabilities=p,states=states,heights=height,unknown_cells=keys[states==0],
            outside_domain='UNKNOWN_NO_UPDATE',checkpoint_sha256=self.checkpoint_sha256)
        self.last_profile=dict(spatial_seconds=spatial-start,candidate_seconds=candidates-spatial,feature_seconds=features-candidates,
            neural_seconds=inferred-features,delta_seconds=time.perf_counter()-inferred,total_seconds=time.perf_counter()-start,**transform_profile)
        self.last_profile.update(self.cuda.last_profile)
        return self._with_base(delta,prepared)
