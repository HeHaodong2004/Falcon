"""Fixed-size CUDA graph for the same neural head; padding is discarded.

Capturing/warming belongs to constructor startup. Map features, original
candidate cells, final probabilities and hypothetical graph outputs remain
inside normal complete inference. No output universe is reduced.
"""
import torch
from .device_pipeline import FastPreparedPipelinePredictor
from .range_bev import SIZE


class CapturedSpatialHead:
    def __init__(self,model,device,rows=65536):
        self.rows=int(rows);self.device=torch.device(device)
        with torch.cuda.device(self.device),torch.inference_mode():
            self.x=torch.zeros((rows,88),device=device);self.extra=torch.zeros((rows,16),device=device);self.indices=torch.zeros((rows,2),device=device,dtype=torch.long);self.ids=torch.zeros(rows,device=device,dtype=torch.long)
            self.encoded=(torch.zeros((1,24,SIZE,SIZE),dtype=torch.float16,device=device),torch.zeros((1,16),dtype=torch.float16,device=device));stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream),torch.cuda.amp.autocast():
                for _ in range(3):model.probabilities(self.x,self.extra,self.indices,self.ids,self.encoded)
            torch.cuda.current_stream(device).wait_stream(stream);self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream),torch.cuda.amp.autocast():
                p,h=model.probabilities(self.x,self.extra,self.indices,self.ids,self.encoded);self.output=torch.cat([p.float(),h.float()[:,None]],1)
            torch.cuda.current_stream(device).wait_stream(stream)

    def run(self,x,extra,indices,encoded,baseline):
        n=len(x);values=torch.empty((n,5),device=self.device);self.encoded[0].copy_(encoded[0]);self.encoded[1].copy_(encoded[1])
        for begin in range(0,n,self.rows):
            end=min(begin+self.rows,n);count=end-begin;self.x[:count].copy_(x[begin:end]);self.extra[:count].copy_(extra[begin:end]);self.indices[:count].copy_(indices[begin:end]);self.graph.replay();values[begin:end,:4].copy_(self.output[:count]);values[begin:end,4].copy_(baseline[begin:end])
        return values


class CapturedPipelinePredictor(FastPreparedPipelinePredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.captured_head=CapturedSpatialHead(self.model,self.device,self.tile_size)

    def _probability_tiles(self,x,extra,indices,encoded,baseline,**kwargs):return self.captured_head.run(x,extra,indices,encoded,baseline)
