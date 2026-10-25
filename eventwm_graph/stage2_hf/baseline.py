"""Strong released baseline with one actual thresholded future graph per call."""
import numpy as np
from eventwm_graph.stage2_segment.api import SegmentPredictor

class ThresholdBaseline(SegmentPredictor):
 def __init__(self,*args,threshold=.2,**kwargs):super().__init__(*args,**kwargs);self.threshold=float(threshold)
 def _predict(self,graph,trajectory,prepared):
  delta=super()._predict(graph,trajectory,prepared);p=delta['probabilities'];keys=delta['candidate_cells'];h=delta['heights'];r=(p[:,1]+p[:,2]).astype(np.float64)
  states=np.where(r>=self.threshold,np.where(p[:,1]>=p[:,2],1,2),0).astype(np.uint8);now=delta['timestamp'];chosen=np.flatnonzero(states)
  delta['states']=states;delta['unknown_cells']=keys[states==0];delta['static_updates']=[dict(cell=keys[i].tolist(),state=int(states[i]),height=float(h[i]),timestamp=now,source='predicted_stage2_research')for i in chosen]
  return delta
 def _complete(self,graph,delta,trajectory):
  elapsed=super()._complete(graph,delta,trajectory);delta['provenance'].update(decision='revelation_threshold_then_conditional_label',revelation_threshold=self.threshold);return elapsed
