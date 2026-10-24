"""CPU/batched research API. Predictions never become actual sensor evidence."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from eventwm_graph.partial.map import PartialGraph
from .common import SCHEMA,sha
from .features import SceneFeatures
from .model import GraphDeltaNet


def graph_identity(graph):
    rows=sorted((c['cell'][0],c['cell'][1],c['state'],float(c['height'])) for c in graph['cells'])
    raw=np.asarray(rows,np.float64).reshape(-1,4).tobytes()
    h=hashlib.sha256(raw+np.asarray([graph['timestamp'],graph['revision'],graph['resolution'],graph['height'],graph['clearance']],np.float64).tobytes())
    return h.hexdigest()


class Stage2Predictor:
    def __init__(self,checkpoint,threads=1):
        if threads is not None:torch.set_num_threads(threads)
        self.checkpoint=Path(checkpoint);saved=torch.load(checkpoint,map_location='cpu')
        if saved.get('schema')!=SCHEMA:raise ValueError('unsupported Stage2 checkpoint')
        self.model=GraphDeltaNet(**saved['model_config']);self.model.load_state_dict(saved['model']);self.model.eval()
        self.checkpoint_sha256=sha(checkpoint)
        self.config=saved['config']

    def predict(self,*,partial_graph,trajectory):
        return self.predict_batch([dict(partial_graph=partial_graph,trajectory=trajectory)])[0]

    def predict_batch(self,inputs):
        if not inputs:return []
        contexts=[];keys=[];features=[];baselines=[]
        for item in inputs:
            context=SceneFeatures(item['partial_graph'],item['trajectory']);candidate=context.candidates()
            x,baseline=context.transform(candidate)
            contexts.append(context);keys.append(candidate);features.append(x);baselines.append(baseline)
        lengths=[len(x) for x in features];x=np.concatenate(features,axis=0)
        with torch.inference_mode():
            logits,height=self.model(torch.from_numpy(x));prob=torch.softmax(logits,dim=-1).numpy();height=height.numpy()
        result=[];begin=0
        for context,candidate,n,baseline in zip(contexts,keys,lengths,baselines):
            p=prob[begin:begin+n].copy();h=(baseline+height[begin:begin+n]).copy();begin+=n
            if len(context.path)==1:
                p.fill(0);p[:,0]=1
            labels=p.argmax(-1).astype(np.uint8);now=float(context.path[-1,0])
            selected=np.flatnonzero(labels)
            updates=[dict(cell=candidate[i].tolist(),state=int(labels[i]),height=float(h[i]),timestamp=now,source='predicted_stage2_research') for i in selected]
            result.append(dict(schema=SCHEMA,predicted=True,base_graph_identity=graph_identity(context.graph),
                base_timestamp=float(context.graph['timestamp']),timestamp=now,robot_pose=context.path[-1,1:].tolist(),
                static_updates=updates,candidate_cells=candidate,probabilities=p,states=labels,heights=h,
                unknown_cells=candidate[labels==0],outside_domain='UNKNOWN_NO_UPDATE',
                checkpoint_sha256=self.checkpoint_sha256))
        return result


def apply_graph_delta(current_graph,predicted_delta):
    """Construct a hypothetical graph through the existing deterministic updater.

    The caller graph is unchanged. `predicted_stage2_research` provenance and
    the top-level hypothetical flag distinguish it from legally observed data.
    """
    if predicted_delta.get('schema')!=SCHEMA or predicted_delta.get('predicted') is not True:raise ValueError('invalid predicted GraphDelta')
    if predicted_delta['base_graph_identity']!=graph_identity(current_graph):raise ValueError('GraphDelta base graph mismatch')
    graph=PartialGraph.restore(current_graph);now=float(predicted_delta['timestamp'])
    if now<graph.timestamp:raise ValueError('GraphDelta clock regressed')
    updates=predicted_delta['static_updates']
    seen=set()
    for row in updates:
        key=tuple(row['cell'])
        if key in seen or graph.state(key)!=0 or row['state'] not in (1,2) or row['timestamp']!=now or not np.isfinite(row['height']):raise ValueError('delta must update unique initially UNKNOWN cells only')
        seen.add(key)
    if now==graph.timestamp:
        if updates:raise ValueError('zero-duration trajectory cannot reveal cells')
    else:
        graph.update(dict(timestamp=now,robot_pose=predicted_delta['robot_pose'],static_updates=updates,tracks=[]))
    result=graph.snapshot();result['hypothetical']=True;result['prediction_checkpoint_sha256']=predicted_delta['checkpoint_sha256']
    return result
