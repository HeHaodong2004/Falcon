"""CPU/batched research API. Predictions never become actual sensor evidence."""
import hashlib
import copy
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
        # Rows are independent: stream inputs and tile the same network to
        # bound temporary activations and avoid a batch-sized feature copy.
        result=[]
        for item in inputs:
            context=SceneFeatures(item['partial_graph'],item['trajectory']);candidate=context.candidates()
            x,baseline=context.transform(candidate)
            n=len(candidate);p=np.empty((n,3),np.float32);h=baseline.copy()
            with torch.inference_mode():
                for begin in range(0,n,8192):
                    end=min(begin+8192,n)
                    logits,height=self.model(torch.from_numpy(x[begin:end]))
                    p[begin:end]=torch.softmax(logits,dim=-1).numpy()
                    h[begin:end]+=height.numpy()
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


def owned_snapshot(graph):
    """Serialize a private updater instance without copying owned data twice.

    restore() already deep-copies visited/global structures. Canonical cell
    values are immutable scalars; preserve deepcopy semantics for extensions.
    The private graph is never retained or exposed after this transfer.
    """
    cells=[]
    for key,value in sorted(graph.cells.items()):
        row={'cell':list(key)}
        row.update((k,v if isinstance(v,(str,int,float,bool,type(None))) else copy.deepcopy(v)) for k,v in value.items())
        cells.append(row)
    return dict(schema='causal_partial_cspace_graph_1',resolution=graph.resolution,height=graph.height,clearance=graph.clearance,
        timestamp=graph.timestamp,revision=graph.revision,ego_support=graph.ego_support.tolist() if graph.ego_support is not None else None,
        cells=cells,visited=graph.visited,global_nodes=graph.global_nodes,global_edges=graph.global_edges)


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
    result=owned_snapshot(graph);result['hypothetical']=True;result['prediction_checkpoint_sha256']=predicted_delta['checkpoint_sha256']
    return result
