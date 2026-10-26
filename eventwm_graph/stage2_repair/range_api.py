"""Legal measured-range sensor extension for complete Stage 2 inference.

There is deliberately no scene, navigation, simulator or future-camera
argument. Sensor acquisition belongs to the environment. Every map build,
feature, transfer, decode and normal hypothetical graph update is timed here.
"""
from collections import OrderedDict
import hashlib
import json
import time
import numpy as np
import torch
from .global_features import GlobalTimePredictor
from .range_gpu import GpuObservedRangeMesh
from .range_history import validate
from .range_evidence import EvidenceNet,features
from .diagnose import sha
from .range_identity import identity as range_identity


class RangeEvidencePredictor(GlobalTimePredictor):
    def __init__(self,checkpoint,*,device='cuda:0',threshold=None,fast_features=True,mesh_stride=1,**kwargs):
        saved=torch.load(checkpoint,map_location='cpu')
        if saved['schema']!='legal_range_evidence16_1':raise ValueError('unsupported measured-range checkpoint')
        super().__init__(saved['config']['point_checkpoint'],device=device,threshold=saved.get('threshold',.2)if threshold is None else threshold,increment_format='columns',**kwargs)
        inherited_source=self.config['source'];self.config=dict(saved['config'],source=inherited_source)
        self.model=EvidenceNet().to(device).eval();self.model.load_state_dict(saved['model'])
        self.checkpoint_sha256=sha(checkpoint);self.predictor_schema=saved['schema'];self.mesh_cache=OrderedDict();self.fast_features=fast_features
        self.decision_rule=saved.get('decision_rule','threshold')
        self.mesh_stride=int(mesh_stride)
        if self.mesh_stride not in (1,2):raise ValueError('only measured stride1/2 geometry is supported')
        if self.decision_rule not in ('threshold','argmax'):raise ValueError('unsupported range operating point')
        if fast_features:
            from .range_fast import RangeFeatureKernel
            self.range_kernel=RangeFeatureKernel(self.device)

    def clear_cache(self):
        super().clear_cache();self.mesh_cache.clear()

    def _projection(self,prepared,graph,path):return path

    def _make_mesh(self,ranges,origins,rotations,timestamps,now,resolution,tracks):
        return GpuObservedRangeMesh(ranges,origins,rotations,timestamps,now,resolution,self.device,stride=self.mesh_stride,tracks=tracks)

    def _decode(self,values,baseline,keys,no_future):
        from .decode import decode_columns
        return decode_columns(values,baseline,keys,self.threshold,no_future,self.decision_rule=='argmax')

    def _network(self,xs,details):
        if not any(len(x)for x in xs):return [np.empty((0,4),np.float32)for x in xs]
        if self.fast_features:
            extras=self.range_kernel.compute(self.active_mesh,details,self.active_graph.keys,self.current_origin,self.current_rotation)
            result=[]
            for x,extra in zip(xs,extras):
                values=[]
                for begin in range(0,len(x),self.tile_size):
                    end=min(begin+self.tile_size,len(x));p,h=self.model.probabilities(x[begin:end],extra[begin:end]);values.append(torch.cat([p,h[:,None]],1).cpu().numpy())
                result.append(np.concatenate(values)if values else np.empty((0,4),np.float32))
            return result
        out=[]
        for x,(keys,baseline,path)in zip(xs,details):
            if not len(keys):out.append(np.empty((0,4),np.float32));continue
            # Only the nearest legal-map height is needed by the range
            # extractor. Avoid a full feature transfer back to the host.
            stub=np.zeros((len(keys),25),np.float32);stub[:,24]=baseline-path[0,2]
            extra=features(self.active_mesh,keys,stub,path,self.active_graph.keys,self.current_origin,self.current_rotation)
            values=[]
            with torch.inference_mode():
                for begin in range(0,len(keys),self.tile_size):
                    end=min(begin+self.tile_size,len(keys));p,h=self.model.probabilities(x[begin:end],torch.from_numpy(extra[begin:end]).to(self.device));values.append(torch.cat([p,h[:,None]],1).cpu().numpy())
            out.append(np.concatenate(values)if values else np.empty((0,4),np.float32))
        return out

    def predict(self,*,partial_graph,trajectory,range_observations,track_history,materialize=True):
        return self.predict_segments(partial_graph=partial_graph,trajectories=[trajectory],range_observations=range_observations,track_history=track_history,materialize=materialize)[0]

    def predict_segments(self,*,partial_graph,trajectories,range_observations,track_history,materialize=True):
        tick=time.perf_counter();packet=range_observations
        if packet.get('sensor_protocol')!='ideal_current_jaw_range_dynamic_occlusion_1':raise ValueError('only measured Euclidean 90-degree range protocol is validated')
        ranges=np.asarray(packet['ranges'],np.float32);origins=np.asarray(packet['origins'],np.float64);rotations=np.asarray(packet['rotations'],np.float64);timestamps=np.asarray(packet['timestamps'],np.float64);now=float(partial_graph['timestamp'])
        validate(ranges,origins,rotations,timestamps,now)
        if abs(timestamps[-1]-now)>1.e-7:raise ValueError('current sensor packet must reach graph start time')
        if len(track_history)!=len(ranges):raise ValueError('legal track/range history alignment')
        paths=[np.asarray(p,np.float64)for p in trajectories]
        if not paths:return[]
        if any(p.ndim!=2 or p.shape[1]!=5 or not len(p)for p in paths):raise ValueError('supplied Nx5 trajectories required')
        if any(not np.allclose(p[0],paths[0][0],rtol=0,atol=1.e-7)for p in paths):raise ValueError('queries must share current starting pose')
        identity=range_identity(ranges,origins,rotations,timestamps,track_history,partial_graph['resolution'],self.mesh_stride)
        if identity not in self.mesh_cache:
            self.mesh_cache[identity]=self._make_mesh(ranges,origins,rotations,timestamps,now,partial_graph['resolution'],track_history)
            contract=self.config.get('observed_obstacle_schema','legacy_8cell_crop')
            if contract not in ('legacy_8cell_crop','capped_2m_complete'):raise ValueError('unsupported observed obstacle feature schema')
            if contract=='legacy_8cell_crop':self.mesh_cache[identity].obstacle_padding_cells=8
            while len(self.mesh_cache)>2:self.mesh_cache.popitem(last=False)
        self.mesh_cache.move_to_end(identity);self.active_mesh=self.mesh_cache[identity];self.active_graph=self.prepare_graph(partial_graph);self.current_origin=origins[-1];self.current_rotation=rotations[-1];mesh_at=time.perf_counter()
        result=super().predict_segments(partial_graph=partial_graph,trajectories=paths,materialize=materialize)
        for d in result:
            point_source=d['provenance'].pop('training_source_commit',None)
            d['provenance'].update(frozen_point_source_commit=point_source,training_source_commit=self.config.get('training_source_commit'),training_source_file_hashes=self.config.get('sources',{}))
            d['provenance'].update(input='current_legal_graph_supplied_base_path_and_measured_past_current_range_history',sensor_extension=True,privileged_runtime_inputs=False,range_history_identity=identity)
            d['provenance']['range_mesh_pixel_stride']=self.mesh_stride
            d['provenance']['observed_obstacle_schema']=self.config.get('observed_obstacle_schema','legacy_8cell_crop')
            d['provenance']['implementation']=type(self).__name__
            d['provenance']['range_feature_kernel']=type(self.range_kernel).__name__
            if self.decision_rule=='argmax':d['provenance'].update(decision='argmax',revelation_threshold=None)
        self.last_batch_profile.update(range_input_and_mesh_seconds=mesh_at-tick,total_seconds=time.perf_counter()-tick,serialization_included=False,sensor_acquisition_included=False)
        return result
