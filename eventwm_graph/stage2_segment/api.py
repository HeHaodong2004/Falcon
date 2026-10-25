"""Complete CPU inference: legal input through a materialized future graph.

No execution is required. Poses can come from any compatible planner. A graph
is hypothetical and every new cell carries predicted provenance. Heights are
conditional point estimates, rather than a distribution over floor surfaces.
"""
import hashlib
import time
import numpy as np
import torch
from eventwm_graph.stage2_quality.api import Stage2Predictor as DeltaPredictor
from eventwm_graph.stage2.common import SCHEMA as GRAPH_SCHEMA
from .update import apply_graph_delta
from .common import SCHEMA


class SegmentPredictor(DeltaPredictor):
    def __init__(self, checkpoint, threads=1, cache_size=4, execution='reference', future_format='snapshot', device='cpu',feature_threads=1,shared_crops=True):
        super().__init__(checkpoint, threads=threads, cache_size=cache_size)
        if self.feature_count == 104:
            from .features import SegmentFeatures
            self.feature_class = SegmentFeatures
        elif self.feature_count == 96:
            from .features import SegmentEndFeatures
            self.feature_class = SegmentEndFeatures
        elif self.feature_count != 92:
            raise ValueError('unsupported segment feature configuration')
        if execution not in ('reference', 'optimized'):
            raise ValueError('unknown execution mode')
        self.execution = execution
        self.shared_crops=bool(shared_crops)
        if future_format not in ('snapshot','native'):raise ValueError('unknown future graph format')
        self.future_format=future_format
        if feature_threads not in (1,2):raise ValueError('feature_threads must be1or2')
        self.feature_threads=feature_threads
        self.device=str(device);self.cuda=None
        if torch.device(device).type=='cuda':
            if execution!='optimized' or self.feature_count not in (96,104):raise ValueError('CUDA execution requires optimized segment checkpoint')
            from .cuda import CudaExecution
            self.cuda=CudaExecution(self.net,device)

    def prepare_graph(self,graph):
        if self.future_format!='native':return super().prepare_graph(graph)
        from .prepared import content_key,CompiledGraph
        from .native import Base
        try:key=content_key(graph)
        except TypeError:return super().prepare_graph(graph)
        if key in self.cache:
            self.cache_hits+=1;self.cache.move_to_end(key);return self.cache[key]
        self.cache_misses+=1
        graph_class=CompiledGraph
        if self.execution=='optimized' and self.shared_crops:
            from .batch import SharedCompiledGraph
            graph_class=SharedCompiledGraph
        prepared=graph_class(graph)
        prepared.future_base=None
        if prepared.native_eligible:
            try:prepared.future_base=Base(graph,prepared)
            except (TypeError,ValueError,OverflowError):pass
        if self.cache_size:
            self.cache[key]=prepared
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return prepared

    def _with_base(self,delta,prepared):
        if self.future_format=='native':delta['_native_base']=getattr(prepared,'future_base',None)
        return delta

    def _predict(self, graph, trajectory, prepared):
        if self.execution == 'reference' or self.feature_count not in (96,104):
            return self._with_base(super()._predict(graph, trajectory, prepared),prepared)
        from .fused import transform, probabilities
        from numba import set_num_threads
        set_num_threads(self.feature_threads)
        if self.cuda is not None:self.cuda.last_profile.clear()
        start = time.perf_counter(); context = self.feature_class(graph, trajectory, prepared); spatial = time.perf_counter()
        keys = context.candidates('both'); candidates = time.perf_counter()
        x, baseline = transform(context, keys, end_only=self.feature_count==96,cuda=self.cuda,device_output=self.cuda is not None); features = time.perf_counter()
        transform_profile=dict(context.last_transform_profile)
        p = np.empty((len(keys),3), np.float32); height = np.empty(len(keys), np.float32)
        with torch.inference_mode():
            if self.cuda is not None and len(keys):
                values=self.cuda.forward(x,context,keys);critical=probabilities(values,baseline,p,height,0)
                if len(critical):p[critical]=torch.softmax(torch.from_numpy(values[critical,:3]),-1).numpy()
            for begin in range(0, len(keys) if self.cuda is None else 0, 8192):
                end = min(begin+8192, len(keys)); values = self.net(torch.from_numpy(x[begin:end]))
                critical=probabilities(values.numpy(), baseline, p, height, begin)
                if len(critical):
                    p[begin+critical]=torch.softmax(values[critical,:3],-1).numpy()
        inferred = time.perf_counter()
        if len(context.path) == 1:
            p.fill(0); p[:,0] = 1
        states = p.argmax(-1).astype(np.uint8); now = float(context.path[-1,0]); selected = np.flatnonzero(states)
        updates = [dict(cell=keys[i].tolist(),state=int(states[i]),height=float(height[i]),timestamp=now,source='predicted_stage2_research') for i in selected]
        delta = dict(schema=GRAPH_SCHEMA,predictor_schema='trajectory_conditioned_graphdelta_quality_2',predicted=True,
            base_graph_identity=prepared.identity,base_timestamp=float(graph['timestamp']),timestamp=now,
            robot_pose=context.path[-1,1:].tolist(),static_updates=updates,candidate_cells=keys,probabilities=p,
            states=states,heights=height,unknown_cells=keys[states==0],outside_domain='UNKNOWN_NO_UPDATE',checkpoint_sha256=self.checkpoint_sha256)
        end = time.perf_counter()
        self.last_profile = dict(spatial_seconds=spatial-start,candidate_seconds=candidates-spatial,feature_seconds=features-candidates,
            neural_seconds=inferred-features,delta_seconds=end-inferred,total_seconds=end-start,**transform_profile)
        if self.cuda is not None:self.last_profile.update(self.cuda.last_profile)
        return self._with_base(delta,prepared)

    def _complete(self, graph, delta, trajectory):
        tick = time.perf_counter()
        delta['prediction_task'] = SCHEMA
        delta['observation_scope'] = 'initially_UNKNOWN_static_cells_only'
        delta['probability_columns'] = ['UNKNOWN', 'FREE', 'BLOCKED']
        delta['provenance'] = dict(hypothetical=True, input='current_legal_graph_and_supplied_pose_time_segment',
            checkpoint_sha256=self.checkpoint_sha256, decision='raw_softmax_argmax',
            trajectory_sha256=hashlib.sha256(np.asarray(trajectory,dtype='<f8').tobytes(order='C')).hexdigest(),
            trajectory_hash_format='little_endian_float64_Nx5_C',training_source_commit=self.config.get('git_commit'),
            temporal_semantics='terminal-time predicted update; observation first_seen times are not forecast')
        native_base=delta.pop('_native_base',None)
        future=native_base.update(delta) if native_base is not None else None
        if future is None:future=apply_graph_delta(graph, delta, verified_identity=delta['base_graph_identity'])
        delta['future_partial_graph']=future
        delta['future_format']='native_columns' if hasattr(future,'cell_columns') else 'snapshot'
        return time.perf_counter()-tick

    def predict(self, *, partial_graph, trajectory):
        tick = time.perf_counter()
        delta = super().predict(partial_graph=partial_graph, trajectory=trajectory)
        self.last_profile['update_seconds'] = self._complete(partial_graph, delta, trajectory)
        self.last_profile['total_seconds'] = time.perf_counter()-tick
        return delta

    def predict_batch(self, inputs):
        inputs = list(inputs)
        if self.shared_crops and self.execution=='optimized' and self.future_format=='native' and len(inputs)>1:
            from .batch import complete_batch
            return complete_batch(self,inputs)
        tick = time.perf_counter()
        deltas = super().predict_batch(inputs)
        update = sum(self._complete(item['partial_graph'], delta,item['trajectory']) for item, delta in zip(inputs, deltas))
        self.last_batch_profile.update(update_seconds=update, total_seconds=time.perf_counter()-tick,
                                       materialized_future_graphs=len(deltas))
        return deltas

    def predict_segments(self, *, partial_graph, trajectories):
        return self.predict_batch([dict(partial_graph=partial_graph, trajectory=p) for p in trajectories])
