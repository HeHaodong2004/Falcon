"""Full-content cache keys and vector-validated owned static feature arrays."""
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import io
import pickle
import numpy as np
from eventwm_graph.stage2_quality.features import PreparedGraph
from scipy.ndimage import distance_transform_edt
from eventwm_graph.stage2.features import RADIUS


def content_key(graph):
    if isinstance(graph,dict):
        # The C pickler preserves container and ndarray dtype identities.
        # Only encoding is used; no unpickling or executable cache is loaded.
        try:raw=pickle.dumps(graph,protocol=5)
        except TypeError:raw=None
        if raw is not None:return 'full-pickle:'+hashlib.sha256(raw).hexdigest()
    class TypedPickler(pickle.Pickler):
        def reducer_override(self,value):
            if isinstance(value,Mapping) and not isinstance(value,dict):return dict,(dict(value),)
            return NotImplemented
    buffer=io.BytesIO();TypedPickler(buffer,protocol=5).dump(graph)
    # Container/array dtype information is now encoded at its actual position.
    # This is a digest-only encoding; these bytes are never unpickled.
    return 'full-pickle:'+hashlib.sha256(buffer.getvalue()).hexdigest()


class CompiledGraph(PreparedGraph):
    def __init__(self,graph):
        if graph.get('schema')!='causal_partial_cspace_graph_1':raise ValueError('unsupported PartialGraph schema')
        self.resolution=float(graph['resolution']);self.height=float(graph['height'])
        if not np.isfinite(self.resolution) or not .02<=self.resolution<=1:raise ValueError('unsupported map resolution')
        if not np.isfinite([self.height,float(graph['timestamp']),float(graph['clearance'])]).all():raise ValueError('nonfinite graph geometry/clock')
        cells=graph['cells'];rawkeys=np.asarray([c['cell'] for c in cells]).reshape(-1,2)
        if rawkeys.dtype.kind not in ('i','u') and len(cells):
            # Preserve the original unusual-coordinate digest and validation.
            super().__init__(graph);self.height64=np.asarray([c['height'] for c in cells],np.float64);self.native_eligible=False;return
        if rawkeys.dtype.kind=='u' and np.any(rawkeys>np.iinfo(np.int32).max):
            raise ValueError('unsigned cell coordinates exceed supported int32 geometry')
        keys=rawkeys.astype(np.int64)
        if len(keys) and (np.any(keys<np.iinfo(np.int32).min) or np.any(keys>np.iinfo(np.int32).max)):
            raise ValueError('cell coordinates exceed supported int32 feature geometry')
        values=np.asarray([c['state'] for c in cells]);heights=np.asarray([c['height'] for c in cells],np.float64)
        if not np.all((values==1)|(values==2)) or not np.isfinite(heights).all():raise ValueError('invalid known cell')
        order=np.lexsort((keys[:,1],keys[:,0])) if len(keys) else np.empty(0,np.int64)
        sortedkeys=keys[order]
        if len(keys)>1 and np.any(np.all(sortedkeys[1:]==sortedkeys[:-1],axis=1)):raise ValueError('duplicate graph cells')
        rows=np.empty((len(keys),4),np.float64);rows[:,:2]=sortedkeys;rows[:,2]=values[order];rows[:,3]=heights[order]
        digest=hashlib.sha256(rows.tobytes());digest.update(np.asarray([graph['timestamp'],graph['revision'],self.resolution,self.height,graph['clearance']],np.float64).tobytes())
        self.identity=digest.hexdigest();self.keys=sortedkeys.astype(np.int32);self.values=values[order].astype(np.uint8)
        self.height64=heights[order];self.heights=self.height64.astype(np.float32);self.input_order=order
        self.crops=OrderedDict();self.crop_hits=0;self.crop_misses=0;self.integrals=None;self.native_eligible=True

    def crop(self,lo,shape):
        key=tuple(lo)+tuple(shape)
        if key in self.crops:
            self.crop_hits+=1;self.crops.move_to_end(key);return self.crops[key]
        self.crop_misses+=1
        states=np.zeros(tuple(shape),np.uint8);heights=np.full(tuple(shape),self.height,np.float32)
        indices=self.keys-lo;ok=np.all((indices>=0)&(indices<shape),axis=1);ij=indices[ok];known=self.heights[ok]
        states[ij[:,0],ij[:,1]]=self.values[ok];heights[ij[:,0],ij[:,1]]=known
        distance=[distance_transform_edt(states!=state).astype(np.float32)*self.resolution if np.any(states==state)
                  else np.full(tuple(shape),RADIUS,np.float32) for state in (1,2)]
        if len(known) and np.all(known.view(np.uint32)==known[:1].view(np.uint32)):
            # Every possible nearest known cell has the identical F32 height.
            # This avoids one feature transform without changing any baseline.
            baseline=heights if np.float32(self.height).view(np.uint32)==known[0].view(np.uint32) else np.full(tuple(shape),known[0],np.float32)
        elif len(known):
            nearest=distance_transform_edt(states==0,return_distances=False,return_indices=True);baseline=heights[nearest[0],nearest[1]]
        else:baseline=heights
        value=(states,baseline,distance);self.crops[key]=value
        while len(self.crops)>getattr(self,'crop_cache_size',4):self.crops.popitem(last=False)
        return value
