"""Attach future surface TARGETS after the legal online input item is built."""
from pathlib import Path
import numpy as np
from .diagnose import read,sha
from .range_online_data import OnlineRangeGroups


class SurfaceOnlineGroups(OnlineRangeGroups):
    def __init__(self,manifest,target_dir,snapshot_sha256):
        super().__init__(manifest);self.target_dir=Path(target_dir);self.snapshot_sha256=snapshot_sha256

    def __iter__(self):
        current=None;targets=None;lookup=None;low=None
        for item in super().__iter__():
            rid=item['root_id']
            if current!=rid:
                path=self.target_dir/(rid+'.npz');receipt=read(path.with_suffix('.json'))
                if receipt['identity']['snapshot_sha256']!=self.snapshot_sha256 or sha(path)!=receipt['sha256']:raise ValueError('surface target source mismatch')
                with np.load(path)as z:targets=z['surface_targets'];low=z['lo'];lookup={int(n):i for i,n in enumerate(z['start_nodes'])}
                current=rid
            mi=lookup[-1 if item['start_node']is None else int(item['start_node'])]
            item['surface_targets']=targets[mi];item['surface_low']=low[mi]
            yield item
