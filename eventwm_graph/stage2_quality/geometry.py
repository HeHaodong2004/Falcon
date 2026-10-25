"""Low-allocation digest with the exact frozen graph-identity byte contract."""
import hashlib
import numpy as np

def graph_identity(graph):
    cells=graph['cells']
    integer_keys=np.empty((len(cells),2),np.int64)
    def values():
        for i,cell in enumerate(cells):
            x,z=cell['cell'];integer_keys[i,0]=x;integer_keys[i,1]=z
            if integer_keys[i,0]!=x or integer_keys[i,1]!=z:raise ValueError('nonintegral cell coordinate')
            yield x;yield z;yield cell['state'];yield cell['height']
    try:rows=np.fromiter(values(),np.float64,count=4*len(cells)).reshape(-1,4)
    except (OverflowError,ValueError,TypeError):
        from eventwm_graph.stage2.api import graph_identity as reference
        return reference(graph)
    if len(rows)>1:rows=rows[np.lexsort((rows[:,3],rows[:,2],integer_keys[:,1],integer_keys[:,0]))]
    h=hashlib.sha256(rows.tobytes())
    h.update(np.asarray([graph['timestamp'],graph['revision'],graph['resolution'],graph['height'],graph['clearance']],np.float64).tobytes())
    return h.hexdigest()
