"""One exact CPU pass for probabilities, hard states and sparse row columns."""
import numpy as np
from numba import njit


@njit(cache=True,nogil=True)
def decode_columns(values,baseline,keys,threshold,no_future):
    n=len(keys);prob=np.empty((n,3),np.float32);heights=np.empty(n,np.float32);states=np.zeros(n,np.uint8);count=0
    for i in range(n):
        for j in range(3):
            if not np.isfinite(values[i,j]):raise ValueError('nonfinite prediction probability')
            prob[i,j]=np.float32(1. if j==0 else 0.)if no_future else values[i,j]
        heights[i]=values[i,3]+baseline[i]
        if not np.isfinite(heights[i]):raise ValueError('nonfinite predicted height')
        r=np.float64(np.float32(prob[i,1]+prob[i,2]))
        if r>=threshold:states[i]=1 if prob[i,1]>=prob[i,2]else 2;count+=1
    newkeys=np.empty((count,2),np.int32);newstates=np.empty(count,np.uint8);newheights=np.empty(count,np.float32);unknown=np.empty((n-count,2),np.int32);a=0;b=0
    for i in range(n):
        if states[i]:newkeys[a]=keys[i];newstates[a]=states[i];newheights[a]=heights[i];a+=1
        else:unknown[b]=keys[i];b+=1
    return prob,heights,states,newkeys,newstates,newheights,unknown
