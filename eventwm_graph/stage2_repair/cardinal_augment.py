"""Legal90degree world-coordinate augmentation; original certificates unchanged.

Both inputs and target cell identities transform together. Cell centres require
the negative-axis offset(-k-1), rather than merely negating integer keys.
"""
import copy
import numpy as np
from .range_online_data import OnlineRangeGroups


def matrix(turns):
    c,s=[(1,0),(0,1),(-1,0),(0,-1)][int(turns)%4]
    return np.array([[c,0,-s],[0,1,0],[s,0,c]],np.float64)


def cell_keys(keys,turns):
    keys=np.asarray(keys);centres=2*keys.astype(np.int64)+1;r=matrix(turns)[np.ix_([0,2],[0,2])].astype(np.int64);out=(centres@r.T-1)//2
    if np.any(out<np.iinfo(np.int32).min)or np.any(out>np.iinfo(np.int32).max):raise ValueError('augmented coordinates exceed original int32 support')
    return out.astype(np.int32)


def item_transform(item,turns):
    turns=int(turns)%4
    if not turns:return item
    r=matrix(turns);out=dict(item);out['keys']=cell_keys(item['keys'],turns);out['ego']=np.asarray(item['ego'])@r.T;out['windows']=[]
    for e in item['windows']:
        window=dict(e);p=np.array(e['path'],copy=True);p[:,1:4]=p[:,1:4]@r.T;p[:,4]-=turns*np.pi/2;window['path']=p;window['target_keys']=cell_keys(e['target_keys'],turns);out['windows'].append(window)
    packet=item['packet'];out['packet']=dict(packet,origins=np.asarray(packet['origins'])@r.T,rotations=r@np.asarray(packet['rotations']))
    tracks=copy.deepcopy(item['tracks'])
    for row in tracks:
        for t in row:
            for key in ['position','estimated_velocity']:
                if t.get(key)is not None:t[key]=(np.asarray(t[key])@r.T).tolist()
    out['tracks']=tracks;out['augmentation_turns']=turns;return out


class CardinalOnlineGroups(OnlineRangeGroups):
    def __iter__(self):
        for item in super().__iter__():
            turns=np.random.default_rng(item['sample_seed']^0xCA12).integers(4)
            yield item_transform(item,int(turns))
