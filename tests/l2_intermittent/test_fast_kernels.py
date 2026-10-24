"""Compiled search must retain the original stable injective top-eight law."""
import math
import unittest
import numpy as np
from eventwm_graph.l2_intermittent.fast_kernels import association_beam


def original(slots,ll,ranks,supported,logpd,hidden,birth,loghd,nextslot):
    beam=[(0.,(),(),(),0)]
    for zi in range(ll.shape[1]):
        options=[]
        for lp,used,assign,history,nb in beam:
            for i in range(len(slots)):
                if i in used or not supported[i,zi]:continue
                options.append((lp+float(ll[i,zi])+float(logpd[zi]),used+(i,),assign+(int(slots[i]),),history+(int(ranks[i]),),nb))
            if nb<len(hidden):
                density=math.log(float(hidden[nb]))+float(birth[zi])+loghd
                options.append((lp+density,used,assign+(nextslot+zi,),history+(2,),nb+1))
        if not options:return []
        rank=min(o[3] for o in options);options=[o for o in options if o[3]==rank]
        options.sort(key=lambda o:o[0],reverse=True);beam=options[:8]
    return [(p,tuple(sorted(u)),a,r,b) for p,u,a,r,b in beam]


class CompiledSearchTests(unittest.TestCase):
    def test_ties_singular_ranks_births_and_large_persistent_track_sets(self):
        rng=np.random.default_rng(73)
        for n,z in [(0,0),(0,3),(1,0),(1,4),(2,2),(4,3),(6,6),(16,16),(70,3)]:
            for trial in range(12):
                slots=np.arange(n,dtype=np.int64)*3;ll=rng.normal(size=(n,z))
                if trial%3==0:ll[:]=0.
                rank=rng.integers(0,3,size=n);ok=rng.random((n,z))>.2
                pd=np.log(rng.uniform(.001,.99,size=z));hidden=rng.uniform(.01,2.,size=trial%5)
                birth=rng.normal(size=z);loghd=math.log(.015);nextslot=3*n
                ref=original(slots,ll,rank,ok,pd,hidden,birth,loghd,nextslot)
                lp,used,assign,nb,history,count=association_beam(slots,ll,rank,ok,pd,hidden,birth,loghd,nextslot)
                actual=[(float(lp[j]),tuple(np.flatnonzero(used[j])),tuple(assign[j]),tuple(history),int(nb[j])) for j in range(count)]
                self.assertEqual(actual,ref,(n,z,trial))


if __name__=='__main__':unittest.main()
