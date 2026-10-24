"""Single-thread compiled numerical loops; no support or law approximation."""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def _insert(logs,used,assign,births,count,lp,oldused,oldassign,nb,zi,slot,track):
    pos=0
    while pos<count and lp<=logs[pos]:pos+=1
    if pos>=8:return count
    newcount=min(8,count+1)
    for j in range(newcount-1,pos,-1):
        logs[j]=logs[j-1];used[j]=used[j-1];assign[j]=assign[j-1];births[j]=births[j-1]
    logs[pos]=lp;used[pos]=oldused;assign[pos]=oldassign;assign[pos,zi]=slot;births[pos]=nb
    if track>=0:used[pos,track]=True
    return newcount


@njit(cache=True)
def association_beam(slots,logdensity,ranks,supported,log_pd,hidden_den,birth_density,hidden_log_pd,nextslot):
    """The original stable top-eight injective search, with common rank pruning."""
    n,z=logdensity.shape
    logs=np.zeros(8);used=np.zeros((8,n),np.bool_);assign=np.zeros((8,z),np.int64)
    births=np.zeros(8,np.int64);rank_history=np.zeros(z,np.int64);count=1
    for zi in range(z):
        bestlogs=np.zeros(8);bestused=np.zeros((8,n),np.bool_);bestassign=np.zeros((8,z),np.int64)
        bestbirths=np.zeros(8,np.int64);bestcount=0;minimum_rank=3
        for j in range(count):
            for i in range(n):
                if used[j,i] or not supported[i,zi]:continue
                rank=ranks[i]
                if rank>minimum_rank:continue
                if rank<minimum_rank:minimum_rank=rank;bestcount=0
                lp=logs[j]+logdensity[i,zi]+log_pd[zi]
                bestcount=_insert(bestlogs,bestused,bestassign,bestbirths,bestcount,lp,
                    used[j],assign[j],births[j],zi,slots[i],i)
            nb=births[j]
            if nb<len(hidden_den) and minimum_rank>=2:
                minimum_rank=2
                ll=math.log(hidden_den[nb])+birth_density[zi]+hidden_log_pd
                bestcount=_insert(bestlogs,bestused,bestassign,bestbirths,bestcount,logs[j]+ll,
                    used[j],assign[j],nb+1,zi,nextslot+zi,-1)
        rank_history[zi]=minimum_rank;logs=bestlogs;used=bestused;assign=bestassign;births=bestbirths;count=bestcount
        if count==0:break
    return logs,used,assign,births,rank_history,count
