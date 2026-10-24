"""Full positive-history feasibility and a nonuniform full-support lineage law.

No tracker/physical identity, Query, future observation, scene or hidden map is
used. A vertex is one actual positive detection, including its multiplicity.
Historical edges are strictly time-directed and speed-supported. Current
population endpoints terminate vertex-disjoint paths covering ALL detections.

Feasibility may start from a cached maximum historical matching. Lineage MUST
start empty: its normalized randomized first-edge choices followed by complete
cyclic edge searches give positive probability to every feasible full matching.
This is neither uniform matching nor a Gibbs posterior over associations.
"""
from dataclasses import dataclass
import math

import numpy as np
from numba import njit

from .types import immutable_array

VERSION = 'full_actual_positive_dag_association_4'
NUMERICAL_POLICY = 'float64_hypot_le_3dt; strictly_positive_historical_dt; exact_same_clock_endpoint_XZ'
LINEAGE_KERNEL = 'empty_matching_random_row_order; normalized_geometric_uniform_first_edge; greedy_free_then_complete_cyclic_BFS_v4'
IMPLEMENTATION = 'immutable_CSR_lazy_feasibility_and_exact_order_bitset_lineage_2'
_CERTIFICATE = object()


@njit(cache=True)
def _allowed(dt, dx, dz):
    if dt < 0.:
        return False
    if dt == 0.:
        # Do not turn underflow/rounding into a zero-radius observation match.
        return dx == 0. and dz == 0.
    return math.hypot(dx, dz) <= 3.*dt


@njit(cache=True)
def _geometry_weight(dt, dx, dz):
    # On supported edges distance/(.25+.5dt) < 6, so no exponential underflow.
    scale=.25+.5*dt
    return math.exp(-.5*((dx/scale)**2+(dz/scale)**2))


@njit(cache=True)
def _append_edges(times, xy, previous_indptr, previous_indices, previous_geometry, old_n):
    """Copy old CSR edges and calculate only edges into newly arrived vertices."""
    n=len(times);indptr=np.zeros(n+1,np.int64)
    for u in range(n):
        count=previous_indptr[u+1]-previous_indptr[u] if u<old_n else 0
        for v in range(max(old_n,u+1),n):
            dt=times[v]-times[u]
            if dt>0. and _allowed(dt,xy[v,0]-xy[u,0],xy[v,1]-xy[u,1]):
                count+=1
        indptr[u+1]=indptr[u]+count
    indices=np.empty(indptr[-1],np.int64);geometry=np.empty(indptr[-1],np.float64)
    for u in range(n):
        at=indptr[u]
        if u<old_n:
            for i in range(previous_indptr[u],previous_indptr[u+1]):
                indices[at]=previous_indices[i];geometry[at]=previous_geometry[i];at+=1
        for v in range(max(old_n,u+1),n):
            dt=times[v]-times[u];dx=xy[v,0]-xy[u,0];dz=xy[v,1]-xy[u,1]
            if dt>0. and _allowed(dt,dx,dz):
                indices[at]=v;geometry[at]=_geometry_weight(dt,dx,dz);at+=1
    return indptr,indices,geometry


@njit(cache=True)
def _augment_workspace(u, indptr, indices, terminal, ml, mr, first,
                       stack, via, cursor, seen, generation):
    """Identical complete DFS, with one reusable workspace per matching world."""
    n=len(ml);m=terminal.shape[1]
    depth=0;stack[0]=u;cursor[0]=0;seen[u]=generation
    while depth>=0:
        row=stack[depth];degree=indptr[row+1]-indptr[row];total=degree+m
        if cursor[depth]>=total:
            depth-=1
            continue
        ordinal=first[row]+cursor[depth];cursor[depth]+=1
        if ordinal>=total:ordinal-=total
        if ordinal<degree:
            column=indices[indptr[row]+ordinal]
        else:
            j=ordinal-degree
            if not terminal[row,j]:
                continue
            column=n+j
        old=mr[column]
        if old<0:
            ml[row]=column;mr[column]=row
            while depth>0:
                column=via[depth];depth-=1;row=stack[depth]
                ml[row]=column;mr[column]=row
            return True
        if seen[old]!=generation:
            seen[old]=generation;depth+=1;stack[depth]=old;via[depth]=column;cursor[depth]=0
    return False


@njit(cache=True)
def _workspace(n):
    return (np.empty(n,np.int64),np.empty(n,np.int64),np.empty(n,np.int64),np.zeros(n,np.int64))


@njit(cache=True)
def _augment(u, indptr, indices, terminal, ml, mr, first):
    """Standalone complete DFS; batched callers reuse the same workspace."""
    stack,via,cursor,seen=_workspace(len(ml))
    return _augment_workspace(u,indptr,indices,terminal,ml,mr,first,stack,via,cursor,seen,1)


@njit(cache=True)
def _history_bits(indptr,indices):
    n=len(indptr)-1;bits=np.zeros((n,(n+63)//64),np.uint64)
    for u in range(n):
        for at in range(indptr[u],indptr[u+1]):
            v=indices[at];bits[u,v//64]|=np.uint64(1)<<np.uint64(v%64)
    return bits


@njit(cache=True)
def _trailing_bit(value):
    """Index of the lowest set uint64 bit, using six exact integer tests."""
    result=0
    for shift,mask in ((32,np.uint64(0xffffffff)),(16,np.uint64(0xffff)),
                       (8,np.uint64(0xff)),(4,np.uint64(0xf)),(2,np.uint64(3)),(1,np.uint64(1))):
        if (value&mask)==0:
            result+=shift;value>>=np.uint64(shift)
    return result


@njit(cache=True)
def _first_unexcluded(bits,excluded,first):
    words=len(bits)
    if not words:return -1
    if first<0 or first>=64*words:first=0
    word=first//64;shift=first%64
    lower=(np.uint64(1)<<np.uint64(shift))-np.uint64(1)
    for step in range(words+1):
        at=word+step
        if at>=words:at-=words
        available=bits[at]&~excluded[at]
        if step==0:available&=~lower
        elif step==words:available&=lower
        if available:return at*64+_trailing_bit(available)
    return -1


@njit(cache=True)
def _bitset_augment(u,bits,ml,mr,first,queue,seen,parent,visited,generation):
    head=0;tail=1;queue[0]=u;seen[u]=generation;visited[:]=np.uint64(0)
    while head<tail:
        row=queue[head];head+=1
        column=_first_unexcluded(bits[row],visited,first[row])
        while column>=0:
            visited[column//64]|=np.uint64(1)<<np.uint64(column%64)
            parent[column]=row;old=mr[column]
            if old<0:
                while column>=0:
                    source=parent[column];previous=ml[source]
                    ml[source]=column;mr[column]=source;column=previous
                return True
            if seen[old]!=generation:
                seen[old]=generation;queue[tail]=old;tail+=1
            column=_first_unexcluded(bits[row],visited,first[row])
    return False


@njit(cache=True)
def _extend_maximum(indptr, indices, previous):
    n=len(indptr)-1;ml=np.full(n,-1,np.int64);mr=np.full(n,-1,np.int64)
    for u in range(len(previous)):
        column=previous[u];ml[u]=column
        if column>=0:
            mr[column]=u
    terminal=np.empty((n,0),np.bool_);first=np.zeros(n,np.int64)
    stack,via,cursor,seen=_workspace(n)
    for u in range(n):
        if ml[u]<0:
            _augment_workspace(u,indptr,indices,terminal,ml,mr,first,stack,via,cursor,seen,u+1)
    return ml


@njit(cache=True)
def _terminal_edges(times, xy, now, endpoints, present, geometry_needed):
    worlds,m,_=endpoints.shape;n=len(times)
    terminal=np.empty((worlds,n,m),np.bool_)
    geometry=np.zeros((worlds,n,m),np.float64) if geometry_needed else np.empty((0,0,0),np.float64)
    for k in range(worlds):
        for u in range(n):
            dt=now-times[u]
            for j in range(m):
                dx=endpoints[k,j,0]-xy[u,0];dz=endpoints[k,j,1]-xy[u,1]
                allowed=present[k,j] and _allowed(dt,dx,dz)
                terminal[k,u,j]=allowed
                if geometry_needed and allowed:
                    geometry[k,u,j]=_geometry_weight(dt,dx,dz)
    return terminal,geometry


@njit(cache=True)
def _feasible(indptr, indices, maximum, terminal):
    worlds,n,m=terminal.shape;answers=np.ones(worlds,np.bool_);first=np.zeros(n,np.int64)
    for k in range(worlds):
        ml=maximum.copy();mr=np.full(n+m,-1,np.int64)
        stack,via,cursor,seen=_workspace(n)
        for u in range(n):
            if ml[u]>=0:
                mr[ml[u]]=u
        for u in range(n):
            if ml[u]<0 and not _augment_workspace(u,indptr,indices,terminal[k],ml,mr,first,
                                                 stack,via,cursor,seen,u+1):
                answers[k]=False
                break
    return answers


@njit(cache=True)
def _augment_lazy(u, indptr, indices, times, xy, now, endpoints, present,
                  terminal, calculated, ml, mr, stack, via, cursor, seen, generation):
    """The same complete DFS, calculating a terminal row only when visited.

    Historical edge order, endpoint column order and augmenting-path rewrites
    match the eager reference above. Cached rows are local to one current-world
    proposal and never survive an endpoint or observation change.
    """
    n=len(ml);m=len(endpoints)
    depth=0;stack[0]=u;cursor[0]=0;seen[u]=generation
    while depth>=0:
        row=stack[depth];degree=indptr[row+1]-indptr[row]
        if cursor[depth]>=degree+m:
            depth-=1
            continue
        ordinal=cursor[depth];cursor[depth]+=1
        if ordinal<degree:
            column=indices[indptr[row]+ordinal]
        else:
            if not calculated[row]:
                dt=now-times[row]
                for j in range(m):
                    terminal[row,j]=present[j] and _allowed(dt,
                        endpoints[j,0]-xy[row,0],endpoints[j,1]-xy[row,1])
                calculated[row]=True
            j=ordinal-degree
            if not terminal[row,j]:
                continue
            column=n+j
        old=mr[column]
        if old<0:
            ml[row]=column;mr[column]=row
            while depth>0:
                column=via[depth];depth-=1;row=stack[depth]
                ml[row]=column;mr[column]=row
            return True
        if seen[old]!=generation:
            seen[old]=generation;depth+=1;stack[depth]=old;via[depth]=column;cursor[depth]=0
    return False


@njit(cache=True)
def _feasible_lazy(indptr, indices, maximum, times, xy, now, endpoints, present):
    """Pointwise exact feasibility without a [proposal,detection,person] array."""
    worlds,m,_=endpoints.shape;n=len(times)
    answers=np.ones(worlds,np.bool_)
    for k in range(worlds):
        ml=maximum.copy();mr=np.full(n+m,-1,np.int64)
        terminal=np.empty((n,m),np.bool_);calculated=np.zeros(n,np.bool_)
        stack,via,cursor,seen=_workspace(n)
        for u in range(n):
            if ml[u]>=0:
                mr[ml[u]]=u
        for u in range(n):
            if ml[u]<0 and not _augment_lazy(u,indptr,indices,times,xy,now,
                    endpoints[k],present[k],terminal,calculated,ml,mr,
                    stack,via,cursor,seen,u+1):
                answers[k]=False
                break
    return answers


@njit(cache=True)
def _first_choices(indptr, indices, geometry, terminal, terminal_geometry, uniforms, uniform_mass):
    """Normalized p(edge)=eta/degree+(1-eta)*g(edge)/sum(g)."""
    worlds,n,m=terminal.shape;first=np.zeros((worlds,n),np.int64)
    first_column=np.full((worlds,n),-1,np.int64);log_probability=np.zeros(worlds,np.float64)
    # Parameter-independent historical prefixes are shared by all worlds. Exact
    # first-edge probabilities remain eta/degree+(1-eta)*g/sum(g).
    cumulative=np.empty(len(geometry),np.float64)
    for u in range(n):
        value=0.
        for at in range(indptr[u],indptr[u+1]):
            value+=geometry[at];cumulative[at]=value
    for k in range(worlds):
        for u in range(n):
            past_degree=indptr[u+1]-indptr[u];degree=past_degree
            past_sum=cumulative[indptr[u+1]-1] if past_degree else 0.
            gsum=past_sum
            for j in range(m):
                if terminal[k,u,j]:
                    degree+=1;gsum+=terminal_geometry[k,u,j]
            if not degree:
                continue
            cdf=uniform_mass*past_degree/degree+(1.-uniform_mass)*past_sum/gsum
            if past_degree and uniforms[k,u]<cdf:
                low=0;high=past_degree-1
                while low<high:
                    mid=(low+high)//2
                    midcdf=uniform_mass*(mid+1)/degree+(1.-uniform_mass)*cumulative[indptr[u]+mid]/gsum
                    if uniforms[k,u]<midcdf:high=mid
                    else:low=mid+1
                chosen=low;at=indptr[u]+chosen;chosen_column=indices[at]
                chosen_probability=uniform_mass/degree+(1.-uniform_mass)*geometry[at]/gsum
            else:
                # Last allowed edge is also a roundoff-safe categorical fallback.
                chosen=past_degree-1;chosen_column=indices[indptr[u+1]-1] if past_degree else -1
                chosen_probability=uniform_mass/degree+(1.-uniform_mass)*geometry[indptr[u+1]-1]/gsum if past_degree else 0.
                for j in range(m):
                    if not terminal[k,u,j]:continue
                    probability=uniform_mass/degree+(1.-uniform_mass)*terminal_geometry[k,u,j]/gsum
                    cdf+=probability
                    chosen=past_degree+j;chosen_column=n+j;chosen_probability=probability
                    if uniforms[k,u]<cdf:break
            first[k,u]=chosen;first_column[k,u]=chosen_column
            log_probability[k]+=math.log(chosen_probability)
    return first,first_column,log_probability


@njit(cache=True)
def _bitset_lineages(past_bits, terminal, first, row_order):
    worlds,n,m=terminal.shape;matching=np.full((worlds,n),-1,np.int64)
    endpoint=np.full((worlds,n),-1,np.int64);physical_seen=np.zeros((worlds,m),np.bool_)
    valid=np.ones(worlds,np.bool_)
    for k in range(worlds):
        # Intentionally EMPTY. The cached feasibility maximum is not a lineage prior.
        ml=np.full(n,-1,np.int64);mr=np.full(n+m,-1,np.int64)
        words=(n+m+63)//64;bits=np.zeros((n,words),np.uint64)
        for u in range(n):
            for word in range(past_bits.shape[1]):bits[u,word]=past_bits[u,word]
            for j in range(m):
                if terminal[k,u,j]:
                    column=n+j;bits[u,column//64]|=np.uint64(1)<<np.uint64(column%64)
        queue=np.empty(n,np.int64);seen=np.zeros(n,np.int64);parent=np.empty(n+m,np.int64)
        occupied=np.zeros(words,np.uint64);visited=np.zeros(words,np.uint64)
        # First take free edges in the declared randomized order. This starts
        # EMPTY and freezes no associations: complete augmentations below may
        # reassign every greedy choice. It avoids repeatedly rearranging early
        # partial matchings before later rows have tried their free neighbors.
        for at in range(n):
            u=row_order[k,at];column=_first_unexcluded(bits[u],occupied,first[k,u])
            if column>=0:
                ml[u]=column;mr[column]=u
                occupied[column//64]|=np.uint64(1)<<np.uint64(column%64)
        for at in range(n):
            u=row_order[k,at]
            if ml[u]<0 and not _bitset_augment(u,bits,ml,mr,first[k],queue,seen,parent,visited,at+1):
                valid[k]=False;break
        if not valid[k]:
            continue
        matching[k]=ml
        for u in range(n-1,-1,-1):
            column=ml[u]
            person=column-n if column>=n else endpoint[k,column]
            # DAG historical indices strictly increase, so descendants are known.
            endpoint[k,u]=person;physical_seen[k,person]=True
    return valid,matching,endpoint,physical_seen


@njit(cache=True)
def _lineages(indptr,indices,terminal,first,row_order):
    """CSR ordinal compatibility entry; production reuses its cached bitsets."""
    worlds,n,m=terminal.shape;columns=np.zeros((worlds,n),np.int64)
    for k in range(worlds):
        for u in range(n):
            degree=indptr[u+1]-indptr[u]
            column=indices[indptr[u]+first[k,u]] if first[k,u]<degree else n+first[k,u]-degree
            columns[k,u]=column if column<n+m else 0
    return _bitset_lineages(_history_bits(indptr,indices),terminal,columns,row_order)


def _columns(history):
    """Validate only the declared causal numeric witness; IDs are never consumed."""
    times=np.asarray(history.timestamps,np.float64)
    offsets=np.asarray(history.positive_offsets,np.int64)
    xyz=np.asarray(history.positive_xyz,np.float64)
    valid=np.asarray(history.observation_valid,bool)
    available=np.asarray(history.human_sensor_available,bool)
    n=len(times)
    if (n==0 or times.shape!=(n,) or not np.isfinite(times).all() or np.any(np.diff(times)<=0.)
            or offsets.shape!=(n+1,) or offsets[0]!=0 or np.any(np.diff(offsets)<0)
            or xyz.shape!=(int(offsets[-1]),3) or not np.isfinite(xyz).all()
            or valid.shape!=(n,) or available.shape!=(n,)
            or float(history.horizontal_motion_bound_m_s)!=3.):
        raise ValueError('invalid causal positive-set association witness')
    if np.any((np.diff(offsets)>0)&~(valid&available)):
        raise ValueError('invalid/unavailable frame asserts positive detections')
    return times,offsets,xyz,valid,available


def _endpoints(endpoints,present):
    points=np.asarray(endpoints,np.float64)
    if points.ndim!=3 or points.shape[-1]!=2 or not np.isfinite(points).all():
        raise ValueError('current endpoint populations must be finite float64 [world,person,XZ]')
    mask=np.ones(points.shape[:2],bool) if present is None else np.asarray(present,bool)
    if mask.shape!=points.shape[:2]:
        raise ValueError('current endpoint presence shape differs')
    # One stable writable C layout avoids a new JIT specialization when a caller
    # switches between immutable world arrays and mutable proposal buffers.
    return np.array(points,copy=True,order='C'),np.array(mask,copy=True,order='C')


@dataclass(frozen=True,init=False)
class LineageWitness:
    """Sampled backward path ownership, conditional on a feasible population.

    event_person[k,d] is the current population column descended from raw
    detection d. physical_seen is per WORLD and person, including Poisson-source
    endpoints with actual ancestors. It is not fixed episode ownership.

    auxiliary_log_probability describes the sampled row permutation and first-
    edge choices. It is NOT the probability of the resulting matching, because
    multiple auxiliary randomizations can produce that matching.
    """
    matching: np.ndarray
    event_person: np.ndarray
    physical_seen: np.ndarray
    row_order: np.ndarray
    first_edge: np.ndarray
    auxiliary_log_probability: np.ndarray
    kernel: str = LINEAGE_KERNEL


@dataclass(frozen=True,init=False)
class AssociationGraph:
    """Immutable full-history CSR DAG and a maximum for feasibility only."""
    frame_timestamps: np.ndarray
    positive_offsets: np.ndarray
    positive_xyz: np.ndarray
    observation_valid: np.ndarray
    human_sensor_available: np.ndarray
    times: np.ndarray
    xy: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    geometry: np.ndarray
    past_bits: np.ndarray
    maximum_history_matching: np.ndarray
    minimum_current_count: int
    version: str = VERSION
    numerical_policy: str = NUMERICAL_POLICY

    @property
    def timestamp(self):
        return float(self.frame_timestamps[-1])

    @property
    def detections(self):
        return len(self.times)

    def validate_certificate(self):
        if getattr(self,'_certificate',None) is not _CERTIFICATE:
            raise ValueError('uncertified full-history association graph')
        return self

    @classmethod
    def _verified(cls,columns,times,xy,indptr,indices,geometry,matching,past_bits=None):
        result=object.__new__(cls)
        names=('frame_timestamps','positive_offsets','positive_xyz','observation_valid','human_sensor_available')
        for name,value in zip(names,columns):
            object.__setattr__(result,name,immutable_array(value))
        for name,value in (('times',times),('xy',xy),('indptr',indptr),('indices',indices),
                           ('geometry',geometry),('maximum_history_matching',matching)):
            object.__setattr__(result,name,immutable_array(value))
        object.__setattr__(result,'past_bits',immutable_array(_history_bits(indptr,indices) if past_bits is None else past_bits))
        for name,value in dict(minimum_current_count=int(np.sum(matching<0)),version=VERSION,
                               numerical_policy=NUMERICAL_POLICY,_certificate=_CERTIFICATE).items():
            object.__setattr__(result,name,value)
        return result

    @classmethod
    def from_history(cls,history):
        columns=_columns(history);frame_times,offsets,xyz,_,_=columns
        times=np.repeat(frame_times,np.diff(offsets));xy=np.ascontiguousarray(xyz[:,[0,2]])
        # Initial and later updates use the same readonly previous-state types;
        # first online append must not discover a second JIT specialization.
        indptr,indices,geometry=_append_edges(times,xy,immutable_array(np.zeros(1,np.int64)),
            immutable_array(np.empty(0,np.int64)),immutable_array(np.empty(0,np.float64)),0)
        matching=_extend_maximum(indptr,indices,immutable_array(np.empty(0,np.int64)))
        return cls._verified(columns,times,xy,indptr,indices,geometry,matching)

    def append(self,history):
        """Consume only a causal extension; all new graph/matching work is online."""
        self.validate_certificate();columns=_columns(history)
        old_frames=len(self.frame_timestamps);old_n=self.detections
        previous=(self.frame_timestamps,self.positive_offsets,self.positive_xyz,
                  self.observation_valid,self.human_sensor_available)
        lengths=(old_frames,old_frames+1,old_n,old_frames,old_frames)
        for new,old,length in zip(columns,previous,lengths):
            if len(new)<length or not np.array_equal(new[:length],old):
                raise ValueError('association append changes or truncates arrived positive history')
        if len(columns[0])==old_frames:
            return self
        if len(columns[2])==old_n:
            return type(self)._verified(columns,self.times,self.xy,self.indptr,self.indices,
                self.geometry,self.maximum_history_matching,self.past_bits)
        frame_times,offsets,xyz,_,_=columns
        times=np.repeat(frame_times,np.diff(offsets));xy=np.ascontiguousarray(xyz[:,[0,2]])
        indptr,indices,geometry=_append_edges(times,xy,self.indptr,self.indices,self.geometry,old_n)
        matching=_extend_maximum(indptr,indices,self.maximum_history_matching)
        return type(self)._verified(columns,times,xy,indptr,indices,geometry,matching)

    def feasible(self,endpoints,present=None):
        self.validate_certificate();points,mask=_endpoints(endpoints,present)
        return _feasible_lazy(self.indptr,self.indices,self.maximum_history_matching,
                             self.times,self.xy,self.timestamp,points,mask)

    def lineage(self,endpoints,present=None,rng=None,uniform_mass=.1):
        """Normalized NONuniform full-support randomized matching pushforward.

        Each row chooses a first edge from eta/degree+(1-eta)*g/sum(g), eta>0,
        then visits every remaining edge cyclically. g is the declared positive
        geometric weight. Starting EMPTY, a free-edge greedy pass is followed
        by complete breadth-first augmenting searches for all unmatched rows, with a random row
        permutation. Greedy choices may all be reassigned. For any feasible matching, choosing each of its
        edges first constructs it without conflict, an event of positive mass.
        No finite rejection cap, association beam or edge truncation is used.
        """
        self.validate_certificate();points,mask=_endpoints(endpoints,present)
        if not np.isfinite(uniform_mass) or not 0.<uniform_mass<=1.:
            raise ValueError('full-support lineage requires positive uniform mixture mass')
        if rng is None:
            rng=np.random.default_rng()
        if not isinstance(rng,np.random.Generator):
            raise TypeError('lineage needs a numpy Generator')
        terminal,terminal_geometry=_terminal_edges(self.times,self.xy,self.timestamp,points,mask,True)
        worlds=len(points);n=self.detections
        uniforms=rng.random((worlds,n))
        order=np.empty((worlds,n),np.int64)
        for k in range(worlds):
            order[k]=rng.permutation(n)
        first,first_column,log_probability=_first_choices(self.indptr,self.indices,self.geometry,terminal,terminal_geometry,uniforms,float(uniform_mass))
        valid,matching,event_person,physical_seen=_bitset_lineages(self.past_bits,terminal,first_column,order)
        if not np.all(valid):
            raise ValueError('infeasible population cannot have a lineage: worlds '+repr(np.flatnonzero(~valid).tolist()))
        if np.any(physical_seen&~mask):
            raise RuntimeError('lineage attached to absent endpoint')
        log_probability-=math.lgamma(n+1)
        result=object.__new__(LineageWitness)
        for name,value in (('matching',matching),('event_person',event_person),('physical_seen',physical_seen),
                           ('row_order',order),('first_edge',first_column),('auxiliary_log_probability',log_probability)):
            object.__setattr__(result,name,immutable_array(value))
        object.__setattr__(result,'kernel',LINEAGE_KERNEL)
        return result
