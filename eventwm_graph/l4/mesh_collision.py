"""Native collision triangles and gait-envelope OBBs; no simulator search."""
import math
import numpy as np
from numba import njit
from .mesh_visibility import _ray_batch


@njit(cache=True)
def below_base_seam_support(tri,base_y,triangles,bounds,left,right,first,count,indices):
    """Stage-0 above-climb probes for a wholly below-base distal seam only."""
    n=np.cross(tri[1]-tri[0],tri[2]-tri[0])
    n[1]=0.
    length=np.sqrt(np.dot(n,n))
    if length<1e-10:
        return False
    n/=length
    origins=np.empty((8,3))
    directions=np.empty((8,3))
    for j in range(4):
        p=tri[j].copy() if j<3 else (tri[0]+tri[1]+tri[2])/3
        p[1]=base_y+.20+.02
        for side in range(2):
            sign=1 if side else -1
            origins[2*j+side]=p+sign*.05*n
            directions[2*j+side]=-sign*n
    distance,hit,edge,ambiguous=_ray_batch(triangles,bounds,left,right,first,count,indices,
        origins,directions,np.full(8,.10))
    return not (hit>=0).any() and not ambiguous.any()


@njit(cache=True)
def triangle_box(tri, lo, hi):
    """Separating-axis triangle/AABB intersection, including thin triangles."""
    center = (lo+hi)*.5
    half = (hi-lo)*.5
    v = tri-center
    for a in range(3):
        if min(v[0,a], v[1,a], v[2,a]) > half[a] or max(v[0,a], v[1,a], v[2,a]) < -half[a]:
            return False
    edges = np.empty((3, 3))
    edges[0] = v[1]-v[0]
    edges[1] = v[2]-v[1]
    edges[2] = v[0]-v[2]
    normal = np.cross(edges[0], edges[1])
    radius = np.sum(np.abs(normal)*half)
    if abs(np.dot(normal,v[0])) > radius+1e-9:
        return False
    for edge in edges:
        for axis in range(3):
            basis = np.zeros(3)
            basis[axis] = 1.
            n = np.cross(edge, basis)
            r = np.sum(np.abs(n)*half)
            p0, p1, p2 = np.dot(v[0],n), np.dot(v[1],n), np.dot(v[2],n)
            if min(p0,p1,p2) > r+1e-9 or max(p0,p1,p2) < -r-1e-9:
                return False
    return True


@njit(cache=True)
def mesh_free(pose, boxes, triangles, bounds, left, right, first, count, indices,
              forward_extension=0., rotation_padding=0., vertical_extension=0., safety_margin=.025,
              arm_obbs=None, seam_cache=None):
    """Conservative union of swept link boxes, native floor support exclusion.

    boxes rows contain min XYZ, max XYZ, distal flag. Only distal geometry
    may intersect an upward support triangle below the .2 m climb envelope.
    """
    c, s = math.cos(pose[3]), math.sin(pose[3])
    margin = safety_margin+rotation_padding
    broad_lo = np.array([1e20,1e20,1e20])
    broad_hi = -broad_lo
    for b in boxes:
        for a in range(3):
            broad_lo[a] = min(broad_lo[a], b[a])
            broad_hi[a] = max(broad_hi[a], b[a+3])
    broad_lo[0] -= forward_extension+margin
    broad_hi[0] += forward_extension+margin
    broad_lo[2] -= margin
    broad_hi[2] += margin
    broad_lo[1] -= vertical_extension
    broad_hi[1] += vertical_extension
    # World AABB broad phase over the rotated union.
    wc = (broad_lo+broad_hi)*.5
    wh = (broad_hi-broad_lo)*.5
    world_center = np.array([pose[0]+c*wc[0]+s*wc[2], pose[1]+wc[1],
                             pose[2]-s*wc[0]+c*wc[2]])
    world_half = np.array([abs(c)*wh[0]+abs(s)*wh[2], wh[1],
                           abs(s)*wh[0]+abs(c)*wh[2]])
    low, high = world_center-world_half, world_center+world_half
    stack = np.empty(128,np.int64)
    stack[0] = 0
    size = 1
    while size:
        size -= 1
        node = stack[size]
        if (bounds[node,:3] > high).any() or (bounds[node,3:] < low).any():
            continue
        if count[node] == 0:
            if size+2 > 128:
                return False
            stack[size], stack[size+1] = left[node],right[node]
            size += 2
            continue
        for k in range(first[node],first[node]+count[node]):
            triangle_index=indices[k]
            tri = triangles[triangle_index]
            local = np.empty((3,3))
            for j in range(3):
                x,z = tri[j,0]-pose[0], tri[j,2]-pose[2]
                local[j,0],local[j,1],local[j,2] = c*x-s*z,tri[j,1]-pose[1],s*x+c*z
            n = np.cross(tri[1]-tri[0],tri[2]-tri[0])
            support = (abs(n[1]) >= .70710678*np.sqrt(np.dot(n,n))
                       and np.max(local[:,1]) <= .2)
            minimum=np.array([min(local[0,a],local[1,a],local[2,a]) for a in range(3)])
            maximum=np.array([max(local[0,a],local[1,a],local[2,a]) for a in range(3)])
            seam_support=-1
            for box_index in range(len(boxes)):
                box=boxes[box_index]
                if box[6] == 1 and support:
                    continue
                # Same first three separating axes as triangle_box, before
                # allocating its transformed vertices for every link. Keep
                # touching/roundoff cases in the exact path.
                if (maximum[1]<box[1]-vertical_extension-1e-9 or
                    minimum[1]>box[4]+vertical_extension+1e-9 or
                    maximum[0]<box[0]-forward_extension-margin-1e-9 or
                    minimum[0]>box[3]+forward_extension+margin+1e-9 or
                    maximum[2]<box[2]-margin-1e-9 or
                    minimum[2]>box[5]+margin+1e-9):
                    continue
                lo,hi = box[:3].copy(),box[3:6].copy()
                lo[0] -= forward_extension+margin
                hi[0] += forward_extension+margin
                lo[2] -= margin
                hi[2] += margin
                lo[1] -= vertical_extension
                hi[1] += vertical_extension
                if triangle_box(local,lo,hi):
                    # Body-aligned link AABBs contain empty corners when the
                    # native wrist/arm is angled. Refine those intersections
                    # with the same asset-local box and its legal FK axes.
                    # Project the entire swept root displacement and unchanged
                    # horizontal margin into each link axis (Minkowski sum).
                    if arm_obbs is not None and len(arm_obbs)==9 and len(boxes)==21 and box_index<9:
                        obb=arm_obbs[box_index]
                        axes=obb[6:].reshape((3,3))
                        link_tri=(local-obb[:3])@axes
                        padding=(np.abs(axes[0,:])*(forward_extension+margin)+
                                 np.abs(axes[2,:])*margin+
                                 np.abs(axes[1,:])*vertical_extension)
                        half=obb[3:6]+padding
                        if not triangle_box(link_tri,-half,half):
                            continue
                    if box[6]==1 and np.max(local[:,1]) < -vertical_extension:
                        if seam_support<0:
                            if seam_cache is not None:
                                seam_support=int(seam_cache[triangle_index])
                            if seam_support<0:
                                seam_support=int(below_base_seam_support(tri,pose[1]-vertical_extension,
                                    triangles,bounds,left,right,first,count,indices))
                                if seam_cache is not None:
                                    seam_cache[triangle_index]=seam_support
                        if seam_support==1:
                            continue
                    return False
    return True


class StaticCollision:
    def __init__(self, visibility, boxes):
        self.triangles = visibility.triangles
        # Ray padding preserves near-edge ambiguity detection for support
        # probes. It broadens traversal only; triangle SAT remains unchanged.
        self.bvh = getattr(visibility,"ray_bvh",visibility.bvh)
        self.boxes = np.asarray(boxes,float)
        self.metadata = dict(visibility.metadata)
        self.arm_obbs=np.empty((0,15))
        self._seam_base_y=None
        self._seam_cache=np.full(len(self.triangles),-1,np.int8)

    @classmethod
    def from_envelope(cls, visibility, envelope):
        # Preserve individual leg boxes; a single box fills the empty area
        # between feet and would misclassify ordinary floor support.
        body = envelope["body_and_arm"]
        boxes = [[*body["minimum"],*body["maximum"],0]]
        for link in envelope["links"]:
            if link["link"] in envelope["leg_links"]:
                boxes.append([*link["minimum"],*link["maximum"],
                              int(link["link"] in envelope["distal_links"])])
        return cls(visibility,boxes)

    def free(self, pose, extension=0., padding=0., vertical=0., boxes=None, margin=.025):
        base_y=float(pose[1]-vertical)
        if self._seam_base_y!=base_y:
            self._seam_cache.fill(-1)
            self._seam_base_y=base_y
        return bool(mesh_free(np.asarray(pose,float),self.boxes if boxes is None else boxes,self.triangles,
                              *self.bvh,extension,padding,vertical,margin,self.arm_obbs,self._seam_cache))

    def swept(self, before, after, boxes=None, margin=.025):
        before,after = np.asarray(before),np.asarray(after)
        angle = math.atan2(math.sin(after[3]-before[3]),math.cos(after[3]-before[3]))
        mid = (before+after)*.5
        mid[3] = before[3]+angle*.5
        displacement=after-before
        lateral=abs(math.sin(mid[3])*displacement[0]+math.cos(mid[3])*displacement[2])*.5
        if self.free(mid,np.linalg.norm(after[:3]-before[:3])*.5,
                     lateral+2*.72*abs(math.sin(angle*.25)),abs(after[1]-before[1])*.5,boxes,margin):
            return True
        if abs(angle)<1e-8:
            return False
        # Refine a rejected rotation envelope, preserving a continuous sweep
        # certificate and the same margin. Each midpoint bounds the full arc
        # of its sub-action with the exact endpoint chord bound. Sixteen is a
        # fixed bound, used only when the single enclosing sweep rejected.
        subdivisions=16
        for i in range(subdivisions):
            a=before+(after-before)*(i/subdivisions)
            b=before+(after-before)*((i+1)/subdivisions)
            a[3]=before[3]+angle*(i/subdivisions)
            b[3]=before[3]+angle*((i+1)/subdivisions)
            middle=(a+b)*.5
            delta=b-a
            lateral=abs(math.sin(middle[3])*delta[0]+math.cos(middle[3])*delta[2])*.5
            if not self.free(middle,np.linalg.norm(b[:3]-a[:3])*.5,
                            lateral+2*.72*abs(math.sin(angle/(4*subdivisions))),
                            abs(b[1]-a[1])*.5,boxes,margin):
                return False
        return True

    def clearance_margin(self,pose,boxes):
        if not self.free(pose,boxes=boxes,margin=.002):
            return 0.
        lo,hi=.002,.06
        for _ in range(6):
            mid=(lo+hi)*.5
            if self.free(pose,boxes=boxes,margin=mid):
                lo=mid
            else:
                hi=mid
        return lo


@njit(cache=True)
def obstacle_raster(triangles,support,origin,resolution):
    """Stage-mesh column occupancy for search only; selected motion uses BVH.

    This broad approximation intentionally includes furniture at torso height
    and walls, even where NavMesh connectivity permits a point path.
    """
    height,width=support.shape
    occupied=np.zeros((height,width),np.bool_)
    for tri in triangles:
        xmin=max(0,int(math.floor((np.min(tri[:,0])-origin[0])/resolution)))
        xmax=min(width-1,int(math.ceil((np.max(tri[:,0])-origin[0])/resolution)))
        zmin=max(0,int(math.floor((np.min(tri[:,2])-origin[1])/resolution)))
        zmax=min(height-1,int(math.ceil((np.max(tri[:,2])-origin[1])/resolution)))
        if xmax<xmin or zmax<zmin:
            continue
        for iz in range(zmin,zmax+1):
            for ix in range(xmin,xmax+1):
                y=support[iz,ix]
                if np.isnan(y):
                    # Outside the center NavMesh, retain walls that constrain
                    # the body at neighboring navigable cells.
                    ys=[]
                    for dz in range(-4,5):
                        for dx in range(-4,5):
                            zz,xx=iz+dz,ix+dx
                            if 0<=zz<height and 0<=xx<width and not np.isnan(support[zz,xx]):
                                ys.append(support[zz,xx])
                    if not len(ys):
                        continue
                    y=np.median(np.asarray(ys))
                if np.max(tri[:,1])<y+.20 or np.min(tri[:,1])>y+1.:
                    continue
                lo=np.array([origin[0]+ix*resolution,y+.20,origin[1]+iz*resolution])
                hi=np.array([lo[0]+resolution,y+1.,lo[2]+resolution])
                if triangle_box(tri,lo,hi):
                    occupied[iz,ix]=True
    return occupied
