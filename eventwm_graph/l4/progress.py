"""Projection of actual poses, ordered waypoint visits, and orientation progress."""
import math
import numpy as np
from eventwm_graph.partial.query import wrap


class RouteProgress:
    def __init__(self, reference):
        self.query=reference.query; self.route=np.asarray(reference.nominal_waypoints)
        self.lengths=np.linalg.norm(np.diff(self.route,axis=0),axis=1)
        self.cumulative=np.r_[0.,np.cumsum(self.lengths)]
        self.progress=0.; self.deviation=0.; self.segment=0
        self.next_waypoint=1 if len(self.route)>1 else 0
        self.rejoined=True; self.detours=0; self.rejoins=0; self.path_length=0.
        self.previous=None; self.best=0.; self.last_meaningful=self.query.timestamp
        self.best_yaw_error=math.inf
        self.best_goal_distance=math.inf

    def update(self, pose, timestamp):
        pose=np.asarray(pose); point=pose[:3]
        if self.previous is not None: self.path_length+=float(np.linalg.norm(point-self.previous[:3]))
        self.previous=pose.copy()
        while self.next_waypoint<len(self.route)-1:
            i=self.next_waypoint; a=self.route[i-1]; b=self.route[i]
            v=b-a; length=float(np.linalg.norm(v)); unit=v/max(length,1e-12)
            along=float(np.dot(point-a,unit)); lateral=float(np.linalg.norm(point-a-along*unit))
            reached=np.linalg.norm(point-b)<=self.query.position_tolerance
            # Crossing the waypoint plane in the same narrow route corridor is
            # a legitimate rejoin; requiring an exact sampled disc visit can
            # send an avoiding robot back toward an already passed waypoint.
            passed=along>=length and lateral<=.25 and along-length<=1.2
            # Rejoin immediately beyond an adjacent corner, never jump to an
            # unrelated later hairpin or replace the selected polyline.
            w=self.route[i+1]-b; wl=float(np.linalg.norm(w)); wu=w/max(wl,1e-12)
            next_along=float(np.dot(point-b,wu))
            next_lateral=float(np.linalg.norm(point-b-next_along*wu))
            corner=(along>=max(0.,length-.25) and 0.<next_along<=min(wl,1.2)
                    and next_lateral<=.20 and np.dot(unit,wu)>-.5)
            if not (reached or passed or corner):break
            self.next_waypoint+=1
        if len(self.route)==1: self.progress=0.; self.deviation=float(np.linalg.norm(point-self.route[0]))
        else:
            # Restrict projection to already visited ordered route and current leg.
            # At a hairpin an unvisited later segment cannot create fictitious progress.
            end=min(self.next_waypoint,len(self.route)-1)
            a=self.route[:end]; v=self.route[1:end+1]-a
            u=np.clip(np.sum((point-a)*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-12),0,1)
            distances=np.linalg.norm(a+u[:,None]*v-point,axis=1)
            self.segment=int(np.argmin(distances)); self.deviation=float(distances[self.segment])
            self.progress=float(self.cumulative[self.segment]+u[self.segment]*self.lengths[self.segment])
        if self.progress>self.best+.1: self.best=self.progress; self.last_meaningful=timestamp
        goal_distance=float(np.linalg.norm(point-self.query.target))
        # Measured rejoining can improve endpoint distance after projected
        # arclength saturates. A new best is required, so back-and-forth motion
        # cannot keep renewing this progress clock.
        if goal_distance<self.best_goal_distance-.1:
            self.best_goal_distance=goal_distance;self.last_meaningful=timestamp
        near=goal_distance<=self.query.position_tolerance
        yaw_error=abs(wrap(pose[3]-self.query.terminal_orientation))
        if near and yaw_error<self.best_yaw_error-.04:
            self.best_yaw_error=yaw_error; self.last_meaningful=timestamp
        if self.deviation>.32 and self.rejoined: self.rejoined=False; self.detours+=1
        if self.deviation<.20 and not self.rejoined: self.rejoined=True; self.rejoins+=1
        return self.complete(pose)

    def complete(self,pose):
        return (self.next_waypoint==len(self.route)-1 and
                np.linalg.norm(np.asarray(pose[:3])-self.query.target)<=self.query.position_tolerance and
                abs(wrap(pose[3]-self.query.terminal_orientation))<=self.query.orientation_tolerance)

    def target(self, lookahead=1.2):
        if len(self.route)==1: return self.route[0].copy()
        # Lookahead never cuts an unvisited ordered corner.
        index=self.next_waypoint; a=self.route[index-1]; b=self.route[index]
        distance=self.cumulative[index-1]
        u=np.clip((self.progress+lookahead-distance)/max(self.lengths[index-1],1e-12),0.,1.)
        return a+u*(b-a)
