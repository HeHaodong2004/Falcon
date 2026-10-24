"""One endpoint-indexed, approximate optical law, shared by actual/coarse sensing.

The persistent two-valued visibility regime is part of each joint hypothesis.
No complete geometry or human-empty certificate is inferred from graph cells.
Exact positions have zero measurement noise. Misses have nonzero likelihood.
"""
import numpy as np

class EndpointObservationLaw:
    version='endpoint_optical_mixture_1'
    regime_prior=(.35,.65)
    # Global regime represents persistent unknown occlusion/detectability.
    in_view=(.12,.85)
    out_view=(.005,.025)
    hidden_detection=(.003,.015)
    birth_sigma=10.

    def __init__(self,parameters=None):
        if parameters is None:return
        for key in ('in_view','out_view','regime_prior'):
            value=np.asarray(parameters[key],float)
            if value.shape!=(2,) or not np.isfinite(value).all() or np.any((value<=0)|(value>=1)):raise ValueError('invalid optical probabilities')
            setattr(self,key,tuple(value.tolist()))
        if abs(sum(self.regime_prior)-1)>1e-9:raise ValueError('invalid optical regime prior')
        if np.any(np.asarray(self.in_view)<self.out_view):raise ValueError('invalid optical sector ordering')

    def detection(self,positions,robot_pose,regime):
        p=np.asarray(positions);r=np.asarray(robot_pose)
        d=p-r[...,None,[0,2]] if r.ndim>1 else p-r[[0,2]]
        yaw=r[...,None,3] if r.ndim>1 else r[3]
        fw=d[...,0]*np.cos(yaw)-d[...,1]*np.sin(yaw)
        lat=np.abs(d[...,0]*np.sin(yaw)+d[...,1]*np.cos(yaw))
        # Smooth sector avoids claiming a certified optical boundary.
        gate=1/(1+np.exp(-np.clip(fw/.4,-30,30)))
        gate*=1/(1+np.exp(-np.clip((fw-lat)/.4,-30,30)))
        gate*=1/(1+np.exp(-np.clip((10-np.linalg.norm(d,axis=-1))/.8,-30,30)))
        regime=np.asarray(regime,dtype=int)
        lo=np.asarray(self.out_view)[regime];hi=np.asarray(self.in_view)[regime]
        if regime.ndim:
            shape=regime.shape+(1,)*(gate.ndim-regime.ndim);lo=lo.reshape(shape);hi=hi.reshape(shape)
        pd=lo+(hi-lo)*gate
        return np.where(np.isfinite(p[...,0]),pd,0.)

    def birth_density(self,z,robot_pose):
        d=np.asarray(z)-np.asarray(robot_pose)[[0,2]];v=self.birth_sigma**2
        return np.exp(-np.sum(d*d,axis=-1)/(2*v))/(2*np.pi*v)

    def no_detection(self,future,indices,poses,return_parts=False):
        """P(no detection at ANY evaluated endpoint | shared world), root excluded."""
        result=np.ones(len(future.worlds))
        before=result.copy();last=result.copy()
        if len(indices)<=1:return (result,before,last) if return_parts else result
        regimes=np.array([h.visibility_mode for h in future.worlds])
        pd=self.detection(future.positions[:,indices[1:]],poses[1:],regimes)
        per_endpoint=np.log1p(-pd).sum(-1)
        result=np.exp(per_endpoint.sum(-1));before=np.exp(per_endpoint[:,:-1].sum(-1));last=np.exp(per_endpoint[:,-1])
        return (result,before,last) if return_parts else result

    def missed_moments(self,h,robot_pose):
        """Positive-weight position Gaussian quadrature; exact point-state limit.

        Integrates the same pointwise emission used by forecast worlds, and
        moment matches p(x,v | no detection). Quadrature error is measured.
        """
        n=h.n
        if h.maneuver_mean is not None:
            # Positive and missing detections must integrate the SAME retained
            # Gaussian mixture, rather than mixing a Gaussian likelihood with
            # the moment-matched marginal used only for state summaries.
            from .maneuver import moment_match
            K=h.maneuver_mean.shape[1]
            flat=h.fork(mean=h.maneuver_mean.reshape(-1,4),covariance=h.maneuver_covariance.reshape(-1,4,4),maneuver_mean=None)
            mass,mean,cov=self.missed_moments(flat,robot_pose)
            mass=mass.reshape(n,K);prior=h.maneuver_probability
            total=np.sum(prior*mass,axis=1);posterior=prior*mass/total[:,None]
            mean,cov=moment_match(mean.reshape(n,K,4),cov.reshape(n,K,4,4),posterior)
            return total,mean,cov
        from .hidden import missed_hidden_moments
        # Integrate positions only and preserve conditional velocity moments.
        # Smooth, small support uses 9x9 nodes; broad aged support uses15x15.
        # This removes the measured4-D axis-rule normalization distortion.
        order=9 if not n or np.max(np.linalg.eigvalsh(h.covariance[:,:2,:2]))<=.25 else 15
        return missed_hidden_moments(h.mean,h.covariance,robot_pose,h.visibility_mode,self,order=order)
