import copy
import unittest
import numpy as np
import torch
from eventforecastwm.data import window,batch_windows


def observation(t,x=0.):
    return dict(timestamp=t,robot_position=[0.,0.,0.],robot_yaw=0.,previous_command=[0.,0.],tracking_available=True,
        human_tracks=[dict(track_id='legal0',position=[x,0.,0.],estimated_velocity=[999.,0.,0.],
            visible_now=True,velocity_valid=False,valid=True,observation_age=0.,last_seen_time=t,
            first_seen_time=0.,num_observations=1)])


def truth(t,x=0.):
    return dict(timestamp=t,robot=dict(position=[0.,0.,0.],yaw=0.),humans=[dict(object_id=99,position=[x,0.,0.])])


class TestForecastData(unittest.TestCase):
    def test_future_observations_not_inputs(self):
        hist=[observation(t,t) for t in np.arange(0,2.01,.25)]
        frames=[truth(t,t) for t in np.arange(0,2.01,.25)]
        a=window(hist,frames,.5,2.0000001)
        altered=copy.deepcopy(hist)
        for h in altered:
            if h['timestamp']>.5:h['human_tracks'][0]['position']=[999.,0.,0.]
        b=window(altered,frames,.5,2.0000001)
        self.assertTrue(torch.equal(a['base'].humans,b['base'].humans))
        self.assertEqual(a['metadata']['timestamp'],.5)
        self.assertTrue(a['targets']['track_valid'][:6].all())
        self.assertFalse(a['targets']['track_valid'][6:].any())

    def test_contact_boundary_is_censored(self):
        hist=[observation(0.)];frames=[truth(t,t) for t in np.arange(0,2.01,.25)]
        w=window(hist,frames,0.,1.)
        self.assertEqual(w['targets']['physical_exposure'][:5].tolist(),[True,True,True,False,False])

    def test_zero_human_scene_has_exposure_but_no_trajectory(self):
        hist=[observation(0.)];hist[0]['human_tracks']=[]
        frames=[truth(t) for t in np.arange(0,2.01,.25)]
        for f in frames:f['humans']=[]
        w=window(hist,frames,0.,2.0000001);w['sensor_config']={};w['robot_geometry']={}
        b=batch_windows([w]);self.assertEqual(b['base'].humans.shape[1],1)
        self.assertTrue(b['targets']['physical_exposure'][:,:8].all())
        self.assertFalse(b['targets']['all_human_valid'].any())
        self.assertFalse(b['targets']['track_valid'].any())


if __name__=='__main__':unittest.main()
