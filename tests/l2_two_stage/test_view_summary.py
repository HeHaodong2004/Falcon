"""Exact view-cache regression against the preserved full-history computation."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from eventwm_graph.l2_two_stage.context import OnlineContext
from eventwm_graph.l2_two_stage.data import read,selected_branch
from eventwm_graph.l2_two_stage.features import extract,historical_view_evidence
from eventwm_graph.l2_two_stage.types import SealedObservation
from eventwm_graph.l2_two_stage.view import ViewSummary

BASE=Path(__file__).resolve().parents[2]
REVIEW=BASE/'docs/l2/two_stage_20261023/reviews'


def reference_module():
    name='eventwm_graph.l2_two_stage._review_features_before_view'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,REVIEW/'features_before_view.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def frame(t,points=(),valid=True,available=True,occupied=False):
    tracks=[dict(track_id=str(i),position=list(p),estimated_velocity=[0.,0.,0.],first_seen_time=t,
        last_seen_time=t,num_observations=1,observation_age=0.,valid=True,velocity_valid=False,visible_now=True) for i,p in enumerate(points)]
    updates=[] if not valid else [dict(cell=[x,z],height=0.,source='visible_native_cspace_square_and_clearance_halo',
        state=1,timestamp=t) for x in range(-5,6) for z in range(-10,0)]
    if occupied:updates.append(dict(cell=[0,-5],height=0.,source='observed_static_wall_inflated_025m',state=2,timestamp=t))
    return dict(timestamp=t,robot_pose=[0.,0.,0.,0.],camera_origin=[0.,1.,0.],camera_rotation=np.eye(3).tolist(),
        tracks=tracks,static_updates=updates,sensor_protocol='ideal_frustum_observed_cspace_dynamic_occlusion_1',
        observation_valid=valid,human_sensor_available=available)


class ViewSummaryTests(unittest.TestCase):
    def test_empty_invalid_unavailable_and_boundary_views_are_exact(self):
        raw=[frame(0.,[(0.,0.,-1.)]),frame(.048,occupied=True),frame(.104,valid=False),frame(.152,available=False)]
        history=tuple(SealedObservation.from_raw(ob) for ob in raw);view=ViewSummary.from_frames(history)
        points=np.r_[np.random.default_rng(19).uniform([-2.,-5.],[2.,1.],(512,2)),
                     [[.5,-1.],[np.nextafter(.5,0.),-1.],[np.nextafter(.5,np.inf),-1.],[-.5,-1.],
                      [1.,-1.],[0.,-1e-6],[0.,-10.],[np.nextafter(1.,np.inf),-1.]]]
        old=reference_module().historical_view_evidence(history,points,0.,.2)
        new=historical_view_evidence(history,points,0.,.2,view)
        np.testing.assert_array_equal(new,old)
        self.assertEqual(new.dtype,old.dtype)
        np.testing.assert_array_equal(view.valid,[True,True,False,False])

    def test_cache_is_deeply_immutable_and_append_is_causal(self):
        raw=frame(0.,[(0.,0.,-1.)]);view=ViewSummary.from_frames([raw]);before=view.positive_xy.copy()
        raw['tracks'][0]['position'][0]=999.
        np.testing.assert_array_equal(view.positive_xy,before)
        for field in ('timestamps','origins','rotations','valid','packed_supported','code_offsets','codes','positive_offsets','positive_xy'):
            with self.assertRaises(ValueError):getattr(view,field).setflags(write=True)
        later=view.append(frame(.048));self.assertEqual(view.frame_count,1);self.assertEqual(later.frame_count,2)
        with self.assertRaisesRegex(ValueError,'noncausal'):view.append(frame(0.))
        with self.assertRaisesRegex(ValueError,'uncertified'):ViewSummary().validate_certificate()
        bad=frame(.1);bad['privileged_future']=[1,2]
        with self.assertRaisesRegex(ValueError,'unknown observation'):view.append(bad)

    def test_pack_overflow_uses_unchanged_full_reference(self):
        raw=frame(0.);raw['static_updates'][0]['cell']=[2**31+1,0]
        history=(SealedObservation.from_raw(raw),);view=ViewSummary.from_frames(history)
        self.assertFalse(view.packed_supported[0])
        points=np.array([[.2,-1.],[.0,-.5]])
        np.testing.assert_array_equal(historical_view_evidence(history,points,0.,.2,view),
            reference_module().historical_view_evidence(history,points,0.,.2))

    def test_real_native_context_updates_features_and_pilot_N_worlds_match(self):
        torch.set_num_threads(1)
        from eventwm_graph.l2_two_stage.conditioned import HistoryConditionedWorldModel
        folder=REVIEW/'runtime_200_30_pilot_k_bounded_unique_20261023'
        protocol=json.loads((folder/'protocol.json').read_text());rows=json.loads((folder/'inventory.json').read_text())['metadata']
        byid={row['root_id']:row for row in rows}
        ids=list(dict.fromkeys(protocol['representative_roots'][:2]+protocol['dense_roots'][:2]))
        model=HistoryConditionedWorldModel(BASE/'output/l2_two_stage_20261023/experiments/pilot_n_deterministic_history_eta05/best.pt')
        compared=0
        for rid in ids:
            path=Path(byid[rid]['folder']);legal=read(path/'legal_root.json.gz');branch=selected_branch(path)[2]
            context=OnlineContext.from_raw(legal['history'],legal['partial_graph'],legal['history_actions']).with_association()
            sequence=[None]+branch[:4]
            for node in sequence:
                if node is not None:context=context.ingest(node['observation'],node['action'])
                new=extract(context);old=reference_module().extract(context)
                for field in ('x','graph','global_graph','global_features','origin','velocity','age','last_positions','last_times','robotxy'):
                    np.testing.assert_array_equal(new[field],old[field],err_msg=field)
                    self.assertEqual(new[field].dtype,old[field].dtype)
                self.assertEqual(context.view_summary.frame_count,len(context.history))
                self.assertEqual(context.view_summary.timestamp,context.history[-1]['timestamp'])
                if node is None or node is sequence[-1]:
                    for f in (new,old):f['positive_history']=context._positive_history
                    left=model.predict_features(old,seed=721,association_graph=context._association_graph)
                    right=model.predict_features(new,seed=721,association_graph=context._association_graph)
                    for field in ('positions','present','weights','times','unseen','source_event_indices','previously_observed'):
                        np.testing.assert_array_equal(getattr(left,field),getattr(right,field),err_msg=field)
                    for field in ('matching','event_person','physical_seen','row_order','first_edge','auxiliary_log_probability'):
                        np.testing.assert_array_equal(getattr(left.association,field),getattr(right.association,field),err_msg=field)
                compared+=1
        self.assertGreaterEqual(compared,12)


if __name__=='__main__':unittest.main()
