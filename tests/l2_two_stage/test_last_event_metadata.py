"""Legal last-event bookkeeping, explicitly distinct from physical identity."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from eventwm_graph.l2_two_stage import last_event_metadata as lm
from eventwm_graph.l2_two_stage import positive_history as ph
from eventwm_graph.l2_two_stage.common import OUT,OLD


def frame(t,rows,available=True):
    return dict(timestamp=t,observation_valid=True,human_sensor_available=available,sensor_protocol=ph.SENSOR,
        tracks=[dict(track_id=key,position=position,visible_now=True,valid=True,last_seen_time=t,estimated_velocity=velocity,velocity_valid=velocity is not None) for key,position,velocity in rows])


class LastEventTests(unittest.TestCase):
    def test_swapped_anonymous_keys_remain_observation_indices(self):
        history=[frame(0.,[('a',[0.,0.,0.],None),('b',[1.,0.,0.],None)])]
        branch=[dict(observation=frame(.048,[('a',[1.,0.,0.],[9.,0.,0.]),('b',[0.,0.,0.],[-9.,0.,0.])]))]
        anchors=[dict(anchor_index=0,timestamp=0.,query_id='q',scene_id='s',keys=['a','b']),dict(anchor_index=1,timestamp=.048,query_id='q',scene_id='s',keys=['a','b'])]
        primary=ph.pack_frames(history,branch,anchors,'q','r','s');events=lm.pack_events(history,branch,anchors,primary)
        np.testing.assert_array_equal(events['anchors'][1]['source_last_event_indices'],[3,2])
        np.testing.assert_array_equal(events['anchors'][1]['source_last_raw_xyz'],[[1.,0.,0.],[0.,0.,0.]])
        np.testing.assert_array_equal(events['anchors'][1]['source_last_estimated_velocity'],[[9.,0.,0.],[-9.,0.,0.]])
        self.assertNotIn('keys',events['anchors'][1])
        self.assertEqual(events['anchors'][0]['positive_event_prefix_end'],2)
        np.testing.assert_array_equal(events['anchors'][0]['source_last_timestamps'],[0.,0.])

    def test_unavailable_frame_does_not_replace_last_actual_positive(self):
        history=[frame(0.,[('a',[.123456789123,0.,0.],None)]),frame(.048,[('a',[999.,0.,0.],[99.,0.,0.])],False)]
        anchors=[dict(anchor_index=0,timestamp=.048,query_id='q',scene_id='s',keys=['a'])]
        primary=ph.pack_frames(history,[],anchors,'q','r','s');events=lm.pack_events(history,[],anchors,primary)
        self.assertEqual(events['anchors'][0]['source_last_raw_xyz'][0,0],.123456789123)
        self.assertEqual(events['anchors'][0]['source_last_event_indices'][0],0)
        self.assertFalse(events['anchors'][0]['source_last_velocity_valid'][0])

    def test_real_native_source_order_and_current_raw_coordinates(self):
        prepared=json.loads((OUT/'prepared/dev_native_v2/manifest.json').read_text());records=np.load(OUT/prepared['shards'][0]['path'],allow_pickle=True);rid=records[0]['root_id'];samples=[r for r in records if r['root_id']==rid]
        pm=json.loads((OUT/'positive_history/dev/manifest.json').read_text());packed=next(r for r in np.load(OUT/pm['shards'][0]['path'],allow_pickle=True) if r['root_id']==rid)
        sm=json.loads((OLD/'snapshot/manifest.json').read_text());source=next(r for r in sm['rows'] if r['root_id']==rid);source=dict(source,folder=str(OLD/'snapshot/roots'/rid))
        anchors=[{k:r[k] for k in ('anchor_index','timestamp','query_id','keys')} for r in samples]
        read=Path.read_bytes
        def guarded(path):
            if path.name.startswith('privileged_'):raise AssertionError('privileged input opened')
            return read(path)
        with patch.object(Path,'read_bytes',guarded):events=lm.root_record(source,anchors,packed)
        for sample,anchor in zip(samples,events['anchors']):
            index=anchor['source_last_event_indices'];w=ph.history_at_anchor(packed,sample['anchor_index'])
            self.assertTrue(np.all(index<len(w.positive_xyz)))
            np.testing.assert_array_equal(anchor['source_last_raw_xyz'].astype(np.float32)[:,[0,2]],sample['origin'])
            np.testing.assert_array_equal(anchor['source_last_raw_xyz'],w.positive_xyz[index])
            self.assertEqual(anchor['source_last_raw_xyz'].dtype,np.float64)


if __name__=='__main__':unittest.main()
