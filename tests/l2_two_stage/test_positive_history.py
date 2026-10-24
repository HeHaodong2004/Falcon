"""Causal legal-set cache tests, including unchanged real development anchors."""
import copy
import gzip
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from eventwm_graph.l2_two_stage import positive_history as ph
from eventwm_graph.l2_two_stage.common import OUT,OLD


def frame(t,points=(),valid=True,available=True):
    return dict(timestamp=t,observation_valid=valid,human_sensor_available=available,sensor_protocol=ph.SENSOR,
        tracks=[dict(track_id='untrusted_'+str(i),position=list(p),visible_now=True,valid=True,last_seen_time=t) for i,p in enumerate(points)])


class PositiveHistoryTests(unittest.TestCase):
    def fixture(self):
        history=[frame(0.,[(1.,0.,2.),(1.,0.,2.)]),frame(.056)]
        branch=[dict(observation=frame(.104,[(1.2,0.,2.)]))]
        anchors=[dict(anchor_index=0,timestamp=.056,query_id='q',scene_id='s'),dict(anchor_index=1,timestamp=.104,query_id='q',scene_id='s')]
        return ph.pack_frames(history,branch,anchors,'q','r','s')

    def test_exact_multiset_preserves_colocated_multiplicity(self):
        record=self.fixture();w=ph.history_at_anchor(record,0)
        self.assertEqual(w.minimum_count,2)
        np.testing.assert_array_equal(w.positive_xyz,[[1.,0.,2.],[1.,0.,2.]])
        self.assertEqual(w.positive_xyz.dtype,np.dtype('float64'))

    def test_future_root_frames_cannot_leak_into_earlier_anchor(self):
        record=self.fixture();before=ph.history_at_anchor(record,0)
        changed=copy.deepcopy(record);changed['timestamps'][-1]=1000.;changed['positive_xyz'][-1]=[999.,888.,777.]
        after=ph.history_at_anchor(changed,0)
        for key in ('timestamps','positive_offsets','positive_xyz','observation_valid','human_sensor_available'):
            np.testing.assert_array_equal(getattr(before,key),getattr(after,key))
        self.assertEqual(before.timestamps[-1],.056)

    def test_output_arrays_cannot_be_reenabled_or_changed_by_backing_input(self):
        record=self.fixture();w=ph.history_at_anchor(record,0);record['positive_xyz'][0]=[-1.,-2.,-3.]
        np.testing.assert_array_equal(w.positive_xyz[0],[1.,0.,2.])
        for key in ('timestamps','positive_offsets','positive_xyz','observation_valid','human_sensor_available'):
            with self.assertRaises(ValueError):getattr(w,key).setflags(write=True)

    def test_unavailable_and_invalid_frames_are_distinct_from_valid_empty(self):
        history=[frame(0.,valid=False),frame(.048,available=False),frame(.104)]
        packed=ph.pack_frames(history,[],[dict(anchor_index=0,timestamp=.104,query_id='q',scene_id='s')],'q','r','s')
        w=ph.history_at_anchor(packed,0)
        np.testing.assert_array_equal(w.observation_valid,[False,True,True])
        np.testing.assert_array_equal(w.human_sensor_available,[True,False,True])
        self.assertEqual(len(w.positive_xyz),0)

    def test_anonymous_id_permutation_cannot_change_witness(self):
        ob=frame(0.,[(1.,0.,2.),(-3.,0.,4.)]);reference=ph.positive_xyz(ob)
        ob['tracks'].reverse()
        for tr in ob['tracks']:tr['track_id']='arbitrary_new_key'
        np.testing.assert_array_equal(ph.positive_xyz(ob),reference)

    def test_noncausal_prefix_or_wrong_anchor_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'clocks'):
            ph.pack_frames([frame(.1),frame(.1)],[],[],'q','r','s')
        with self.assertRaisesRegex(ValueError,'anchor'):
            ph.pack_frames([frame(.1)],[],[dict(anchor_index=1,timestamp=.1,query_id='q',scene_id='s')],'q','r','s')

    def test_real_native_anchors_match_full_decode_without_privileged_files(self):
        prepared=json.loads((OUT/'prepared/dev/manifest.json').read_text());anchors={}
        for shard in prepared['shards']:
            for row in np.load(OUT/shard['path'],allow_pickle=True):
                rid=row['root_id']
                if len(anchors)<4 or rid in anchors:
                    anchors.setdefault(rid,[]).append({k:row[k] for k in ('anchor_index','timestamp','query_id','scene_id')})
            if len(anchors)>=4:break
        meta=json.loads((OLD/'snapshot/manifest.json').read_text());rows={r['root_id']:r for r in meta['rows'] if r['split']=='dev'}
        original_read=Path.read_bytes
        def guarded_read(path):
            if path.name.startswith('privileged_'):raise AssertionError('privileged payload opened')
            return original_read(path)
        total=0
        with patch.object(Path,'read_bytes',guarded_read):
            for rid,samples in anchors.items():
                row=dict(rows[rid],folder=str(OLD/'snapshot/roots'/rid));packed=ph.root_record((row,samples));folder=Path(row['folder'])
                legal=json.loads(gzip.decompress((folder/'legal_root.json.gz').read_bytes()))
                queries=json.loads(gzip.decompress((folder/'queries.json.gz').read_bytes()));outcomes=json.loads(gzip.decompress((folder/'outcomes.json.gz').read_bytes()))
                selected=max(range(len(queries)),key=lambda i:(len(queries[i]['compiled']['actions']),queries[i]['query_id']))
                nodes=json.loads(gzip.decompress((folder/'future_observations.json.gz').read_bytes()))
                reference=legal['history']+[nodes[i]['observation'] for i in outcomes[selected]['node_ids']]
                for sample in samples:
                    end=len(legal['history'])+sample['anchor_index'];w=ph.history_at_anchor(packed,sample['anchor_index'])
                    np.testing.assert_array_equal(w.timestamps,[ob['timestamp'] for ob in reference[:end]])
                    for i,ob in enumerate(reference[:end]):
                        np.testing.assert_array_equal(w.positive_xyz[w.positive_offsets[i]:w.positive_offsets[i+1]],ph.positive_xyz(ob))
                    self.assertEqual(w.timestamps[-1],sample['timestamp']);total+=1
                self.assertNotIn('track_id',packed)
        self.assertGreaterEqual(total,4)


if __name__=='__main__':unittest.main()
