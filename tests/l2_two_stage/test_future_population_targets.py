"""Native all-population target packing; this is offline supervision only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from eventwm_graph.l2_two_stage import future_population_targets as ft
from eventwm_graph.l2_two_stage.common import OUT,OLD


class FutureTargetTests(unittest.TestCase):
    def test_current_overlay_preserves_colocated_multiplicity_and_raw_truth(self):
        native=np.array([[1.+4e-7,2.],[1.+4e-7,2.],[9.,10.]],np.float64);original=native.copy()
        observed=np.array([[1.,2.],[1.,2.]],np.float64)
        overlay,unobserved=ft.current_measurement_overlay(native,observed)
        np.testing.assert_array_equal(native,original)
        np.testing.assert_array_equal(overlay[:2],observed)
        np.testing.assert_array_equal(unobserved,[[9.,10.]])
        self.assertEqual(len(overlay),len(native))

    def test_exact_native_indices_and_censoring(self):
        times=np.array([0.,.048,.104,.152],np.float64)
        self.assertEqual(ft.native_index(times,.104),2)
        with self.assertRaisesRegex(ValueError,'uniquely native'):ft.native_index(times,.1)

    def make_chunk(self,base,jobs):
        dest=base/'population_future_targets/test';dest.mkdir(parents=True)
        prepared=base/'prepared/test/manifest.json';prepared.parent.mkdir(parents=True);prepared.write_text('{}')
        identity=dict(prepared_manifest_sha256=ft.sha(prepared))
        with patch.multiple(ft,OUT=base,JOBS=jobs,DEST=dest,IDENTITY=identity):report=ft.chunk((0,len(jobs)))
        manifest=dict(schema=ft.SCHEMA,identity=identity,prepared_manifest=str(prepared),shards=[report])
        (dest/'manifest.json').write_text(json.dumps(manifest))
        return report

    def test_joint_actor_columns_and_valid_empty_population_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);jobs=[];records=[]
            for i,people in enumerate((2,0)):
                folder=base/f'root{i}';folder.mkdir();times=np.array([0.,.048,.104,.152]);positions=np.arange(4*people*3,dtype=float).reshape(4,people,3)
                source=folder/'privileged_future.npz';np.savez_compressed(source,timestamps=times,positions=positions)
                row=dict(root_id=f'r{i}',scene_id='s',folder=str(folder),files={'privileged_future.npz':ft.sha(source)})
                anchor=dict(anchor_index=1,timestamp=.048,times=np.array([.056,.104,0,0,0,0,0,0]),query_id='q')
                jobs.append((row,[anchor]));records.append(dict(anchor,root_id=row['root_id'],scene_id='s'))
            self.make_chunk(base,jobs)
            with patch.object(ft,'OUT',base):ft.attach_targets(records,'test')
            expected=np.arange(24,dtype=float).reshape(4,2,3)
            np.testing.assert_array_equal(records[0]['population_current_positions'],expected[1][...,[0,2]])
            self.assertEqual(records[0]['population_current_positions'].dtype,np.float64)
            np.testing.assert_array_equal(records[0]['population_future_positions'][:2],expected[[2,3]][...,[0,2]])
            np.testing.assert_array_equal(records[0]['population_future_valid'],[True,True,False,False,False,False,False,False])
            self.assertEqual(records[1]['population_future_positions'].shape,(8,0,2))
            self.assertTrue(records[1]['population_future_valid'][:2].all())

    def test_real_native_targets_match_every_supported_unchanged_horizon(self):
        prepared=json.loads((OUT/'prepared/dev_native_v2/manifest.json').read_text());rows=np.load(OUT/prepared['shards'][0]['path'],allow_pickle=True)
        rid=rows[0]['root_id'];samples=[copy.deepcopy(r) for r in rows if r['root_id']==rid]
        manifest=json.loads((OLD/'snapshot/manifest.json').read_text());row=next(r for r in manifest['rows'] if r['root_id']==rid);row=dict(row,folder=str(OLD/'snapshot/roots'/rid))
        anchors=[{k:r[k] for k in ('anchor_index','timestamp','times','query_id')} for r in samples]
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);self.make_chunk(base,[(row,anchors)])
            with patch.object(ft,'OUT',base):ft.attach_targets(samples,'test')
            with np.load(Path(row['folder'])/'privileged_future.npz',allow_pickle=False) as tape:
                for sample in samples:
                    current=ft.native_index(tape['timestamps'],sample['timestamp'])
                    np.testing.assert_array_equal(sample['population_current_positions'],tape['positions'][current][...,[0,2]])
                    for h,delta in enumerate(sample['times']):
                        self.assertEqual(bool(sample['population_future_valid'][h]),bool(delta>0))
                        if delta>0:
                            index=ft.native_index(tape['timestamps'],sample['timestamp']+delta)
                            np.testing.assert_array_equal(sample['population_future_positions'][h],tape['positions'][index][...,[0,2]])


if __name__=='__main__':unittest.main()
