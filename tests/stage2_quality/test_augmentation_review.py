"""Independent scientific alignment/causality checks for frozen-cache augmentation.

All factual payloads are fitting-only. Synthetic altered formats exercise the
advertised conservative fallback, never alter original data or fitting caches.
"""
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.stage2.data import branches,read_member
from eventwm_graph.stage2_quality.augment import legal_input,executed_inputs,recover_keys,augment_root,without_static_arrays
from eventwm_graph.stage2_quality.data import sample
from eventwm_graph.stage2_quality.features import Features
from eventwm_graph.stage2_quality.common import ORIGINAL


def save_members(folder,values):
    checksums={}
    for name,value in values.items():
        raw=gzip.compress(json.dumps(value,sort_keys=True,separators=(',',':')).encode(),mtime=0)
        (Path(folder)/name).write_bytes(raw);checksums[name]=hashlib.sha256(raw).hexdigest()
    return dict(folder=str(folder),queries=len(values['queries.json.gz']),files=checksums)


def synthetic_legal():
    g=PartialGraph(.25,0.)
    g.update(dict(timestamp=1.,robot_pose=[0.,0.,0.,0.],static_updates=[dict(cell=[0,0],state=1,height=0.,timestamp=1.,source='measured')]))
    obs=dict(camera_origin=[0.,1.,0.],camera_rotation=np.eye(3).tolist(),observation_valid=True,robot_pose=[0.,0.,0.,0.],sensor_protocol='test',static_updates=[],timestamp=1.,tracks=[])
    return dict(history=[obs],history_actions=[],partial_graph=g.snapshot(),sensor={'max_range_m':10.})


class FrozenAugmentationReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest=ORIGINAL/'output/l2_two_stage_20261023/final_collection_20261024/manifest.json'
        cls.rows={r['root_id']:r for r in json.loads(manifest.read_text())['rows']}
        path=ORIGINAL/'output/stage2_graphdelta_20261024/prepared/train/manifest.json'
        cls.cached=json.loads(path.read_text());cls.shard=cls.cached['shards'][0]
        assert hashlib.sha256(Path(cls.shard['path']).read_bytes()).hexdigest()==cls.shard['sha256']
        with np.load(cls.shard['path'],allow_pickle=False) as a:cls.old={k:a[k] for k in a.files}
        cls.slices={};start=0
        for r in cls.shard['roots']:
            end=start+r['counts']['sampled_cells'];cls.slices[r['root_id']]=(start,end);start=end
        assert start==len(cls.old['y'])

    def test_all_100_first_shard_roots_fast_inputs_equal_full_json(self):
        for metadata in self.shard['roots']:
            row=self.rows[metadata['root_id']]
            legal=read_member(row['folder'],'legal_root.json.gz',row['files']['legal_root.json.gz'])
            graph,start,_=legal_input(row)
            self.assertEqual(graph,legal['partial_graph'])
            self.assertEqual(start,[legal['history'][-1]['timestamp'],*legal['history'][-1]['robot_pose']])
            nodes=read_member(row['folder'],'future_observations.json.gz',row['files']['future_observations.json.gz'])
            queries=read_member(row['folder'],'queries.json.gz',row['files']['queries.json.gz'])
            outcomes=read_member(row['folder'],'outcomes.json.gz',row['files']['outcomes.json.gz'])
            fast,_=executed_inputs(row)
            expected=[(q['query_id'],[[nodes[i]['observation']['timestamp'],*nodes[i]['observation']['robot_pose']] for i in o['node_ids']]) for q,o in zip(queries,outcomes)]
            self.assertEqual(fast,expected,row['root_id'])

    def test_fast_features_exactly_match_full_replay_on_selected_roots(self):
        ids=[self.shard['roots'][i]['root_id'] for i in (0,7,33,66,99)]
        zero=[r['root_id'] for r in self.shard['roots'] if r['counts']['new_free']+r['counts']['new_blocked']==0]
        ids=list(dict.fromkeys(ids+zero[:2]))
        for root in ids:
            row=self.rows[root];start,end=self.slices[root];old={k:v[start:end] for k,v in self.old.items()}
            actual,meta,ledger=augment_root(row,old);expected=[];cursor=0
            np.testing.assert_array_equal(actual[:,:47],old['x'])
            for ex,qmeta in zip(branches(row),meta):
                value,_=sample(ex,Features(ex['graph'],ex['trajectory']))
                self.assertEqual(qmeta['query_id'],ex['query_id']);self.assertEqual(qmeta['start'],cursor)
                size=len(value['y']);self.assertEqual(qmeta['end'],cursor+size)
                for name in ('y','height','weight'):np.testing.assert_array_equal(value[name],old[name][cursor:cursor+size])
                expected.append(value['x']);cursor+=size
            self.assertEqual(cursor,len(old['y']));self.assertEqual(len(meta),row['queries'])
            np.testing.assert_array_equal(actual,np.concatenate(expected),root)
            self.assertEqual(ledger['rows_verified'],len(old['y']))

    def test_key_recovery_handles_negative_and_far_origins_conservatively(self):
        for resolution,start in [(.25,[1.,-21.125,0.,-13.75,0.]),(.02,[1.,40.,0.,-16.,0.]),(.25,[1.,5000.,0.,-5000.,0.])]:
            p=np.asarray([start,[2.,start[1]+.3,0.,start[3]-.2,.3]])
            graph=synthetic_legal()['partial_graph'];graph['resolution']=resolution
            context=Features(graph,p);keys=context.candidates('baseline')[::127]
            x,_=context.transform(keys,'baseline');encoded=x.astype(np.float16)
            try:decoded=recover_keys(encoded,start,resolution)
            except ValueError:
                # Conservative fallback is explicitly supported at coarse precision.
                continue
            np.testing.assert_array_equal(decoded,keys)
        x=np.zeros((1,47),np.float16);x[:,36:38]=np.float16(20.)
        with self.assertRaises(ValueError):recover_keys(x,[1.,0.,0.,0.,0.],.02)

    def test_balanced_static_skipper_respects_nested_arrays_strings_and_escapes(self):
        static=[{'cell':[1,2],'nested':[[[1,2],[3,4]],[]],
            'text':'brackets [] ][, quote \" and backslash \\ and escaped \\\"'}]
        source=[{'observation':{'static_updates':static,'robot_pose':[1.,2.,3.,4.],
            'timestamp':5.,'metadata':{'static_updates':[{'nested':[[1],[2]],'text':']\\\"['}]}},
            'node_id':0,'parent':None}]
        raw=json.dumps(source,sort_keys=True,separators=(',',':')).encode()
        actual=json.loads(without_static_arrays(raw));expected=copy.deepcopy(source)
        expected[0]['observation']['static_updates']=[]
        expected[0]['observation']['metadata']['static_updates']=[]
        self.assertEqual(actual,expected)
        # Whitespace and empty/nonmatching trees remain valid, with all other fields unchanged.
        raw=json.dumps(source,indent=2).encode();self.assertEqual(json.loads(without_static_arrays(raw)),expected)
        self.assertEqual(without_static_arrays(b'[]'),b'[]')
        with self.assertRaises(ValueError):without_static_arrays(b'{"static_updates":[[1],"unterminated]')

    def test_unreviewed_nested_pose_uses_actual_observation_field(self):
        # Equal regex counts are insufficient if robot_pose is nested metadata.
        legal=synthetic_legal();ob=copy.deepcopy(legal['history'][0]);ob['timestamp']=2.
        ob['metadata']={'robot_pose':[0.,0.,0.,math.pi]}
        nodes=[dict(action='left',graph_delta={},node_id=0,observation=ob,parent=None)]
        values={'legal_root.json.gz':legal,'queries.json.gz':[{'query_id':'q'}],
            'outcomes.json.gz':[{'query_id':'q','node_ids':[0]}],'future_observations.json.gz':nodes}
        with tempfile.TemporaryDirectory() as folder:
            row=save_members(folder,values);actual,fallback=executed_inputs(row)
            self.assertEqual(actual,[('q',[[2.,0.,0.,0.,0.]])])

    def test_unreviewed_nested_timestamp_uses_observation_clock(self):
        legal=synthetic_legal();ob=copy.deepcopy(legal['history'][0]);ob['timestamp']=2.
        ob['synth']={'slots':[],'timestamp':1.5,'tracks':[]}
        nodes=[dict(action='left',graph_delta={},node_id=0,observation=ob,parent=None)]
        values={'legal_root.json.gz':legal,'queries.json.gz':[{'query_id':'q'}],
            'outcomes.json.gz':[{'query_id':'q','node_ids':[0]}],'future_observations.json.gz':nodes}
        with tempfile.TemporaryDirectory() as folder:
            row=save_members(folder,values);actual,_=executed_inputs(row)
            self.assertEqual(actual,[('q',[[2.,0.,0.,0.,0.]])])

    def test_unreviewed_nested_history_pose_uses_top_level_robot_pose(self):
        legal=synthetic_legal();legal['history'][0]['tracks']=[{'robot_pose':[4.,0.,5.,math.pi]}]
        values={'legal_root.json.gz':legal,'queries.json.gz':[{'query_id':'q'}],
            'outcomes.json.gz':[{'query_id':'q','node_ids':[]}],'future_observations.json.gz':[]}
        with tempfile.TemporaryDirectory() as folder:
            row=save_members(folder,values);graph,start,fallback=legal_input(row)
            self.assertEqual(start,[1.,0.,0.,0.,0.])  # legal_input always uses the standard JSON reader

    def test_short_or_absent_unknown_stratum_falls_back_to_exact_replay(self):
        for unknown_count in (0,37):
            legal=synthetic_legal();g=legal['partial_graph'];g['resolution']=1.
            path=np.array([[1.,0.,0.,0.,0.],[2.,0.,0.,0.,0.]])
            keys=Features(g,path).candidates('baseline')
            ob=copy.deepcopy(legal['history'][0]);ob['timestamp']=2.
            selected=keys if unknown_count==0 else keys[:-unknown_count]
            ob['static_updates']=[dict(cell=k.tolist(),state=1,height=0.,timestamp=2.,source='measured') for k in selected]
            nodes=[dict(action='left',graph_delta={},node_id=0,observation=ob,parent=None)]
            values={'legal_root.json.gz':legal,'queries.json.gz':[{'query_id':'q'}],
                'outcomes.json.gz':[{'query_id':'q','node_ids':[0],'summary':{'termination':'timeout'}}],'future_observations.json.gz':nodes}
            with tempfile.TemporaryDirectory() as folder:
                row=save_members(folder,values);row.update(root_id='short-stratum',scene_id='synthetic')
                ex=next(branches(row));expected,_=sample(ex,Features(ex['graph'],ex['trajectory']))
                self.assertEqual(int(np.count_nonzero(expected['y']==0)),unknown_count)
                old={k:v[:,:47] if k=='x' else v.copy() for k,v in expected.items()}
                actual,meta,ledger=augment_root(row,old)
                np.testing.assert_array_equal(actual,expected['x'])
                self.assertEqual(ledger['replayed_roots'],1);self.assertEqual(len(meta),1)

    def test_empty_dag_zero_positive_query_and_checksum_rejection(self):
        legal=synthetic_legal();values={'legal_root.json.gz':legal,'queries.json.gz':[{'query_id':'q'}],
            'outcomes.json.gz':[{'query_id':'q','node_ids':[],'summary':{'termination':'zero_duration'}}],'future_observations.json.gz':[]}
        with tempfile.TemporaryDirectory() as folder:
            row=save_members(folder,values);row.update(root_id='synthetic',scene_id='synthetic')
            path,_=executed_inputs(row);self.assertEqual(path,[('q',[])])
            ex=next(branches(row));reference,_=sample(ex,Features(ex['graph'],ex['trajectory']))
            old={k:v[:,:47] if k=='x' else v.copy() for k,v in reference.items()}
            augmented,meta,ledger=augment_root(row,old)
            np.testing.assert_array_equal(augmented,reference['x']);self.assertEqual(len(meta),1)
            bad=copy.deepcopy(row);bad['files']['future_observations.json.gz']='0'*64
            with self.assertRaises(ValueError):executed_inputs(bad)


if __name__=='__main__':unittest.main()
