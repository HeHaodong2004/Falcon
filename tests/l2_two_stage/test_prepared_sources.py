"""Metadata regressions for immutable multi-generation preparation accounting."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eventwm_graph.l2_two_stage import data


class PreparedSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patcher=patch.object(data,'OUT',self.root);self.patcher.start()
        self.rows=[dict(root_id=f'root{i}',scene_id=f'scene{i}',seed=i,split='train',source_split='train',queries=1,folder=str(self.root/f'roots/root{i}')) for i in range(4)]
        self.reports=[dict(root_id=r['root_id'],scene_id=r['scene_id'],anchors=2,counts=dict(roots=1,anchors=2)) for r in self.rows]
        self.first=self.root/'prepared/first/manifest.json'
        shard=dict(path='prepared/first/000000_000002.npy',sha256='payload_one',anchors=4)
        self.write(self.root/Path(shard['path']).with_suffix('.json'),dict(sha256=shard['sha256'],anchors=4,roots=self.reports[:2]))
        self.write(self.first,dict(schema='two_stage_full_native_anchor_cache_1',roots=2,scenes=2,split='train',shards=[shard]))
        self.second=self.root/'prepared/second/manifest.json'
        delta=dict(path='prepared/second/delta_000000_000001.npy',sha256='payload_two',anchors=2)
        self.write(self.root/Path(delta['path']).with_suffix('.json'),dict(sha256=delta['sha256'],anchors=2,roots=self.reports[2:3]))
        self.snapshot=self.root/'frozen_cutoff/manifest.json';self.write(self.snapshot,dict(rows=self.rows))
        self.write(self.second,dict(schema='two_stage_extended_native_anchor_cache_1',roots=3,scenes=3,split='train',shards=[shard,delta],parent_manifest=str(self.first),snapshot=str(self.snapshot),identity=dict(parent_manifest_sha256=data.sha(self.first),snapshot_sha256=data.sha(self.snapshot))))

    def tearDown(self):
        self.patcher.stop();self.tmp.cleanup()

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def test_recursive_extended_parent_keeps_all_prior_roots(self):
        self.assertEqual(data.prepared_root_ledger(self.second),self.reports[:3])

    def test_named_cache_uses_exact_snapshot_and_exact_root_cohort(self):
        self.write(self.root/'latest_snapshot/manifest.json',dict(rows=[self.rows[3]]))
        self.assertEqual([r['root_id'] for _,r in data.jobs_for('second')],['root0','root1','root2'])
        self.assertEqual([r['root_id'] for _,r in data.jobs_for('train')],['root3'])

    def test_changed_parent_manifest_is_rejected(self):
        self.first.write_text(self.first.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'parent manifest hash'):data.prepared_root_ledger(self.second)

    def test_inherited_shard_substitution_is_rejected(self):
        value=json.loads(self.second.read_text());value['shards'][0]['sha256']='replaced';self.write(self.second,value)
        with self.assertRaisesRegex(ValueError,'inherited shard'):data.prepared_root_ledger(self.second)

    def test_hashed_ledger_rejects_duplicate_roots(self):
        value=json.loads(self.second.read_text());ledger=self.second.parent/'root_ledger.json.gz'
        with gzip.open(ledger,'wt') as f:json.dump([self.reports[0],self.reports[0],self.reports[2]],f)
        value['root_ledger']=dict(path=str(ledger.relative_to(self.root)),sha256=data.sha(ledger));self.write(self.second,value)
        with self.assertRaisesRegex(ValueError,'duplicate'):data.prepared_root_ledger(self.second)

    def test_snapshot_hash_or_protected_rows_are_rejected(self):
        value=json.loads(self.snapshot.read_text());value['rows'][3]['split']='confirmation';self.write(self.snapshot,value)
        with self.assertRaisesRegex(ValueError,'nontraining'):data.jobs_for('train',snapshot=self.snapshot)
        with self.assertRaisesRegex(ValueError,'nontraining'):data.jobs_for('second')


if __name__=='__main__':unittest.main()
