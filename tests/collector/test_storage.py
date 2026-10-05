"""Production compaction must never destroy the sole or mismatched raw truth."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from collector.core import read,write
from collector.storage import compact_workers


class StorageTests(unittest.TestCase):
    def fixture(self,p):
        bs=[dict(branch_id='b000',truth=[1]),dict(branch_id='b001',truth=[2])]
        for b in bs: write(p/'audits/workers'/b['branch_id']/'branch.json',b)
        (p/'branch_truth').mkdir()
        (p/'branch_truth/branches.jsonl').write_text('\n'.join(map(json.dumps,bs))+'\n')
        write(p/'labels/labels.json',[dict(branch_id=b['branch_id']) for b in bs])
        return bs

    def test_only_verified_duplicates_removed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); bs=self.fixture(p)
            write(p/'audits/workers/b000/observation_diagnostics.json',{'retained':True})
            original=(p/'branch_truth/branches.jsonl').read_bytes()
            report=compact_workers(p)
            self.assertEqual(len(report['removed_worker_records']),2)
            self.assertEqual((p/'branch_truth/branches.jsonl').read_bytes(),original)
            self.assertTrue((p/'audits/workers/b000/observation_diagnostics.json').exists())
            self.assertFalse(list(p.glob('audits/workers/*/branch.json')))

    def test_mismatch_removes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); self.fixture(p)
            write(p/'audits/workers/b001/branch.json',{'branch_id':'b001','truth':['wrong']})
            with self.assertRaises(RuntimeError): compact_workers(p)
            self.assertEqual(len(list(p.glob('audits/workers/*/branch.json'))),2)

    def test_missing_canonical_or_labels_removes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); write(p/'audits/workers/b000/branch.json',{'truth':[1]})
            with self.assertRaises(FileNotFoundError): compact_workers(p)
            self.assertTrue((p/'audits/workers/b000/branch.json').exists())


if __name__=='__main__': unittest.main()
