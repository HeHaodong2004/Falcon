import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from collector.branch_storage import iter_branches, write_branches, branch_path


class LosslessBranchStorage(unittest.TestCase):
    def test_roundtrip_exact_json_bytes_and_duplicates(self):
        rows=[{'timestamp':0.048666666666666664,'contacts':[{'distance':-1e-12}]*2,'sr_label':0},
              {'timestamp':1.000000000000001,'sr_label':None,'unicode':'é'}]
        expected=''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows).encode()
        with tempfile.TemporaryDirectory() as d:
            for compressed in (False,True):
                root=Path(d)/str(compressed);path=write_branches(root,rows,compressed)
                self.assertEqual(list(iter_branches(root)),rows)
                actual=gzip.decompress(path.read_bytes()) if compressed else path.read_bytes()
                self.assertEqual(actual,expected)
                before=path.read_bytes()
                with self.assertRaises(ValueError):write_branches(root,[],compressed)
                self.assertEqual(path.read_bytes(),before)

    def test_corruption_and_ambiguous_streams_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=write_branches(root,[{'x':1}],True)
            data=bytearray(path.read_bytes());data[-8]^=1;path.write_bytes(data)
            with self.assertRaises((gzip.BadGzipFile,EOFError)):list(iter_branches(root))
            (root/'branch_truth/branches.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'exactly one'):branch_path(root)



class CompressedCompaction(unittest.TestCase):
    def test_canonical_compressed_truth_survives_verified_worker_compaction(self):
        from collector.core import write
        from collector.storage import compact_workers
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[dict(branch_id='b000',truth=[{'native_human_contact':True}])]
            path=write_branches(root,rows,True);before=path.read_bytes()
            write(root/'audits/workers/b000/branch.json',rows[0])
            write(root/'labels/labels.json',[{'branch_id':'b000'}])
            compact_workers(root)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(list(iter_branches(root)),rows)
            self.assertFalse((root/'audits/workers/b000/branch.json').exists())

if __name__=='__main__':unittest.main()
