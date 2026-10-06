import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from collector.core import read,write
from collector.storage import write_compact_json,write_prefix_truth
from eventwm_mass_collection import external_roots

class MetadataStorageTests(unittest.TestCase):
    def test_prefix_gzip_preserves_original_bytes(self):
        rows=[{'timestamp':0.048666666666666664,'contacts':[{'d':-1e-12}]*2,'value':None}, {'x':'é','safe':True}]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plain=root/'plain';packed=root/'packed'
            write_prefix_truth(plain,rows)
            write_prefix_truth(packed,rows,compressed=True)
            self.assertEqual(gzip.decompress((packed/'roots/prefix_truth.json.gz').read_bytes()),(plain/'roots/prefix_truth.json').read_bytes())
            self.assertEqual(json.loads(gzip.decompress((packed/'roots/prefix_truth.json.gz').read_bytes())),rows)
    def test_compact_manifest_is_identical_json_value(self):
        value={'floats':[1e-12,-0.0,0.048666666666666664],'rows':[{'label':None,'valid':False}],'text':'a b é'}
        with tempfile.TemporaryDirectory() as d:
            plain=Path(d)/'plain.json';compact=Path(d)/'compact.json'
            write(plain,value);write_compact_json(compact,value)
            self.assertEqual(read(plain),read(compact));self.assertLess(compact.stat().st_size,plain.stat().st_size)
    def test_external_budget_only_accepts_a_draining_predecessor(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'old';args=SimpleNamespace(drain_from=old,output=Path(d)/'new')
            previous={'state':'draining_user_stop','roots':[{'state':'running'},{'state':'running'},{'state':'completed'}]}
            write(old/'manifest.json',previous);self.assertEqual(external_roots(args),2)
            previous.update(state='stopped_by_user',roots=[]);write(old/'manifest.json',previous)
            self.assertEqual(external_roots(args),0)
            previous['state']='running';write(old/'manifest.json',previous)
            with self.assertRaises(RuntimeError):external_roots(args)
            with self.assertRaises(ValueError):external_roots(SimpleNamespace(drain_from=old,output=old))

if __name__=='__main__':unittest.main()
