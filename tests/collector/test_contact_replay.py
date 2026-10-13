"""Exact contact multiset serialization must not hide physical changes."""
import copy
import json
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from collector.core import digest

class ContactReplay(unittest.TestCase):
    def test_permutation_only_and_duplicate_preservation(self):
        # The runtime applies this sort at capture, before root and branch hashes.
        source=(Path(__file__).resolve().parents[2]/'scripts/collector/runtime.py').read_text()
        self.assertIn('contacts.sort(key=lambda contact: json.dumps(contact, sort_keys=True, allow_nan=False))',source)
        a=[dict(object_id_a=2,contact_distance=-.1),dict(object_id_a=1,contact_distance=-.2)]
        a.append(copy.deepcopy(a[0]));b=list(reversed(a))
        for contacts in (a,b):contacts.sort(key=lambda c:json.dumps(c,sort_keys=True,allow_nan=False))
        self.assertEqual(digest(a),digest(b));self.assertEqual(len(a),3)
        b[0]['contact_distance']+=1e-10
        self.assertNotEqual(digest(a),digest(b))
