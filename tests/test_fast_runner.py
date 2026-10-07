import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, os.environ['EVENTWM_JSON_VENDOR'])
import orjson
from eventwm_fast_runner import encode_metadata, install_overrides, write_metadata


class FastCheckpointTests(unittest.TestCase):
    def test_lossless_json_values(self):
        value = {'unicode': '测试\u2028', 'floats': [1e-100, 1e100, -0.0, 0.1, 1.2345678901234567],
                 'nested': [{'seed': 410000, 'safe': True, 'invalid': None}], 'tuple': (1, 2)}
        expected = json.loads(json.dumps(value, allow_nan=False))
        self.assertEqual(json.loads(encode_metadata(value, orjson)), expected)

    def test_fallback_preserves_stdlib_edge_values(self):
        for value in ({'large': 2**100}, {'surrogate': '\ud800'}, {1: 'integer key'}, {'tuple': (1, 2)}):
            self.assertEqual(json.loads(encode_metadata(value, orjson)), json.loads(json.dumps(value)))

    def test_nonfinite_never_becomes_null(self):
        for value in ([float('nan')], {'x': [float('inf')]}, {1: float('-inf')}):
            with self.assertRaises(ValueError):
                encode_metadata(value, orjson)

    def test_atomic_replacement_and_failed_write_preserves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            write_metadata(path, {'roots': [1, 2]}, orjson)
            previous = path.read_bytes()
            with self.assertRaises(ValueError):
                write_metadata(path, {'bad': float('nan')}, orjson)
            self.assertEqual(path.read_bytes(), previous)
            write_metadata(path, {'roots': [1, 2, 3]}, orjson)
            self.assertEqual(json.loads(path.read_bytes()), {'roots': [1, 2, 3]})
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_trace_and_aggregate_preserve_frozen_scientific_identity(self):
        original = {'frozen_commit': 'worker-commit', 'source_hashes': {'collector': 'frozen'}, 'roots': [1]}
        mass = SimpleNamespace(init=lambda args: original, aggregate=lambda manifest: {'count': len(manifest['roots'])})
        install_overrides(mass, orjson, {'launcher_commit': 'launcher-commit'})
        result = mass.init(None)
        self.assertEqual(result['frozen_commit'], 'worker-commit')
        self.assertEqual(result['source_hashes'], {'collector': 'frozen'})
        self.assertEqual(result['orchestration_history'][0]['launcher_commit'], 'launcher-commit')
        with tempfile.TemporaryDirectory() as directory:
            mass.save(Path(directory), result)
            self.assertEqual(json.loads((Path(directory) / 'manifest.json').read_bytes()), result)
            self.assertEqual(json.loads((Path(directory) / 'aggregate.json').read_bytes()), {'count': 1})


if __name__ == '__main__':
    unittest.main()
