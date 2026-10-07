import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, os.environ['EVENTWM_JSON_VENDOR'])
import orjson
from eventwm_fast_runner import (_FrozenDict, _FrozenList, encode_metadata,
                                 freeze_terminal_roots)


class TerminalMetadataTests(unittest.TestCase):
    def root(self, state='completed', reason=None):
        return dict(state=state, attempts=[dict(state=state, reason=reason)],
                    stats={'values': [0.1, -0.0, 1e-100, 1e100, True, None]},
                    recipe={'seed': 410000}, query_ids=['q_example'])

    def test_identical_json_and_recursive_immutability(self):
        original = self.root()
        manifest = {'roots': [original]}
        before = encode_metadata(manifest, orjson)
        freeze_terminal_roots(manifest)
        root = manifest['roots'][0]
        self.assertIsInstance(root, _FrozenDict)
        self.assertEqual(before, encode_metadata(manifest, orjson))
        self.assertEqual(json.loads(before), manifest)
        operations = [lambda: root.update(state='failed'),
                      lambda: root.__setitem__('state', 'failed'),
                      lambda: root.__delitem__('state'), lambda: root.clear(),
                      lambda: root.pop('state'), lambda: root.popitem(),
                      lambda: root.setdefault('extra', 1),
                      lambda: root.__ior__({'extra': 1}),
                      lambda: root['stats']['values'].append(float('nan')),
                      lambda: root['stats']['values'].__setitem__(slice(None), []),
                      lambda: root['stats']['values'].__delitem__(0),
                      lambda: root['stats']['values'].extend([1]),
                      lambda: root['stats']['values'].insert(0, 1),
                      lambda: root['stats']['values'].pop(),
                      lambda: root['stats']['values'].remove(0.1),
                      lambda: root['stats']['values'].reverse(),
                      lambda: root['stats']['values'].sort(),
                      lambda: root['stats']['values'].clear(),
                      lambda: root['stats']['values'].__iadd__([1]),
                      lambda: root['stats']['values'].__imul__(2)]
        for operation in operations:
            with self.assertRaises(TypeError):
                operation()
        root.__init__({'bad': float('nan')})
        root['stats']['values'].__init__([float('nan')])
        original['stats']['values'].append(float('nan'))
        self.assertEqual(before, encode_metadata(manifest, orjson))

    def test_retryable_and_active_objects_stay_mutable(self):
        roots = [self.root('running'), self.root('pending'),
                 self.root('failed', 'worker_timeout'),
                 self.root('failed', 'root_timeout'),
                 self.root('failed', 'interrupted_before_commit')]
        manifest = {'roots': list(roots)}
        freeze_terminal_roots(manifest)
        for original, current in zip(roots, manifest['roots']):
            self.assertIs(original, current)
            current['attempts'].append({'state': 'running'})
            current['state'] = 'running'

    def test_nonretryable_failure_is_cached_once(self):
        manifest = {'roots': [self.root('failed', 'insufficient_static_feasible_queries')]}
        before = encode_metadata(manifest, orjson)
        freeze_terminal_roots(manifest)
        root = manifest['roots'][0]
        freeze_terminal_roots(manifest)
        self.assertIs(root, manifest['roots'][0])
        self.assertEqual(before, encode_metadata(manifest, orjson))

    def test_nonfinite_is_rejected_before_caching(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            root = self.root()
            root['stats']['values'].append(value)
            with self.assertRaises(ValueError):
                freeze_terminal_roots({'roots': [root]})
            with self.assertRaises(ValueError):
                _FrozenDict({'bad': value})
            with self.assertRaises(ValueError):
                _FrozenList([value])

    def test_edge_values_keep_stdlib_fallback(self):
        root = self.root()
        root.update(big=2**100, surrogate='\ud800', tuple=(1, 2))
        manifest = {'roots': [root]}
        expected = json.loads(json.dumps(manifest))
        freeze_terminal_roots(manifest)
        self.assertEqual(json.loads(encode_metadata(manifest, orjson)), expected)


if __name__ == '__main__':
    unittest.main()
