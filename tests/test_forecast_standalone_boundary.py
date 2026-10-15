from pathlib import Path
import tempfile
import unittest

from eventforecastwm.performance_v2.standalone_verify import permitted


class TestStandaloneBoundary(unittest.TestCase):
    def test_explicit_inputs_code_and_runtime_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'worktree/scripts';runtime=root/'python';output=root/'proof'
            inputs={root/'release/primary.pt',root/'release/history.json'}
            check=lambda p:permitted(p,inputs,output,source,runtime)
            self.assertTrue(check(root/'release/primary.pt'))
            self.assertTrue(check(source/'eventforecastwm/inference.py'))
            self.assertTrue(check(runtime/'lib/library.so'))
            self.assertTrue(check(output/'forecast.npz'))
            self.assertFalse(check(source/'future.pt'))
            self.assertFalse(check(root/'cache_full/root.pt'))
            self.assertFalse(check(root/'release/hidden_state.json'))
            source.mkdir(parents=True)
            (source/'leak.py').symlink_to(root/'cache_full/root.pt')
            self.assertFalse(check(source/'leak.py'))


if __name__=='__main__':unittest.main()
