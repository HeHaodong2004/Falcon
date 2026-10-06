"""The audit must detect action overrides without constructing a simulator."""
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from eventwm_human_defaults_audit import action_differences,FILES,REPO,UPSTREAM


class HumanDefaultsAuditTests(unittest.TestCase):
    def test_equal_defaults_and_unrelated_robot_action(self):
        defaults={'human':{'threshold':-1.,'speed':10.}}
        self.assertEqual(action_differences(defaults,{**defaults,'robot':{'speed':2}}),{})

    def test_changed_missing_and_added_fields_detected(self):
        defaults={'human':{'threshold':-1.,'speed':10.}}
        for actual in ({},{'human':{'threshold':1.,'speed':10.}},
                       {'human':{'threshold':-1.,'speed':10.,'override':True}}):
            self.assertTrue(action_differences(defaults,actual))

    def test_native_sources_match_pinned_upstream(self):
        for name in FILES:
            with self.subTest(source=name):
                upstream=subprocess.check_output(['git','show',UPSTREAM+':'+name],cwd=REPO)
                self.assertEqual((REPO/name).read_bytes(),upstream)


if __name__=='__main__':unittest.main()
