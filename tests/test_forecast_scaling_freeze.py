import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from eventforecastwm.performance_v2.scaling_freeze import eligibility,freeze,CAMPAIGN


class ScalingFreezeGateTest(unittest.TestCase):
    def test_verified_count_and_completed_physical_receipts_are_both_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);index=root/'verified.sqlite3';physical=root/'physical';physical.mkdir()
            with sqlite3.connect(index) as database:
                database.execute('CREATE TABLE roots (queries INTEGER)')
                database.execute('INSERT INTO roots VALUES (494967)')
            self.assertFalse(eligibility(index,[physical])['eligible_to_attempt_freeze'])
            with sqlite3.connect(index) as database:database.execute('INSERT INTO roots VALUES (33)')
            self.assertFalse(eligibility(index,[physical])['eligible_to_attempt_freeze'])
            (physical/'result.json').write_text(json.dumps({'status':'training_complete'}))
            evidence=eligibility(index,[physical]);self.assertTrue(evidence['eligible_to_attempt_freeze'])
            self.assertTrue(evidence['actual_audit_still_required'])
            self.assertEqual(evidence['verified_queries'],495000)

    def test_closed_gate_does_not_create_output_or_invoke_freezer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'new_snapshot'
            with patch('eventforecastwm.performance_v2.scaling_freeze.CAMPAIGN',root),\
                 patch('eventforecastwm.performance_v2.scaling_freeze.eligibility',return_value={'eligible_to_attempt_freeze':False}):
                with self.assertRaisesRegex(RuntimeError,'Scaling gate closed'):freeze(run,root/'parent',root/'index',[])
            self.assertFalse(run.exists())

    def test_output_cannot_replace_parent_or_production(self):
        for run in (CAMPAIGN,Path('/tmp/not-a-research-snapshot')):
            with self.assertRaisesRegex(ValueError,'new nested campaign output'):
                freeze(run,run,Path('/tmp/nonexistent-index'),[])


if __name__=='__main__':unittest.main()
