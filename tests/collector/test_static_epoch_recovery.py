"""Source changes must preserve IDs, recipes, attempts and completed provenance."""
import copy, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/eventwm_production'))
from recover_static_epoch import transition, OLD, CHANGED
from control import C

class EpochRecoveryTests(unittest.TestCase):
    def test_transition_preserves_completed_and_validated_old_roots(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'validated').mkdir();(out/'validated/VALIDATED.json').write_text('{}')
            m=dict(frozen_commit=OLD,source_hashes={k:'old' for k in CHANGED},pipeline_settings=C['pipeline_settings'],
                   master_seed=731000000,protected_scientific_hashes={},production_source_epochs=[{'commit':OLD}],
                   next_sequence=30829,max_attempts=2)
            release=dict(implementation_hashes={k:'new' for k in CHANGED},configuration={'master_seed':731000000},protected_scientific_hashes={})
            roots=[dict(root_id='completed',state='completed',source_commit=OLD,attempts=[{'number':1,'path':'completed'}]),
                   dict(root_id='validated',state='running',source_commit=OLD,attempts=[{'number':1,'path':'validated'}]),
                   dict(root_id='retry',state='running',source_commit=OLD,attempts=[{'number':1,'path':'retry'}],recipe={'seed':1}),
                   dict(root_id='preparation_failed',state='failed',source_commit=OLD,attempts=[{'number':1,'reason':'preparation_failure'}])]
            original=copy.deepcopy((m,roots))
            new,updates=transition(m,roots,release,'new_epoch',out)
            self.assertEqual((m,roots),original)
            self.assertEqual([b['root_id'] for b,_ in updates],['retry'])
            self.assertEqual(updates[0][0]['attempts'],updates[0][1]['attempts'])
            self.assertEqual(updates[0][0]['recipe'],updates[0][1]['recipe'])
            self.assertEqual(new['next_sequence'],m['next_sequence'])
            self.assertEqual(new['pipeline_settings'],m['pipeline_settings'])
            self.assertEqual(new['production_source_epochs'][0],{'commit':OLD})
    def test_unrelated_source_change_is_refused(self):
        m=dict(frozen_commit=OLD,source_hashes={'labels.py':'old'})
        release=dict(implementation_hashes={'labels.py':'new'})
        with self.assertRaisesRegex(RuntimeError,'Unexpected scientific'):
            transition(m,[],release,'new_epoch',Path('/tmp'))

if __name__=='__main__':unittest.main()
