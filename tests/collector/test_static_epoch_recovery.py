"""Source changes must preserve IDs, recipes, attempts and completed provenance."""
import copy, json, sys, tempfile, unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/eventwm_production'))
from recover_static_epoch import transition, OLD, CHANGED, ADDED
from control import C
import settings

class EpochRecoveryTests(unittest.TestCase):
    def test_transition_preserves_completed_and_validated_old_roots(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'validated').mkdir();(out/'validated/VALIDATED.json').write_text('{}')
            m=dict(frozen_commit=OLD,source_hashes={k:'old' for k in CHANGED},pipeline_settings=C['pipeline_settings'],
                   master_seed=731000000,protected_scientific_hashes={},production_source_epochs=[{'commit':OLD}],
                   next_sequence=30829,max_attempts=2)
            release=dict(implementation_hashes={k:'new' for k in CHANGED|ADDED},configuration={'master_seed':731000000},protected_scientific_hashes={})
            roots=[dict(root_id='completed',state='completed',source_commit=OLD,attempts=[{'number':1,'path':'completed'}]),
                   dict(root_id='validated',state='running',source_commit=OLD,attempts=[{'number':1,'path':'validated'}]),
                   dict(root_id='retry',state='running',source_commit=OLD,attempts=[{'number':1,'path':'retry'}],recipe={'seed':1}),
                   dict(root_id='preparation_failed',state='failed',source_commit=OLD,attempts=[{'number':1,'reason':'preparation_failure'}])]
            original=copy.deepcopy((m,roots))
            new,updates=transition(m,roots,release,'new_epoch',out)
            self.assertEqual((m,roots),original)
            self.assertEqual([b['root_id'] for b,_ in updates],['retry'])
            self.assertEqual([a['path'] for a in updates[0][0]['attempts']],[a['path'] for a in updates[0][1]['attempts']])
            self.assertEqual(updates[0][1]['state'],'pending')
            self.assertEqual(updates[0][1]['attempts'][-1]['reason'],'scientific_epoch_transition')
            self.assertEqual(updates[0][0]['recipe'],updates[0][1]['recipe'])
            self.assertEqual(new['next_sequence'],m['next_sequence'])
            self.assertEqual(new['pipeline_settings'],m['pipeline_settings'])
            self.assertEqual(new['production_source_epochs'][0],{'commit':OLD})
    def test_unrelated_source_change_is_refused(self):
        m=dict(frozen_commit=OLD,source_hashes={'labels.py':'old'})
        release=dict(implementation_hashes=dict({'labels.py':'new'},**{p:'hash' for p in ADDED}))
        with self.assertRaisesRegex(RuntimeError,'Unexpected scientific'):
            transition(m,[],release,'new_epoch',Path('/tmp'))

    def test_recovery_publication_baseline_excludes_old_epoch_staged_roots(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);receipt=out/'recovery_epochs/new_epoch/APPLIED.json'
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps(dict(applied_at=1234,retained_new_after_old_publication=73524)))
            with patch.object(settings,'DATA',out),patch.object(settings,'CONFIG',dict(recovery_gate='gate',scientific_commit='new_epoch')):
                self.assertEqual(settings.publication_epoch(),dict(time=1234,retained_before_resume=73524))

    def test_original_publication_baseline_remains_available(self):
        with tempfile.TemporaryDirectory() as td:
            project=Path(td);receipt=project/'output/eventwm_72h_v4_resume_checkpoint.json'
            receipt.parent.mkdir();receipt.write_text(json.dumps(dict(time=100,retained_before_resume=12276)))
            with patch.object(settings,'PROJECT',project),patch.object(settings,'CONFIG',{}):
                self.assertEqual(settings.publication_epoch(),dict(time=100,retained_before_resume=12276))

if __name__=='__main__':unittest.main()
