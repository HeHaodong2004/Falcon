"""Source transition preserves seals and gives interrupted work one epoch attempt."""
import copy
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/eventwm_production'))
import recover_static_epoch as epoch

class ScientificEpochTransition(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name)
        self.old=dict(frozen_commit=epoch.OLD,source_hashes={p:'old' for p in epoch.CHANGED},
            pipeline_settings=copy.deepcopy(epoch.C['pipeline_settings']),master_seed=731000000,
            protected_scientific_hashes={'reference':'immutable'},max_attempts=2,next_sequence=100)
        self.release=dict(implementation_hashes={p:'new' for p in epoch.CHANGED|epoch.ADDED},
            configuration={'master_seed':731000000},protected_scientific_hashes={'reference':'immutable'})
        self.root=dict(root_id=epoch.FAILED_ROOT,state='running',source_commit=epoch.OLD,pid=42,
            attempts=[dict(number=i,state='running' if i==2 else 'failed',path=f'roots/bad/attempt_{i:02d}') for i in (1,2)])
    def test_exhausted_attempts_preserved_and_one_repaired_epoch_attempt_scheduled(self):
        before=copy.deepcopy(self.root)
        new,changes=epoch.transition(self.old,[self.root],self.release,'repaired',self.output)
        old,after=changes[0]
        self.assertEqual(self.root,before)
        self.assertEqual(new['max_attempts'],2)
        self.assertEqual(after['state'],'pending')
        self.assertEqual(len(after['attempts']),2)
        self.assertEqual(after['attempts'][1]['path'],before['attempts'][1]['path'])
        self.assertEqual(after['attempts'][1]['reason'],'scientific_epoch_transition')
        self.assertEqual(after['source_epoch_history'][0]['attempt_numbers'],[1,2])
    def test_completed_and_validated_old_epoch_roots_unchanged(self):
        completed=dict(self.root,state='completed',query_ids=['sealed'])
        validated=copy.deepcopy(self.root)
        path=self.output/validated['attempts'][-1]['path'];path.mkdir(parents=True)
        (path/'VALIDATED.json').write_text('{}')
        before=copy.deepcopy([completed,validated])
        new,changes=epoch.transition(self.old,[completed,validated],self.release,'repaired',self.output)
        self.assertEqual(changes,[])
        self.assertEqual([completed,validated],before)
    def test_unexpected_source_inventory_refused(self):
        self.release['implementation_hashes']['unrelated']='x'
        with self.assertRaisesRegex(RuntimeError,'inventory'):
            epoch.transition(self.old,[self.root],self.release,'repaired',self.output)
    def test_changed_seed_refused(self):
        self.release['configuration']['master_seed']=0
        with self.assertRaisesRegex(RuntimeError,'seed'):
            epoch.transition(self.old,[self.root],self.release,'repaired',self.output)
