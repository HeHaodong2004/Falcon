import copy
from pathlib import Path
import sys
import unittest
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from eventwm_wm_train_index import balance_weights,index_dataset,verify_job_declaration
from collector.core import write
from collector.v3 import PROTOCOL
from collector.static_feasibility import CHECKER_VERSION


def sample(root,group,label,mixed):
    return dict(root_id=root,group_id=group,social_outcome_class=label,mixed_outcome_group=mixed)


class TrainingIndexTests(unittest.TestCase):
    def test_collision_contrasts_preferred_within_three_class_mixed_budget(self):
        rows=[]
        for k in range(3):
            for mixed,binary in [(False,False),(True,True)]+([(True,False)] if k<2 else []):
                r=sample(str((k,mixed,binary)),str((k,mixed,binary)),k,mixed)
                r['mixed_binary_collision_group']=binary;rows.append(r)
        w,info=balance_weights(rows)
        for k,mass in enumerate((.3,.3,.4)):
            self.assertAlmostEqual(sum(p for p,r in zip(w,rows) if r['social_outcome_class']==k),mass)
        self.assertAlmostEqual(sum(p for p,r in zip(w,rows) if r['mixed_outcome_group']),.45)
        self.assertAlmostEqual(sum(p for p,r in zip(w,rows) if r['mixed_binary_collision_group']),.40)
        self.assertAlmostEqual(info['achieved_binary_mixed_mass'],.40)
    def test_declared_small_workload_prefix(self):
        jobs=[dict(root_id=str(i),scene='castle',seed=i,recipe={}) for i in range(3)]
        verify_job_declaration(jobs[:1],jobs,dict(job_limit=1,roots=[dict(root_id='0',scene='castle',seed=0)]))
        with self.assertRaisesRegex(ValueError,'job declaration'):
            verify_job_declaration(jobs[:1],jobs,dict(roots=[]))
    def test_root_must_match_job_not_just_seal(self):
        jobs=[dict(root_id='r',scene='castle',seed=1,recipe={})]
        with self.assertRaisesRegex(ValueError,'metadata differs'):
            verify_job_declaration(jobs,jobs,dict(roots=[dict(root_id='r',scene='other',seed=1)]))
        with self.assertRaisesRegex(ValueError,'Undeclared'):
            verify_job_declaration(jobs,jobs,dict(roots=[dict(root_id='unlisted')]))
    def test_requested_margins_when_feasible_without_mutating_labels(self):
        rows=[sample(str(k)+str(m),str(k)+str(m),k,m) for k in range(3) for m in (False,True)]
        before=copy.deepcopy(rows);w,info=balance_weights(rows)
        self.assertEqual(rows,before);self.assertAlmostEqual(sum(w),1.)
        for k,target in enumerate((.3,.3,.4)):
            self.assertAlmostEqual(sum(p for p,r in zip(w,rows) if r['social_outcome_class']==k),target)
        self.assertAlmostEqual(sum(p for p,r in zip(w,rows) if r['mixed_outcome_group']),.45)
    def test_no_fabricated_mixed_groups(self):
        rows=[sample(str(k),str(k),k,False) for k in range(3)]
        w,info=balance_weights(rows)
        self.assertEqual(info['achieved_mixed_mass'],0.)
        self.assertEqual(info['feasible_mixed_interval'],[0,0])
    def test_missing_class_explicit_not_invented(self):
        w,info=balance_weights([sample('r','g',0,False)])
        self.assertEqual(w,[1.]);self.assertEqual(info['missing_classes'],['SOCIAL_VIOLATION_SUCCESS','HUMAN_COLLISION'])
    def test_many_branches_cannot_dominate_root_within_stratum(self):
        rows=[sample('a','ga',0,False)]*20+[sample('b','gb',0,False)]
        w,_=balance_weights(rows)
        self.assertAlmostEqual(sum(w[:20]),.5);self.assertAlmostEqual(w[-1],.5)
    def test_feasibility_projection_reported(self):
        rows=[sample('a','ga',0,False),sample('b','gb',1,False),sample('c','gc',2,True)]
        w,info=balance_weights(rows)
        self.assertAlmostEqual(info['achieved_mixed_mass'],.4)
    def test_empty_index_explicit(self):
        w,info=balance_weights([]);self.assertEqual(w,[]);self.assertEqual(len(info['missing_classes']),3)
    def test_natural_holdout_cannot_become_training(self):
        with tempfile.TemporaryDirectory() as d:
            write(Path(d)/'manifest.json',dict(protocol_version=PROTOCOL,checker_version=CHECKER_VERSION,purpose='natural_sampling_pilot'))
            with self.assertRaisesRegex(ValueError,'purpose mismatch'):index_dataset(d,'train')
    def test_targeted_roots_cannot_become_natural_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            write(Path(d)/'manifest.json',dict(protocol_version=PROTOCOL,checker_version=CHECKER_VERSION,purpose='wm_train_targeted_pilot'))
            with self.assertRaisesRegex(ValueError,'purpose mismatch'):index_dataset(d,'natural_holdout')

if __name__=='__main__':unittest.main()
