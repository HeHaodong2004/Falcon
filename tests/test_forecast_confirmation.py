import unittest
from eventforecastwm.confirmation import select


class TestConfirmation(unittest.TestCase):
    def test_fixed_membership_excludes_entire_old_plan(self):
        blocks=[dict(split=s,root=dict(root_id=s+str(i))) for s in ('train','natural_reference','unseen_evaluation') for i in range(20)]
        old=dict(seed=3,counts={'train':4,'unseen_evaluation':6,'natural_reference':0})
        selected,banks,excluded=select(blocks,old,10)
        self.assertEqual(len(selected),20)
        self.assertFalse({b['root']['root_id'] for b in selected}&excluded)
        self.assertEqual(banks,select(list(reversed(blocks)),old,10)[1])
        self.assertEqual(len(excluded),10)
        self.assertFalse(any(b['split']=='train' for b in selected))


if __name__=='__main__':unittest.main()
