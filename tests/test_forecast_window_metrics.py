import unittest
from eventforecastwm.window_evaluate import aggregate


class TestWindowWeighting(unittest.TestCase):
    def test_missing_windows_and_root_balance(self):
        rows=[dict(root_id='a',scene='x',energy=2.),dict(root_id='a',scene='x',energy=4.),
              dict(root_id='a',scene='x'),dict(root_id='b',scene='y',energy=9.)]
        result=aggregate(rows,['energy'])
        self.assertEqual([r['energy'] for r in result],[3.,9.])
        self.assertEqual([r['energy_windows'] for r in result],[2,1])
        self.assertEqual(sum(r['energy'] for r in result)/len(result),6.)


if __name__=='__main__':unittest.main()
