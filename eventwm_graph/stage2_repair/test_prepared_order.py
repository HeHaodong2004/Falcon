from pathlib import Path
import tempfile
import unittest
import numpy as np
from .range_online_data import prepared_order


class PreparedOrder(unittest.TestCase):
    def test_complete_snapshot_retained_when_receipts_arrive(self):
        with tempfile.TemporaryDirectory()as folder:
            records=[dict(sensor_path=str(Path(folder)/(str(i)+'.npz')))for i in range(6)];order=np.array([3,1,5,0,4,2])
            for i in [1,4,5]:Path(records[i]['sensor_path']).with_suffix('.json').touch()
            iterator=prepared_order(order,records,True);self.assertEqual(next(iterator),1);self.assertEqual(next(iterator),5)
            for i in [0,2,3]:Path(records[i]['sensor_path']).with_suffix('.json').touch()
            remainder=list(iterator);self.assertEqual(remainder,[4,3,0,2]);self.assertEqual(sorted([1,5]+remainder),list(range(6)))

    def test_all_prepared_and_original_mode_preserve_seeded_order(self):
        with tempfile.TemporaryDirectory()as folder:
            records=[dict(sensor_path=str(Path(folder)/(str(i)+'.npz')))for i in range(17)];order=np.random.default_rng(42).permutation(17);self.assertEqual(list(prepared_order(order,records,False)),order.tolist())
            for r in records:Path(r['sensor_path']).with_suffix('.json').touch()
            self.assertEqual(list(prepared_order(order,records,True)),order.tolist())


if __name__=='__main__':unittest.main()
