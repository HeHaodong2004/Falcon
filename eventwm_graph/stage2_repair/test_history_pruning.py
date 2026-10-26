import unittest
import numpy as np
from .range_planar import select_keyframes


class HistoryPruning(unittest.TestCase):
    def test_stationary_intermediate_measurements_can_be_retained(self):
        origins=np.zeros((4,3));rotations=np.repeat(np.eye(3)[None],4,0)
        self.assertEqual(select_keyframes(origins,rotations),[0,3])
        self.assertEqual(select_keyframes(origins,rotations,0,0),[0,1,2,3])

    def test_default_matches_original_motion_rule(self):
        rng=np.random.default_rng(937)
        for size in [1,2,17,104]:
            origins=np.cumsum(rng.normal(0,.02,(size,3)),0);rotations=np.repeat(np.eye(3)[None],size,0)+rng.normal(0,.02,(size,3,3));expected=[]
            for i in range(size):
                if expected and i<size-1 and np.linalg.norm(origins[i]-origins[expected[-1]])<.07 and np.max(np.abs(rotations[i]-rotations[expected[-1]]))<.07:continue
                expected.append(i)
            self.assertEqual(select_keyframes(origins,rotations),expected)


if __name__=='__main__':unittest.main()
