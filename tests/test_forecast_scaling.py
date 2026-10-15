import unittest
from eventforecastwm.scaling import validate_partitions


def block(rid,split,scene,episode,trajectory):
    return dict(root=dict(root_id=rid),split=split,rows=[dict(query_id=rid+'/q',scene=scene,
        episode_id=episode,trajectory_id=trajectory)])


class TestScalingReservations(unittest.TestCase):
    def test_new_data_cannot_leak_or_duplicate_physical_roots(self):
        a=block('a','train','scene1','episode1','trajectory1')
        b=block('b','unseen_evaluation','scene2','episode2','trajectory2')
        self.assertEqual(validate_partitions([a,b])['training_episodes'],1)
        with self.assertRaisesRegex(ValueError,'unseen scenes'):
            validate_partitions([a,b,block('c','train','scene2','episode3','trajectory3')])
        with self.assertRaisesRegex(ValueError,'grouped sampling'):
            validate_partitions([a,b,block('c','train','scene1','episode1','trajectory3')])
        with self.assertRaisesRegex(ValueError,'cross-reservation'):
            validate_partitions([a,b,block('c','train','scene1','episode2','trajectory3')])


if __name__=='__main__':unittest.main()
