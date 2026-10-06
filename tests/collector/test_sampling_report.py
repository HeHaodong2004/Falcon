"""Audit denominators and matching must not manufacture sampling evidence."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from eventwm_v3_sampling_report import matched_comparison,profile_statistics


def root(rid,seed,profile,state='completed'):
    return dict(root_id=rid,seed=seed,scene='castle',human_count=1,
        root_profile=profile,state=state,diagnostics=dict(
            nearest_social_distance_at_root=2.,already_in_social_violation_at_root=False,
            visible_tracks=0,lost_tracks=0))


def sample(rid,profile,label):
    return dict(root_id=rid,sampling_profile=profile,human_count=1,
        social_outcome_valid=True,social_outcome_name=label)


class SamplingReportTests(unittest.TestCase):
    def test_root_and_branch_denominators_are_separate(self):
        report=dict(roots=[root('a',1,'natural'),root('b',2,'natural')],
            samples=[sample('a','natural','SAFE_SUCCESS')]*4+
                    [sample('b','natural','HUMAN_COLLISION')])
        row=profile_statistics(report)['natural']
        self.assertEqual(row['completed_human_roots'],2)
        self.assertEqual(row['branch_outcomes']['count'],5)
        self.assertEqual(row['branch_outcomes']['rates']['SAFE_SUCCESS'],.8)

    def test_censored_root_is_in_attempts_not_distance_or_class_denominator(self):
        bad=root('b',2,'natural','failed');bad['diagnostics']=None
        row=profile_statistics(dict(roots=[root('a',1,'natural'),bad],
            samples=[sample('a','natural','SAFE_SUCCESS')]))['natural']
        self.assertEqual(row['attempted_human_roots'],2)
        self.assertEqual(row['completed_human_roots'],1)
        self.assertEqual(row['mean_psc_distance'],2.)
        self.assertEqual(row['branch_outcomes']['count'],1)

    def test_matching_requires_same_episode_and_completed_both_sides(self):
        natural=dict(roots=[root('n1',1,'natural'),root('n2',2,'natural','failed'),root('n3',3,'natural')],
            samples=[sample('n1','natural','SAFE_SUCCESS'),sample('n3','natural','HUMAN_COLLISION')])
        coverage=dict(roots=[root('c1',1,'seek_lost'),root('c2',2,'seek_lost'),root('c4',4,'seek_lost')],
            samples=[sample('c1','seek_lost','SOCIAL_VIOLATION_SUCCESS'),
                     sample('c2','seek_lost','HUMAN_COLLISION'),sample('c4','seek_lost','SAFE_SUCCESS')])
        row=matched_comparison(natural,coverage)['seek_lost']
        self.assertEqual(row['completed_pairs'],1)
        self.assertEqual(row['profiles']['natural']['branch_outcomes']['rates']['SAFE_SUCCESS'],1.)
        self.assertEqual(row['profiles']['seek_lost']['branch_outcomes']['rates']['SOCIAL_VIOLATION_SUCCESS'],1.)


if __name__=='__main__':unittest.main()
