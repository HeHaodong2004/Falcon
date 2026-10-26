import unittest
from . import test_fast_prepared_pipeline as parity
from .fast_point_api import FastPointPredictor


class FastPointAPI(unittest.TestCase):
    predictor_class=FastPointPredictor
    test_mixed_horizons_single_batch_and_fallback=parity.FastPreparedPipelineParity.test_mixed_horizons_single_batch_and_fallback


if __name__=='__main__':unittest.main()
