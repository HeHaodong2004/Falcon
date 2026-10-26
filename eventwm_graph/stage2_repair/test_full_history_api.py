"""Same supplied-input contract and single/batched invariants with all frames."""
import unittest
from .test_fast_prepared_pipeline import FastPreparedPipelineParity
from .full_history_api import FullHistoryPredictor


class FullHistoryAPI(unittest.TestCase):
    predictor_class=FullHistoryPredictor
    test_mixed_horizons_single_batch_and_fallback=FastPreparedPipelineParity.test_mixed_horizons_single_batch_and_fallback


if __name__=='__main__':unittest.main()
