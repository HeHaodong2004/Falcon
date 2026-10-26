import unittest
import numpy as np
import torch
from .range_benchmark import workloads
from .range_bev import RangeCanvas
from .range_canvas_fast import FastRangeCanvas
from .global_features import GlobalPrepared


class ExactCanvasTest(unittest.TestCase):
    def test_all_input_channels_equal_for_real_maps(self):
        torch.set_num_threads(1);old=RangeCanvas('cuda:0');fast=FastRangeCanvas('cuda:0')
        for index in [0,4,12,20,31]:
            ex,packet,tracks=workloads()[index];g=GlobalPrepared(ex['graph']);p=np.asarray(ex['paths'][0]['trajectory'])
            args=(g.keys,g.values,g.heights,p[0,1:4],g.resolution,packet['ranges'],packet['origins'],packet['rotations'],packet['timestamps'],tracks)
            a,alow=old.build(*args);b,blow=fast.build(*args)
            np.testing.assert_array_equal(alow,blow);np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
