import copy
import unittest
import numpy as np
from .fast_native_base import FastColumnsMixin
from .range_benchmark import workloads
from eventwm_graph.stage2_segment.batch import SharedCompiledGraph


class FastCompiled(FastColumnsMixin,SharedCompiledGraph):pass


class FastColumns(unittest.TestCase):
    def compare(self,g):
        a=SharedCompiledGraph(g);b=FastCompiled(g)
        for name in ['keys','values','heights','height64','input_order','known_lo','known_hi']:np.testing.assert_array_equal(getattr(a,name,None),getattr(b,name,None),err_msg=name)
        for name in ['identity','native_eligible','uniform','resolution','height']:self.assertEqual(getattr(a,name),getattr(b,name))

    def test_all32_real_maps(self):
        for ex,_,_ in workloads():self.compare(ex['graph'])

    def test_unusual_valid_and_invalid_fallback(self):
        graph=copy.deepcopy(workloads()[0][0]['graph'])
        for mode in ['array_key','float_key','float_state','string_height','empty']:
            g=copy.deepcopy(graph)
            if mode=='array_key':g['cells'][0]['cell']=np.asarray(g['cells'][0]['cell'])
            elif mode=='float_key':g['cells'][0]['cell']=[float(x)for x in g['cells'][0]['cell']]
            elif mode=='float_state':g['cells'][0]['state']=float(g['cells'][0]['state'])
            elif mode=='string_height':g['cells'][0]['height']=str(g['cells'][0]['height'])
            else:g['cells']=[]
            self.compare(g)
        for mode in ['overflow','duplicate','nonfinite','bad_state']:
            g=copy.deepcopy(graph)
            if mode=='overflow':g['cells'][0]['cell']=[2**40,0]
            elif mode=='duplicate':g['cells'].append(copy.deepcopy(g['cells'][0]))
            elif mode=='nonfinite':g['cells'][0]['height']=np.nan
            else:g['cells'][0]['state']=0
            for cls in [SharedCompiledGraph,FastCompiled]:
                with self.assertRaises(ValueError):cls(g)


if __name__=='__main__':unittest.main()
