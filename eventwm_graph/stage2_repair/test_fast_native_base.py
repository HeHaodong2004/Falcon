import copy
import unittest
import numpy as np
from .range_benchmark import workloads
from .fast_native_base import FastNativeBase
from eventwm_graph.stage2_segment.native import Base
from eventwm_graph.stage2_segment.prepared import CompiledGraph


class NativeMetadataParity(unittest.TestCase):
    def test_all_original32_workloads_columns_headers_and_sources(self):
        for ex,packet,tracks in workloads():
            graph=ex['graph'];prepared=CompiledGraph(graph);old=Base(graph,prepared);new=FastNativeBase(graph,prepared)
            for name in ['keys','states','heights','first','last','source_ids']:np.testing.assert_array_equal(getattr(old.table,name),getattr(new.table,name))
            self.assertEqual(old.table.sources,new.table.sources);self.assertEqual(old.headers,new.headers);self.assertEqual(old.valid_topology,new.valid_topology);np.testing.assert_array_equal(old.known_codes,new.known_codes)

    def test_metadata_extensions_and_unusual_values_keep_legacy_fallback(self):
        graph=copy.deepcopy(workloads()[0][0]['graph']);graph['cells'][0]['extra']='preserved'
        for cls in [Base,FastNativeBase]:
            with self.assertRaises(TypeError):cls(graph,CompiledGraph(graph))
        del graph['cells'][0]['extra'];graph['cells'][0]['first_seen']=str(graph['cells'][0]['first_seen']);p=CompiledGraph(graph)
        np.testing.assert_array_equal(Base(graph,p).table.first,FastNativeBase(graph,p).table.first)


if __name__=='__main__':unittest.main()
