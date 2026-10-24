"""Actual native terrain guidance and explicit unsupported-cell counterexamples."""
import unittest
from pathlib import Path

import numpy as np

from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.l2_two_stage.context import (
    OnlineContext, _guidance_support, _segment_support,
)
from eventwm_graph.l2_two_stage.data import jobs_for, read
from eventwm_graph.l2_two_stage.features import extract
from eventwm_graph.l2_two_stage.types import LegalContext, freeze


def synthetic_grid(heights):
    """Geometric unit fixture; this is not a native sensor/forecast sample."""
    grid = PartialGraph(.2, 0., .25)
    grid.timestamp, grid.revision = 0., 1
    grid.cells = {
        (i, 0): dict(state=1, height=float(height), first_seen=0., last_seen=0.,
                    source='visible_native_cspace_square_and_clearance_halo')
        for i, height in enumerate(heights)
    }
    return grid


class TestTerrainGuidance(unittest.TestCase):
    def test_actual_dense_native_guide_preserves_every_input_feature(self):
        root = '4210ef67367c610897b7a406eb87c1061c1ca7e9f4f859911de95737bd3fc844'
        folder, _ = next((p, r) for p, r in jobs_for('dev') if r['root_id'] == root)
        legal = read(Path(folder) / 'legal_root.json.gz')
        graph = legal['partial_graph']
        grid = PartialGraph.restore(graph)
        edge = graph['global_edges'][2]
        a, b = edge['path']
        self.assertIsNone(_segment_support(grid, a, b))
        self.assertEqual(_guidance_support(grid, a, b),
                         frozenset(grid.segment_cells(a, b)))
        context = OnlineContext.from_raw(legal['history'], graph,
                                         legal['history_actions'])
        self.assertEqual(len(context.partial_graph['global_edges']),
                         len(graph['global_edges']))
        self.assertEqual(context.partial_graph['global_edges'][2], freeze(edge))
        original = extract(LegalContext.from_inputs(
            legal['history'], legal['history'][-1], graph,
            legal['history_actions']))
        retained = extract(context)
        for key in ('x', 'graph', 'global_graph', 'global_features', 'origin',
                    'velocity', 'age', 'robotxy', 'last_positions', 'last_times'):
            np.testing.assert_array_equal(original[key], retained[key], err_msg=key)
        self.assertEqual(original['keys'], retained['keys'])

    def test_synthetic_smooth_hump_is_a_supported_terrain_connection(self):
        grid = synthetic_grid([0., .1, .2, .3, .4, .3, .2, .1, 0.])
        a, b = grid.center((0, 0)), grid.center((8, 0))
        self.assertTrue(grid.segment_free(a, b))
        self.assertIsNone(_segment_support(grid, a, b))
        self.assertEqual(_guidance_support(grid, a, b), frozenset(grid.cells))

    def test_unknown_cell_cannot_be_bridged(self):
        grid = synthetic_grid([0., .1, .2, .3, .4, .3, .2, .1, 0.])
        del grid.cells[(4, 0)]
        self.assertIsNone(_guidance_support(grid, grid.center((0, 0)),
                                            grid.center((8, 0))))

    def test_ten_meter_interior_height_discontinuity_stays_rejected(self):
        grid = synthetic_grid([0., 0., 10., 0., 0.])
        self.assertIsNone(_guidance_support(grid, grid.center((0, 0)),
                                            grid.center((4, 0))))

    def test_wrong_height_cell_cannot_be_ignored_for_an_alternate_corridor_path(self):
        grid = PartialGraph(1., 0., .25)
        grid.timestamp, grid.revision = 0., 1
        grid.cells = {
            (x, z): dict(state=1, height=10. if (x, z) == (1, 1) else 0.,
                         first_seen=0., last_seen=0.,
                         source='visible_native_cspace_square_and_clearance_halo')
            for x in range(3) for z in range(2)
        }
        # The guide lies on a grid boundary: both rows belong to its exact
        # supercover. A low alternative row must not hide the isolated high cell.
        a, b = np.array([.5, 0., 1.]), np.array([2.5, 0., 1.])
        self.assertTrue(grid.segment_free(a, b))
        self.assertIsNone(_guidance_support(grid, a, b))

    def test_wrong_height_waypoint_stays_rejected(self):
        grid = synthetic_grid([0., 0., 0.])
        a, b = grid.center((0, 0)), grid.center((2, 0))
        a[1] = 10.
        self.assertIsNone(_guidance_support(grid, a, b))


if __name__ == '__main__':
    unittest.main()
