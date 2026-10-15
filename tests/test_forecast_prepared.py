import tempfile
from pathlib import Path
import unittest
import json
import torch
from dataclasses import fields
from eventforecastwm.model import ForecastWorldModel
from eventforecastwm.train import prepare_roots
from eventforecastwm.prepared_cache import prepare
from test_forecast_data import observation,truth
from eventforecastwm.data import window


class TestPreparedCache(unittest.TestCase):
    def test_roundtrip_and_source_mismatch(self):
        model=ForecastWorldModel()
        w=window([observation(0),observation(.25,.1)],[truth(0),truth(.25,.1),truth(.5,.2)],.25,1.)
        w['sensor_config']={};w['robot_geometry']={}
        root=dict(root_id='legal_example',windows=[w]);prepare_roots(model,[root])
        expected=w.pop('_prepared')
        with tempfile.TemporaryDirectory() as temp:
            manifest=Path(temp)/'manifest.json';manifest.write_text(json.dumps(dict(roots=[dict(root_id='legal_example',sha256='fixed')])) )
            cache=Path(temp)/'cache';prepare(model,[root],manifest,cache,prepare_roots)
            actual=w.pop('_prepared')
            for f in fields(expected):
                if isinstance(getattr(expected,f.name),torch.Tensor):self.assertTrue(torch.equal(getattr(expected,f.name),getattr(actual,f.name)))
            def must_not_recompute(*args):raise AssertionError('Cache was not reused')
            prepare(model,[root],manifest,cache,must_not_recompute)
            manifest.write_text(json.dumps(dict(roots=[dict(root_id='legal_example',sha256='changed')])) )
            with self.assertRaises(ValueError):prepare(model,[root],manifest,cache,must_not_recompute)
