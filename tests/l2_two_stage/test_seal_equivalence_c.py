"""Independent native/static counterexamples for the one-pass seal boundary."""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from eventwm_graph.l2_two_stage import types as current
from eventwm_graph.l2_two_stage.data import jobs_for,read


def old_types():
    source=Path(__file__).resolve().parents[2]/'output/l2_two_stage_20261023/frozen_sources/7e55c5db813b0520f9a13c71f4a4aa9dc17a2b7d460fdf96cc9075a47df576da/eventwm_graph/l2_two_stage/types.py'
    spec=importlib.util.spec_from_file_location('review_c_original_types_7ad3',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def native_frame():
    folder,_=jobs_for('dev')[0]
    return next(ob for ob in read(Path(folder)/'legal_root.json.gz')['history'] if ob['tracks'])


def test_one_pass_matches_original_native_duplicate_and_occupied_precedence():
    old=old_types();frame=native_frame();generator=np.random.default_rng(151)
    for case in range(24):
        ob=copy.deepcopy(frame)
        if case:
            rows=ob['static_updates'];rows=rows[:50]+rows[:10]+rows[-10:]
            for index,row in enumerate(rows):
                row['cell']=[int(generator.integers(-8,9)),int(generator.integers(-8,9))]
                row['state']=1+index%2
            ob['static_updates']=[rows[i] for i in generator.permutation(len(rows))]
        a=old.SealedObservation.from_raw(ob);b=current.SealedObservation.from_raw(ob)
        assert a.free_support==b.free_support and a.occupied_support==b.occupied_support
        for field in ('cells','heights','states','sources'):
            assert getattr(a['static_updates'],field).tobytes()==getattr(b['static_updates'],field).tobytes()
        if a.free_codes is None:assert b.free_codes is None
        else:assert a.free_codes.tobytes()==b.free_codes.tobytes()
        ac={};bc={};assert old.update_cells(ac,a)==current.update_cells(bc,b)
        assert ac==bc
        if len(b['static_updates']):
            with pytest.raises(ValueError):b['static_updates'].cells.setflags(write=True)
        # Mutation after arrival cannot change any published row/clock/cell.
        if ob['static_updates']:
            original=b['static_updates'].cells.tobytes();ob['static_updates'][0]['cell'][0]+=100
            assert original==b['static_updates'].cells.tobytes()


def test_nonfinite_history_and_static_clocks_are_rejected_before_publication():
    frame=native_frame()
    for field in ('first_seen_time','last_seen_time','observation_age'):
        for invalid in (float('nan'),float('inf'),-float('inf')):
            ob=copy.deepcopy(frame);ob['tracks'][0][field]=invalid
            with pytest.raises(ValueError):current.SealedObservation.from_raw(ob)
            with pytest.raises(ValueError):current.validate_frame(ob)
    for invalid in (float('nan'),float('inf'),-float('inf')):
        ob=copy.deepcopy(frame);ob['static_updates'][0]['timestamp']=invalid
        with pytest.raises(ValueError):current.SealedObservation.from_raw(ob)
        with pytest.raises(ValueError):current.validate_frame(ob)
