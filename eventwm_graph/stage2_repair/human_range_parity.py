"""Offline check of sensing replay against private recorded human positions.

Truth arrays are used only to compare the environment replay, never passed to
any Stage 2 predictor or included in its observation packets.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import time
import habitat_sim
import numpy as np
from .diagnose import OUT,HF,read,save,source_rows,member,sha
from eventwm_graph.stage1.runtime import NativeEpisode,SharedFuture


def run():
    from habitat.tasks.nav.nav import NavigationEpisode,NavigationGoal
    rows,_=source_rows('dev');pilot=set(read(HF/'protocol.json')['pilot_selection_roots'])
    selected=[];counts=[0,0]
    for row in rows:
        if row['root_id']not in pilot:continue
        private=member(row,'privileged_initialization.json.gz');human=int(private['episode']['info']['human_num']>0)
        if counts[human]>=(8 if human else 2):continue
        counts[human]+=1;selected.append((row,private))
    reports=[];tick=time.perf_counter()
    for row,private in selected:
        legal=member(row,'legal_root.json.gz');ep=private['episode'];config=read(private['source_map_manifest'])['contract'];seed=read(Path(row['folder'])/'manifest.json')['seed']
        episode=NavigationEpisode(episode_id=ep['episode_id'],scene_id=ep['scene_id'],start_position=ep['start_position'],start_rotation=ep['start_rotation'],goals=[NavigationGoal(**g)for g in ep['goals']],info=copy.deepcopy(ep['info']))
        truth_path=Path(row['folder'])/'privileged_future.npz'
        assert sha(truth_path)==row['files']['privileged_future.npz']
        with np.load(truth_path)as z:expected=z['positions'];when=z['timestamps']
        with NativeEpisode(SimpleNamespace(contract=config),episode,seed,visual=False)as rt:
            for action in legal['history_actions']:rt.step(action)
            tape=SharedFuture(rt).generate(len(when)-1)
            actual=np.array([[h['position']for h in f]for f in tape.truth]).reshape(expected.shape)
            position_error=float(np.max(abs(actual-expected)))if expected.size else 0.
            time_error=float(np.max(abs(np.asarray(tape.times)-when)))
        record=dict(root_id=row['root_id'],scene_id=row['scene_id'],actual_human_count=expected.shape[1],frames=len(when),max_position_coordinate_error_m=position_error,max_timestamp_error_s=time_error,passed=position_error<=1.e-6 and time_error<=1.e-7)
        reports.append(record);print(record,flush=True)
    save(OUT/'human_range_parity.json',dict(role='OFFLINE environment replay parity, private comparison targets never WM inputs',records=reports,passed=all(r['passed']for r in reports),seconds=time.perf_counter()-tick))


if __name__=='__main__':run()
