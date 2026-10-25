"""Stream immutable packed training Roots and legal ancestor sensor packets."""
from collections import Counter,defaultdict
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import IterableDataset,get_worker_info
from .diagnose import OUT,HF,read,sha,source_rows,member
from .range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment


def inventory(roots=4096):
    rows,source=source_rows('train');rows=rows[:roots]
    dev,_=source_rows('dev')
    assert not ({r['scene_id']for r in rows}&{r['scene_id']for r in dev})
    assert not ({r['root_id']for r in rows}&set(read(HF/'protocol.json')['opened_protected_roots']))
    records=[]
    for row in rows:
        packed=read(HF/'packed/full'/(row['root_id']+'.json'))
        if packed['source_members']!=row['files']:raise ValueError('packed training source changed')
        records.append(dict(source=row,packed=packed,sensor_path=str(OUT/'expanded_train_range_branches'/(row['root_id']+'.npz'))))
    return dict(roots=len(records),scenes=len({r['source']['scene_id']for r in records}),
                windows=sum(len(r['packed']['segments'])for r in records),
                groups=sum(len({e['start_node']for e in r['packed']['segments']})for r in records),
                source_manifest_sha256=sha(source),records=records)


class OnlineRangeGroups(IterableDataset):
    def __init__(self,manifest):
        self.manifest=manifest;self.epoch=mp.Value('i',0)
        counts=Counter(r['source']['scene_id']for r in manifest['records'])
        self.scene_factor={s:manifest['roots']/(len(counts)*n)for s,n in counts.items()}

    def __iter__(self):
        torch.set_num_threads(1);info=get_worker_info();worker=info.id if info else 0;workers=info.num_workers if info else 1
        epoch=self.epoch.value;order=np.random.default_rng(5511+epoch).permutation(self.manifest['roots'])
        for index in order[worker::workers]:
            record=self.manifest['records'][int(index)];row=record['source'];packed=record['packed'];rid=row['root_id'];sp=Path(record['sensor_path']);receipt=read(sp.with_suffix('.json'))
            if receipt['identity']['files']!=row['files']or receipt['sha256']!=sha(sp):raise ValueError('training range source mismatch')
            if sha(packed['path'])!=packed['sha256']:raise ValueError('packed training arrays changed')
            with np.load(sp)as z:sensors={k:z[k]for k in z.files}
            a=load_arrays(packed);legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz')
            tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
            grouped=defaultdict(list)
            for i,e in enumerate(packed['segments']):grouped[e['start_node']].append(i)
            sequence=list(grouped);rng=np.random.default_rng(5511+epoch*1000003+int(index))
            for seq in rng.permutation(len(sequence)):
                sid=sequence[int(seq)];ids=grouped[sid];v=get_segment(a,ids[0]);take=ancestor_frames(sensors['parents'],sensors['history_count'],sid)
                assert abs(sensors['timestamps'][take[-1]]-v[3][0,0])<1.e-7
                windows=[]
                for i in ids:
                    vv=get_segment(a,i);e=packed['segments'][i]
                    windows.append(dict(path=vv[3],target_keys=vv[4],target_states=vv[5],known_terminal_differences=e['known_terminal_differences'],end_node=e['end_node']))
                yield dict(root_id=rid,scene_id=row['scene_id'],keys=v[0],states=v[1],heights=v[2],geometry=v[-1],ego=v[3][0,1:4],now=float(v[3][0,0]),windows=windows,windows_in_root=len(packed['segments']),scene_factor=self.scene_factor[row['scene_id']],sample_seed=5511+epoch*1000003+int(index)*1009+int(seq),packet=dict(ranges=sensors['depth'][take],origins=sensors['origins'][take],rotations=sensors['rotations'][take],timestamps=sensors['timestamps'][take]),tracks=[tracks[j]for j in take])
