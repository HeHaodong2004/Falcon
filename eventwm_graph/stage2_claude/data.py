"""Streaming TRAIN-split groups (one starting graph + its windows).

Each item carries only legal inputs (graph, ancestor range frames) plus the
window targets used as labels. Target arrays never enter feature code.
"""
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from eventwm_graph.stage2_repair.diagnose import OUT, HF, read, source_rows
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment


def stable(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def split_roots(roots=16384, sensor='expanded_train_range_branches', val_mod=25, start=0):
    rows, _ = source_rows('train')
    rows = rows[start:start+roots] if roots else rows[start:]
    dev, _ = source_rows('dev')
    assert not ({r['scene_id'] for r in rows} & {r['scene_id'] for r in dev})
    rows = [r for r in rows if (OUT/sensor/(r['root_id']+'.npz')).exists()]
    train = [r for r in rows if stable(r['scene_id']) % val_mod]
    val = [r for r in rows if not stable(r['scene_id']) % val_mod]
    return train, val


def scene_factors(rows):
    counts = Counter(r['scene_id'] for r in rows)
    return {s: len(rows)/(len(counts)*n) for s, n in counts.items()}


def load_root(row, sensor='expanded_train_range_branches', tracks=False, aux=False):
    packed = read(HF/'packed/full'/(row['root_id']+'.json'))
    if packed['source_members'] != row['files']:
        raise ValueError('packed source changed')
    a = load_arrays(packed)
    with np.load(OUT/sensor/(row['root_id']+'.npz')) as z:
        s = {k: z[k] for k in z.files}
    res = float(a['geometry'][0])
    track_rows = None
    if tracks:
        from eventwm_graph.stage2_repair.diagnose import member
        from .features import human_arrays, current_humans
        legal = member(row, 'legal_root.json.gz'); nodes = member(row, 'future_observations.json.gz')
        track_rows = [o['tracks'] for o in legal['history']]+[n['observation']['tracks'] for n in nodes]
    groups = defaultdict(list)
    for i, e in enumerate(packed['segments']):
        groups[e['start_node']].append(i)
    out = []
    for sid, ids in groups.items():
        take = ancestor_frames(s['parents'], s['history_count'], sid)
        first = get_segment(a, ids[0])
        if abs(float(s['timestamps'][take[-1]])-float(first[3][0, 0])) > 1e-7:
            raise ValueError('sensor/graph clock mismatch')
        windows = []
        for i in ids:
            v = get_segment(a, i)
            windows.append(dict(path=v[3].astype(np.float64), tkeys=v[4].astype(np.int64), tstates=v[5].astype(np.uint8)))
        out.append(dict(root_id=row['root_id'], scene_id=row['scene_id'], start_node=sid, res=res,
                        gkeys=first[0].astype(np.int64), gstates=first[1].astype(np.uint8),
                        ranges=s['depth'][take].astype(np.float32), origins=s['origins'][take].astype(np.float64),
                        rotations=s['rotations'][take].astype(np.float64), windows=windows,
                        windows_in_root=len(packed['segments']),
                        humans=human_arrays([track_rows[j] for j in take]) if track_rows is not None else None,
                        current_humans=current_humans([track_rows[j] for j in take]) if track_rows is not None else None))
    if aux:
        # LABEL SIDE ONLY: every recorded frame of the Root (incl. future
        # branches) describes the static geometry the robot came to observe.
        for g in out:
            g['aux_frames'] = (s['depth'].astype(np.float32), s['origins'].astype(np.float64), s['rotations'].astype(np.float64))
    return out


class GroupStream(IterableDataset):
    def __init__(self, rows, epoch_seed, sensor='expanded_train_range_branches', shuffle=True, tracks=False, aux=False):
        self.tracks = tracks
        self.aux = aux
        self.rows = rows
        self.seed = epoch_seed
        self.sensor = sensor
        self.shuffle = shuffle
        self.factors = scene_factors(rows)

    def __iter__(self):
        torch.set_num_threads(1)
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        order = np.random.default_rng(self.seed).permutation(len(self.rows)) if self.shuffle else np.arange(len(self.rows))
        for index in order[worker::workers]:
            row = self.rows[int(index)]
            try:
                groups = load_root(row, self.sensor if 'sensor_dir' not in row else row['sensor_dir'], self.tracks, self.aux)
            except (OSError, ValueError, KeyError) as error:
                print('SKIP', row['root_id'], repr(error), flush=True)
                continue
            for g in groups:
                g['scene_factor'] = self.factors[row['scene_id']]
            if self.aux:
                # One item per Root so label-side frames are transferred once.
                frames = groups[0].pop('aux_frames') if groups else None
                for g in groups[1:]:
                    g.pop('aux_frames', None)
                yield dict(root=True, groups=groups, aux_frames=frames)
            else:
                for g in groups:
                    yield g


def collate(items):
    return items
