"""Audit legal track multiplicity and target association without changing inputs.

Physical population counts appear in this diagnostic only. They never enter
the forecaster or an inference-time track filter. Counts cannot tell us which
specific legal IDs are aliases.
"""
import argparse
from collections import Counter
from pathlib import Path

import torch

from eventbeliefwm.data import read, sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark_reuse import CooperativeGuard


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    guard = CooperativeGuard(out.parent)
    manifest = Path(a.manifest)
    rows = []
    for entry in read(manifest)['roots']:
        guard.check()
        path = manifest.parent / entry['file']
        if sha256(path) != entry['sha256']:
            raise ValueError('Immutable compact payload changed')
        root = torch.load(path, map_location='cpu')
        windows = []
        for i, window in enumerate(root['windows']):
            h = window['base'].humans
            observed = (h[..., 4] > .5) & (h[..., 6] > .5) & (h[..., 10] > .5)
            slots = int(observed.any(1).sum())
            physical = window['metadata']['physical_humans_target_only']
            windows.append(dict(index=i, observed_track_slots=slots,
                recorded_physical_population=physical,
                excess_track_slots=max(0, slots - physical),
                association=dict(window['metadata']['association'])))
        rows.append(dict(root_id=root['root_id'], scene=root['scene'], split=root['split'], windows=windows))
    summaries = {}
    for split in ('all', *sorted({r['split'] for r in rows})):
        selected = [r for r in rows if split == 'all' or r['split'] == split]
        windows = [w for r in selected for w in r['windows']]
        associations = Counter()
        for w in windows:
            associations.update(w['association'])
        summaries[split] = dict(roots=len(selected), windows=len(windows),
            association_occurrences=dict(associations),
            roots_with_excess_slots=sum(any(w['excess_track_slots'] for w in r['windows']) for r in selected),
            origins_with_excess_slots=sum(r['windows'][0]['excess_track_slots'] > 0 for r in selected),
            windows_with_excess_slots=sum(w['excess_track_slots'] > 0 for w in windows),
            excess_slot_occurrences=sum(w['excess_track_slots'] for w in windows),
            max_observed_track_slots=max(w['observed_track_slots'] for w in windows),
            max_recorded_physical_population=max(w['recorded_physical_population'] for w in windows))
    save_json(out, dict(manifest_sha256=sha256(manifest), source_sha256=sha256(__file__),
        summary=summaries, records=rows,
        interpretation='Association and excess counts are correlated window occurrences, not independent humans. Physical population is diagnostic target information only. Excess slots indicate a representation mismatch but do not identify aliases or prescribe an inference-time merge.',
        limitations='No identity repair, removal of tracks, new supervision or changed training/evaluation membership is performed.'))
    print(summaries)


if __name__ == '__main__':
    main()
