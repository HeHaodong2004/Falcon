"""Pack a frozen Stage 2 checkpoint for release: inference-only checkpoint + receipts + model card.

usage: python -m ... package <name> [dest]   (name of output/claude_stage2/final/<name>.pt)

The packed checkpoint keeps exactly what LegalVoxelPredictor reads (EMA weights `model`, `config`, frozen `decision`)
plus the TRAIN-validation record `val`; the raw (non-EMA) training weights are dropped, so predictions are unchanged.
"""
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

from eventwm_graph.stage2_repair.diagnose import OUT

ROOT = OUT.parent/'claude_stage2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(name, dest=None):
    dest = Path(dest or ('release/stage2_claude_%s' % name))
    dest.mkdir(parents=True, exist_ok=True)
    src = ROOT/'final'/(name+'.pt')
    saved = torch.load(src, map_location='cpu')
    slim = {k: saved[k] for k in ('model', 'config', 'decision', 'val') if k in saved}
    ckpt = dest/('stage2_claude_%s.pt' % name)
    torch.save(slim, ckpt)
    receipts = dest/'receipts'; receipts.mkdir(exist_ok=True)
    copied = {}
    for label, path in (('full_dev_summary.json', ROOT/'evaluation'/('full_%s_256' % name)/'summary.json'),
                        ('full_dev_identity.json', ROOT/'evaluation'/('full_%s_256' % name)/'identity.json'),
                        ('latency_benchmark.json', ROOT/'benchmark'/('%s_bench256.json' % name)),
                        ('latency_quiet_receipt.json', ROOT/'benchmark'/('quiet_%s_256.json' % name)),
                        ('selection_400_validation_roots.json', ROOT/'selection'/(name+'.json')),
                        ('final_pipeline_log.txt', ROOT/'logs'/('final_%s.log' % name)),
                        ('query_bench_report.json', ROOT/'query_bench'/'report.json'),
                        ('query_bench_audit.json', ROOT/'query_bench'/'audit.json')):
        if path.exists():
            if label == 'selection_400_validation_roots.json':
                sel = json.loads(path.read_text()); sel.pop('grid', None)          # keep the chosen point, drop the grid
                (receipts/label).write_text(json.dumps(sel, indent=1))
            elif label == 'query_bench_audit.json':
                (receipts/label).write_text(json.dumps(json.loads(path.read_text())['summary'], indent=1))
            else:
                shutil.copy(path, receipts/label)
            copied[label] = str(path)
    manifest = dict(name=name, checkpoint=ckpt.name, checkpoint_sha256=sha(ckpt), source_checkpoint=str(src),
                    source_checkpoint_sha256=sha(src), parameters=int(sum(v.numel() for v in slim['model'].values())),
                    config={k: slim['config'][k] for k in ('widths', 'blocks', 'side', 'dside', 'attn', 'sub', 'render', 'near', 'human_filter',
                                                            'human_occluders', 'iou_weight', 'train_roots', 'warm', 'teacher') if k in slim['config']},
                    decision={k: v for k, v in slim['decision'].items() if k != 'validation'}, receipts=copied)
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps(manifest, indent=1, default=str))
    return manifest


if __name__ == '__main__':
    main(*sys.argv[1:])
