"""Micro-benchmark: K per-query batch-1 CUDA graphs, sequential vs overlapped lanes (and equality)."""
import sys
import time
import torch
from .model import GraphRunner, build_model

ckpt, dev = sys.argv[1], sys.argv[2]
saved = torch.load(ckpt, map_location='cpu'); c = saved['config']
cin = len(c['map_channels'])+len(c['query_channels'])
torch.manual_seed(0)
frames = [(300, 290), (288, 310), (320, 300), (270, 280), (305, 305), (296, 260)]
x = torch.rand((6, cin, 320, 320), device=dev).half()
x[:, c['valid_channel']] = 0
for i, (h, w) in enumerate(frames):
    x[i, c['valid_channel'], :h, :w] = 1
    x[i, :, h:, :] = 0; x[i, :, :, w:] = 0
res = {}
for lanes in (1, 2, 3, 6):
    m = build_model(cin, c); m.load_state_dict(saved['model'])
    r = GraphRunner(m.to(dev), dev, batches=(1,), sizes=(240, 256, 272, 288, 304, 320, 352, 384), channels=cin, lanes=lanes)
    for _ in range(5):
        y = r(x, frames=frames)
    torch.cuda.synchronize()
    ts = []
    for _ in range(40):
        t0 = time.perf_counter(); y = r(x, frames=frames); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    ts.sort()
    res[lanes] = y.float()
    print('lanes %d: median %.2f ms p90 %.2f ms; max |diff| vs 1 lane %.3g' % (lanes, 1e3*ts[len(ts)//2], 1e3*ts[int(.9*len(ts))],
          (y.float()-res[1]).abs().max().item()), flush=True)
    del r, m
    torch.cuda.empty_cache()
