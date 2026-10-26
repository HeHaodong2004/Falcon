"""Full-resolution BEV U-Net for trajectory-conditioned certificate revelation.

Outputs two logits per cell: revelation r and BLOCKED-given-revelation b.
Probabilities are [1-r, r(1-b), rb] for UNKNOWN/FREE/BLOCKED.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1, dilation=1, act=True):
        layers = [nn.Conv2d(cin, cout, k, stride, padding=dilation*(k//2), dilation=dilation, bias=False), nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class Block(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.a = ConvBN(c, c, dilation=dilation)
        self.b = ConvBN(c, c, dilation=dilation, act=False)

    def forward(self, x):
        return F.silu(x+self.b(self.a(x)))


class Context(nn.Module):
    """Global pooled context injected at the coarsest level."""
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c), nn.SiLU(inplace=True), nn.Linear(c, c))

    def forward(self, x):
        g = self.fc(x.mean((2, 3)))
        return x+g[:, :, None, None]


class UNet(nn.Module):
    def __init__(self, cin, widths=(32, 64, 96, 128, 192), blocks=(1, 2, 2, 2, 2)):
        super().__init__()
        self.widths = widths
        self.stem = nn.Sequential(ConvBN(cin, widths[0]), *[Block(widths[0]) for _ in range(blocks[0])])
        self.down = nn.ModuleList()
        for i in range(1, len(widths)):
            dil = 2 if i == len(widths)-1 else 1
            layers = [ConvBN(widths[i-1], widths[i], stride=2)]
            layers += [Block(widths[i], dilation=dil if j % 2 else 1) for j in range(blocks[i])]
            self.down.append(nn.Sequential(*layers))
        self.context = Context(widths[-1])
        self.up = nn.ModuleList()
        for i in range(len(widths)-1, 0, -1):
            self.up.append(nn.ModuleDict(dict(fuse=ConvBN(widths[i]+widths[i-1], widths[i-1], k=1), block=Block(widths[i-1]))))
        self.head = nn.Sequential(ConvBN(widths[0], 32), nn.Conv2d(32, 2, 1))

    def forward(self, x):
        skips = [self.stem(x)]
        for d in self.down:
            skips.append(d(skips[-1]))
        y = self.context(skips[-1])
        for i, u in enumerate(self.up):
            s = skips[-2-i]
            y = F.interpolate(y, size=s.shape[-2:], mode='bilinear', align_corners=False)
            y = u['block'](u['fuse'](torch.cat([y, s], 1)))
        return self.head(y)


class MConvBN(nn.Module):
    """Conv-BN-(SiLU) whose output is zeroed outside the valid frame."""
    def __init__(self, cin, cout, k=3, stride=1, dilation=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding=dilation*(k//2), dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = act

    def forward(self, x, m):
        y = self.bn(self.conv(x))
        if self.act:
            y = F.silu(y)
        return y*m


class MBlock(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.a = MConvBN(c, c, dilation=dilation)
        self.b = MConvBN(c, c, dilation=dilation, act=False)

    def forward(self, x, m):
        return F.silu(x+self.b(self.a(x, m), m))*m


class MaskedUNet(nn.Module):
    """U-Net whose valid-frame outputs are independent of canvas padding.

    Every activation outside the query frame is forced to zero (masks are
    max-pooled per level) and global context is a masked mean. A query's
    prediction therefore does not depend on batch composition or padding.
    The validity mask is channel `valid_channel` of the input.
    """
    def __init__(self, cin, widths=(32, 64, 96, 128, 192), blocks=(1, 2, 2, 2, 2), valid_channel=None, aux=0):
        super().__init__()
        self.widths = widths
        self.aux = aux
        self.valid_channel = valid_channel
        self.stem = MConvBN(cin, widths[0])
        self.stem_blocks = nn.ModuleList([MBlock(widths[0]) for _ in range(blocks[0])])
        self.down = nn.ModuleList()
        for i in range(1, len(widths)):
            dil = 2 if i == len(widths)-1 else 1
            self.down.append(nn.ModuleList([MConvBN(widths[i-1], widths[i], stride=2)] +
                                           [MBlock(widths[i], dilation=dil if j % 2 else 1) for j in range(blocks[i])]))
        self.context = nn.Sequential(nn.Linear(widths[-1], widths[-1]), nn.SiLU(inplace=True), nn.Linear(widths[-1], widths[-1]))
        self.up = nn.ModuleList()
        for i in range(len(widths)-1, 0, -1):
            self.up.append(nn.ModuleList([MConvBN(widths[i]+widths[i-1], widths[i-1], k=1), MBlock(widths[i-1])]))
        self.head_a = MConvBN(widths[0], 32)
        self.head_b = nn.Conv2d(32, 2+aux, 1)

    def forward(self, x, mask=None):
        if mask is None:
            mask = x[:, self.valid_channel:self.valid_channel+1]
        masks = [mask.to(x.dtype)]
        for _ in self.down:
            masks.append(F.max_pool2d(masks[-1], 2))
        y = self.stem(x, masks[0])
        for blk in self.stem_blocks:
            y = blk(y, masks[0])
        skips = [y]
        for level, stage in enumerate(self.down, 1):
            y = stage[0](skips[-1], masks[level])
            for blk in stage[1:]:
                y = blk(y, masks[level])
            skips.append(y)
        m = masks[-1]
        g = (y*m).sum((2, 3))/m.sum((2, 3)).clamp(min=1)
        y = (y+self.context(g)[:, :, None, None])*m
        for i, (fuse, blk) in enumerate(self.up):
            s = skips[-2-i]; ms = masks[-2-i]
            y = F.interpolate(y, size=s.shape[-2:], mode='bilinear', align_corners=False)*ms
            y = blk(fuse(torch.cat([y, s], 1), ms), ms)
        return self.head_b(self.head_a(y, masks[0]))


def build_model(cin, config):
    arch = config.get('arch', 'unet')
    if arch == 'masked_unet':
        return MaskedUNet(cin, tuple(config['widths']), tuple(config.get('blocks', (1, 2, 2, 2, 2))), valid_channel=config['valid_channel'],
                          aux=int(config.get('aux', 0)))
    return UNet(cin, tuple(config['widths']))


def probabilities(logits):
    r = torch.sigmoid(logits[:, 0].float())
    b = torch.sigmoid(logits[:, 1].float())
    return torch.stack([1-r, r*(1-b), r*b], 1)


class GraphRunner:
    """Inference runner: frozen F16 channels-last network replayed through
    CUDA graphs captured once per (batch, square size) bucket.

    Graph capture is model startup (like kernel compilation): it holds no map,
    graph or query state. Inputs are copied into a static buffer whose region
    outside the supplied tensor is zero, exactly like normal zero padding.
    """
    def __init__(self, model, device, batches=(1, 2, 4, 6, 8), sizes=(240, 256, 272, 288, 320, 352), channels=36, capture=True):
        self.device = torch.device(device)
        self.model = model.eval().half().to(self.device).to(memory_format=torch.channels_last)
        self.channels = channels
        self.batches = tuple(sorted(batches))
        self.sizes = tuple(sorted(sizes))
        self.graphs = {}
        self.pool = None
        if capture:
            with torch.cuda.device(self.device):
                for b in self.batches:
                    for s in self.sizes:
                        self._capture(b, s)

    def _capture(self, b, s):
        x = torch.zeros((b, self.channels, s, s), dtype=torch.float16, device=self.device).contiguous(memory_format=torch.channels_last)
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(2):
                self.model(x)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g, pool=self.pool):
            y = self.model(x)
        if self.pool is None:
            self.pool = g.pool()
        self.graphs[(b, s)] = (g, x, y)

    def __call__(self, x, frames=None):
        with torch.cuda.device(self.device):
            return self._call(x, frames)

    def _call(self, x, frames=None):
        """frames: optional (Q,2) per-query valid sizes. With frames, every
        query is computed alone in the batch-1 graph of ITS OWN size bucket,
        so its result is identical whatever else is in the call."""
        q, c, h, w = x.shape
        if frames is not None and 1 in self.batches:
            out = torch.zeros((q, 2, h, w), dtype=torch.float16, device=x.device)
            # (auxiliary training heads, if any, are ignored at inference)
            for i in range(q):
                fh, fw = int(frames[i][0]), int(frames[i][1])
                size = next((s for s in self.sizes if s >= max(fh, fw)), None)
                if size is None or (1, size) not in self.graphs:
                    ph, pw = fh+(-fh) % 16, fw+(-fw) % 16
                    xi = torch.zeros((1, c, ph, pw), dtype=torch.float16, device=x.device)
                    xi[0, :, :fh, :fw] = x[i, :, :fh, :fw]
                    with torch.no_grad():
                        y = self.model(xi.contiguous(memory_format=torch.channels_last))
                    out[i, :, :fh, :fw] = y[0, :2, :fh, :fw]
                    continue
                g, sx, sy = self.graphs[(1, size)]
                sx.zero_()
                sx[0, :, :fh, :fw].copy_(x[i, :, :fh, :fw])
                g.replay()
                out[i, :, :fh, :fw] = sy[0, :2, :fh, :fw]
            return out
        need = max(h, w)
        size = next((s for s in self.sizes if s >= need), None)
        batch = next((b for b in self.batches if b >= q), None)
        if size is None or batch is None or (batch, size) not in self.graphs:
            with torch.no_grad():
                return self.model(x.half().contiguous(memory_format=torch.channels_last))
        g, sx, sy = self.graphs[(batch, size)]
        sx.zero_()
        sx[:q, :, :h, :w].copy_(x)
        g.replay()
        return sy[:q, :, :h, :w]
