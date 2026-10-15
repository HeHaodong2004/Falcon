"""Query-free legal motion estimation. Never consumes labels or future truth."""
import torch


def arrived_tracks(value, steps=16, window_s=2., max_gap_s=.5, max_age_s=2.):
    """Return [B,N,8] tracks, validity, [B,N,S,10] history and sequence mask.

    Observation time = snapshot time - observation age in the legacy synchronous
    schema. Repeated stale states are not new measurements. A long reacquisition
    gap invalidates velocity rather than importing a simulator/current velocity.
    Tracks absent at decision time remain absent. Positions use the fixed
    decision-time robot frame; the query tensor is deliberately never accessed.
    """
    h = value.humans.detach().cpu()
    hp, tp = value.human_padding.cpu(), value.history_padding.cpu()
    b, n, t, _ = h.shape
    if h.shape[-1] != 13 or value.robot.shape != (b,t,12) or hp.shape != (b,n) or tp.shape != (b,t):
        raise ValueError('Invalid legal history tensor shapes')
    tracks = torch.zeros(b, n, 8)
    valid = torch.zeros(b, n, dtype=torch.bool)
    sequence = torch.zeros(b, n, steps, 10)
    sequence_mask = torch.zeros(b, n, steps, dtype=torch.bool)
    for i in range(b):
        indices = (~tp[i]).nonzero().flatten().tolist()
        if not indices:
            raise ValueError('Each sample requires arrived history')
        if indices != list(range(len(indices))):
            raise ValueError('History padding must be contiguous on the right')
        clocks = value.robot[i, indices, 7].detach().cpu()
        if not torch.isfinite(clocks).all() or (clocks[1:] <= clocks[:-1]).any() or clocks[-1].abs() > 1e-5:
            raise ValueError('History must have increasing clocks ending at decision time')
        for k in range(n):
            if hp[i, k]:
                continue
            last = h[i, k, indices[-1]]
            if last[6] < .5 or last[10] < .5 or last[7] > max_age_s:
                continue
            previous_time, previous_position = None, None
            velocity = torch.zeros(2)
            velocity_valid = False
            features = []
            for j in indices:
                row = h[i, k, j]
                if row[6] < .5 or row[10] < .5:
                    continue
                used = row[[0, 1, 4, 6, 7, 8, 10]]
                if not torch.isfinite(used).all() or row[7] < 0 or row[8] > 1e-5:
                    raise ValueError('Invalid arrived track clock or position')
                snapshot, age = float(row[8]), float(row[7])
                observed = snapshot-age
                if previous_time is not None and observed < previous_time-1e-5:
                    raise ValueError('Observation acquisition clock moved backwards')
                if previous_time is None or observed-previous_time > 1e-5:
                    gap = 0. if previous_time is None else observed-previous_time
                    velocity_valid = previous_time is not None and gap <= max_gap_s+1e-5
                    velocity = (row[:2]-previous_position)/gap if velocity_valid else torch.zeros(2)
                    previous_time, previous_position = observed, row[:2].clone()
                if snapshot >= -window_s-1e-5:
                    features.append(torch.cat([row[:2]/6, velocity/2, row.new_tensor(
                        [snapshot/2, observed/2, age/5, float(row[4]), float(velocity_valid), 1.])]))
            if not features:
                continue
            tracks[i, k] = torch.cat([last[:2], velocity, last[7:8], last[4:5],
                                      last.new_tensor([float(velocity_valid), 1.])])
            valid[i, k] = True
            if len(features) > steps:
                keep = torch.linspace(0, len(features)-1, steps).round().long().tolist()
                features = [features[j] for j in keep]
            sequence[i, k, :len(features)] = torch.stack(features)
            sequence_mask[i, k, :len(features)] = True
    device = value.humans.device
    return tuple(x.to(device) for x in (tracks, valid, sequence, sequence_mask))


def local_to_world(points, final_robot):
    """Invert tensorize's rotation. points [...,2], robot [...,12]."""
    sine, cosine = final_robot[..., 3], final_robot[..., 4]
    x, z = points[..., 0], points[..., 1]
    return torch.stack([cosine*x+sine*z+final_robot[..., 0],
                        -sine*x+cosine*z+final_robot[..., 2]], -1)
