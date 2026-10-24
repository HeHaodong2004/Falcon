"""Packed full-state validation; no cached assumptions about mutable beliefs."""
import numpy as np

def validate_batched(belief, *, hidden_psd=True):
    """Validate all existing invariants using packed numerical checks.

    This returns the original belief and mutates nothing. Ragged arrays are
    packed only inside this call: there is no stale validity certificate or
    reliance on the caller keeping mutable arrays unchanged. Slot uniqueness
    retains the exact reference set check; covariance eigenvalues are batched.
    As in the reference, eigvalsh uses the lower triangle and threshold -1e-7;
    this is a PSD check, not a new symmetry requirement.
    """
    hs = belief.hypotheses
    weights = np.array([h.weight for h in hs])
    assert np.isfinite(weights).all() and np.all(weights >= 0)
    assert 0 <= belief.unresolved_mass <= 1
    assert abs(weights.sum() + belief.unresolved_mass - 1) < 1e-7

    means, covariances, seen = [], [], []
    hidden_means, hidden_covariances = [], []
    pmf_groups = {}
    for h in hs:
        assert h.mean.shape == (h.n, 4)
        assert h.covariance.shape == (h.n, 4, 4)
        assert len(set(h.slot_ids.tolist())) == h.n
        means.append(h.mean)
        covariances.append(h.covariance)
        seen.append(np.asarray(h.last_seen).reshape(-1))
        # Group equal shapes so each per-row sum has the same reduction shape
        # as the original h.hidden_pmf.sum(), without per-hypothesis dispatch.
        pmf_groups.setdefault((h.hidden_pmf.shape, h.hidden_pmf.dtype.str), []).append(h)
        if h.hidden_mean is not None:
            assert h.hidden_mean.ndim == 2 and h.hidden_mean.shape[1] == 4
            assert h.hidden_covariance.shape == (len(h.hidden_mean), 4, 4)
            if h.hidden_exchangeable:
                assert len(h.hidden_mean) == 1
            hidden_means.append(h.hidden_mean)
            hidden_covariances.append(h.hidden_covariance)

    for _, group in pmf_groups.items():
        pmfs = np.stack([h.hidden_pmf for h in group])
        assert np.all(pmfs >= 0)
        axes = tuple(range(1, pmfs.ndim))
        sums = pmfs.sum(axis=axes) if axes else pmfs
        assert np.all(np.abs(sums - 1) < 1e-8)
        private = np.array([h.hidden_mean is not None and not h.hidden_exchangeable
                            for h in group], dtype=bool)
        if np.any(private):
            rows = pmfs[private].reshape(int(private.sum()), -1)
            assert np.all(np.count_nonzero(rows > 1e-10, axis=1) == 1)
            counts = np.array([len(h.hidden_mean) for h, take in zip(group, private) if take])
            assert np.array_equal(counts, np.argmax(rows, axis=1))

    if hs:
        assert np.isfinite(np.concatenate(means, axis=0)).all()
        tracked = np.concatenate(covariances, axis=0)
        assert np.isfinite(tracked).all()
        assert np.all(np.concatenate(seen) <= belief.timestamp + 1e-7)
    else:
        tracked = np.empty((0, 4, 4))
    if hidden_means:
        assert np.isfinite(np.concatenate(hidden_means, axis=0)).all()
        hidden = np.concatenate(hidden_covariances, axis=0)
        assert np.isfinite(hidden).all()
    else:
        hidden = np.empty((0, 4, 4))
    # Preserve each source dtype at numerical tolerance boundaries. A mixed
    # float32/float64 stack would otherwise change the reference eigensolver's
    # precision; similarly, PMF groups above preserve sum precision.
    covariance_groups = {}
    for value in covariances + (hidden_covariances if hidden_psd else []):
        if len(value):
            covariance_groups.setdefault(value.dtype.str, []).append(value)
    for group in covariance_groups.values():
        assert np.linalg.eigvalsh(np.concatenate(group, axis=0)).min() > -1e-7

    # Preserve the entire StructuralBeliefState carrier contract as well. It
    # normally has only 16 components, so keep exact array equality checks.
    c = getattr(belief, 'carrier', None)
    if c is not None:
        if (abs(belief.timestamp - c.times[c.index]) > 1e-8
                or abs(belief.unresolved_mass - c.unresolved_mass) > 1e-10
                or len(hs) != len(c.weights)):
            raise ValueError('belief/carrier clock, count or mass mismatch')
        for w, h in enumerate(hs):
            if abs(h.weight - c.weights[w]) > 1e-10 or not np.array_equal(h.slot_ids, c.worlds[w].slot_ids):
                raise ValueError('belief/carrier population or weight mismatch')
            if (not np.array_equal(h.mean[:, :2], c.positions[w, c.index, :h.n])
                    or not np.array_equal(h.mean[:, 2:], c.velocities[w, c.index, :h.n])):
                raise ValueError('belief/carrier physical endpoint mismatch')
    return belief

