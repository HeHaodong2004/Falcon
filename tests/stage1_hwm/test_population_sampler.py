"""Exact conditional population sampler == brute-force rejection (statistically), torch and NumPy.

Run: python scripts/stage1_hwm_run.py tests/stage1_hwm/test_population_sampler.py
"""
import numpy as np
import torch

from eventwm_graph.stage1_hwm.sampler import draw_population, draw_population_np, count_log_law


def main():
    rng = np.random.default_rng(0);W = 200000
    for case in range(6):
        K = int(rng.integers(1, 6));p = rng.random(K);p[rng.random(K) < .3] = 1.   # some visible keys
        mask = np.ones(K, bool);mask[-1] = case % 2 == 0
        res = rng.dirichlet(np.ones(11) * .5);m = int(rng.integers(0, K + 3))
        # brute force rejection
        pr = (rng.random((W * 4, K)) < p) & mask;ct = rng.choice(11, W * 4, p=res)
        ok = pr.sum(1) + ct >= m
        ref_e = pr[ok].mean(0);ref_c = np.bincount(ct[ok], minlength=11) / ok.sum()
        a, c = draw_population_np(p, mask, res, m, W, rng)
        g = torch.Generator();g.manual_seed(case)
        at, ctt = draw_population(torch.tensor(p)[None], torch.tensor(mask)[None], torch.tensor(res)[None], torch.tensor([m]), W, g, 10)
        for name, e, cc in (('numpy', a, c), ('torch', at[0].numpy(), ctt[0].numpy())):
            de = np.abs(e.mean(0) - ref_e).max();dc = np.abs(np.bincount(cc, minlength=11)[:11] / W - ref_c).max()
            assert de < .01 and dc < .01, (case, name, de, dc)
            assert ((e.sum(1) + cc) >= m).all() and not (e & ~mask).any()
        # exact law: truncated count law equals empirical total distribution
        law = np.exp(count_log_law(np.where(p >= 1, 1., p)[None], (p >= 1)[None] & mask[None], mask[None], np.log(res)[None], np.array([m]))[0])
        tot = np.bincount(a.sum(1) + c, minlength=len(law))[:len(law)] / W
        assert np.abs(tot - law).max() < .01, (case, np.abs(tot - law).max())
    print('conditional population sampler matches rejection sampling and the exact count law PASS')


if __name__ == '__main__':
    main()
