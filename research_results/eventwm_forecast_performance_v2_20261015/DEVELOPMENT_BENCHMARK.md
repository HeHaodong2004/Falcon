# Full-data development comparison

All scores use the frozen 256 development roots and identical supported targets.
Energy Score uses 48 paths per bank; primary joint density coverage uses 64 draws
with nominal 58/65. Lower ES/NLL/Brier/log score is better; higher AP is better.
Root-origin and all-window populations are separate. Query branches are not
independent observations. Two-second results and ADE/FDE are retained in the
structured receipts; five seconds is not an in-distribution selection score.

| Model | ES1 | ES3 | NLL3 | ADE3 | Total foreground AP3 | All-window ES1 | All-window ES3 | Joint coverage3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Released v1 seed 14 | 0.347622 | 0.703191 | -0.913030 | 1.168215 | 0.093107 | 0.615829 | 0.813458 | 0.850340 |
| Released v1 seed 15 | 0.347225 | 0.710418 | -0.925287 | 1.183461 | 0.093133 | 0.616537 | 0.816018 | 0.843537 |
| CV with same geometry prior | 0.411134 | 0.892640 | -0.945349 | 1.414931 | 0.113897 | 0.862594 | 1.043179 | 0.870748 |
| Age / geometry seed 14 | 0.348900 | 0.702145 | -0.933626 | 1.168215 | 0.104795 | 0.624366 | 0.818560 | 0.823129 |
| Age / geometry seed 15 | 0.349847 | 0.708606 | -0.954442 | 1.183461 | 0.105357 | 0.633184 | 0.825417 | 0.809524 |
| Stopped / age / geometry seed 14 | 0.350176 | 0.703454 | -1.018782 | 1.172979 | 0.112423 | 0.634354 | 0.827969 | 0.823129 |
| Stopped / age / geometry seed 15 | 0.352738 | 0.710687 | -1.018127 | 1.191178 | 0.112296 | 0.642928 | 0.836716 | 0.823129 |
| Horizon age / geometry seed 14 | 0.343168 | 0.699516 | -0.932118 | 1.168215 | 0.107210 | 0.616753 | 0.812705 | 0.802721 |
| Horizon regime / geometry seed 14 | 0.344673 | 0.698830 | -1.016955 | 1.172874 | 0.115125 | 0.627358 | 0.822390 | 0.795918 |
| New full self-only 14 / agent_scale | 0.339231 | 0.725777 | -0.927038 | 1.170136 | 0.090212 | 0.614546 | 0.812271 | 0.863946 |
| New full self-only 14 / age_geometry | 0.341839 | 0.719321 | -0.958211 | 1.170136 | 0.102280 | 0.631443 | 0.822269 | 0.823129 |
| New full self-only 14 / regime_geometry | 0.345363 | 0.716884 | -1.016547 | 1.177416 | 0.109148 | 0.641469 | 0.833938 | 0.829932 |
| New full self-only 15 / agent_scale | 0.345150 | 0.722501 | -0.927293 | 1.181805 | 0.091167 | 0.619080 | 0.808261 | 0.843537 |
| New full self-only 15 / age_geometry | 0.347260 | 0.719491 | -0.962108 | 1.181805 | 0.103313 | 0.641570 | 0.823702 | 0.816327 |
| New full self-only 15 / regime_geometry | 0.352551 | 0.719404 | -1.010916 | 1.187699 | 0.110625 | 0.650048 | 0.836315 | 0.836735 |
| Horizon age / geometry seed 15 | 0.343194 | 0.703932 | -0.953033 | 1.183461 | 0.107581 | 0.626832 | 0.816743 | 0.789116 |
| Horizon regime / geometry seed 15 | 0.345237 | 0.703276 | -1.016346 | 1.191034 | 0.115078 | 0.635001 | 0.829043 | 0.782313 |

The new full self-only physical family does not reproduce its intermediate
advantage after matched final calibration. Seed 14 agent-scale ES3 worsens
by .022586, paired scene-bootstrap 95% interval [.002986, .044133]. Seed 15
worsens by .012082 [−.010506, .035074]. The later age/geometry variants have
small origin-one-second gains or ties but worse three-second ES; seed 15
all-window one-second ES costs exceed 3%. Full budget completion is not
evidence that this architecture is better. No schedule extension is justified
by the saturated raw development curves.

The released physical family with age-aware all-window variance fitting and
the geometry anonymous prior is under final software and probabilistic review.
It retains point forecasts, improves older-history proper likelihood and
high-nominal coverage, and improves occupancy. Central one-second coverage
can be too broad, while origin-three-second coverage remains deficient.
These tradeoffs are explicit; this is not a demonstrated ES breakthrough.
The horizon-balanced variant improves ES but materially undercovers central
three-second sets. The stopped-motion alternative improves NLL and AP, with
larger all-window one-second ES costs. Neither gets promoted solely for its
best individual metric.

The same geometry prior raises CV origin-three-second AP to .113897 from
.106467. Consequently an improvement over the original CV alone cannot be
attributed wholly to learned human dynamics. The candidate age/geometry AP
remains below both CV comparisons.

Evidence: `full_self_s20261014_comparison/result.json`,
`full_self_s20261015_comparison/result.json`, the fixed-horizon comparison
receipts and `development_results.json`. All paired intervals cluster scenes
and retain root weighting; an interval spanning zero is inconclusive.

The completed horizon/regime alternative has AP3 .115125/.115078 across seeds,
slightly above same-prior CV .113897 as point estimates. Its primary joint
coverage is only .795918/.782313 at nominal .892308. In the separate conditional
rank diagnostic, origin-three-second central coverage is .374150/.360544 at
nominal .492308, and 75%-level coverage .612245/.591837 at nominal .738462.
Those substantial primary-population calibration deficits prevent promotion
based on its lower ES and higher AP alone. Preserve the family as an explicit
probability/occupancy tradeoff, not a calibrated primary.
