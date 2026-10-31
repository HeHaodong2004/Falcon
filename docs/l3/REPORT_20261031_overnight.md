# Overnight report, 30 Oct 21:50 to 31 Oct 04:00

## Summary

* Dev results were optimistic.  For two days, design choices were made on the same 72 dev episodes, and the
  reported gains on them overstate what generalizes.  Model selection now uses 143 held-out TRAIN episodes:
  never probed, never trained on, 143 distinct scenes (list in the session scratchpad, `selection_train144.txt`;
  rule `index % 23 == 3`, minus every probed episode).
* On those held-out episodes the **shared learned ranker** is selected.  It is one model for all five robot
  settings, trained on the exact outcomes of about 4,350 TRAIN decisions (option
  `L3_MW_RISK=output/agents/m6_ranker/model_multi`).  It gains +9 to +12 points over the floor on ORCA, DWA, SFM
  and slow-turn (p = 0.005-0.053), and +4 on quick-inertia.
* **The world model is essential.**  The same ranker without the Sandbox's imagined-rollout features gains only
  -2 to +3 points.  On held-out DWA it loses 9 episodes to 26 against the full ranker (p = 0.006); on dev
  it scores 45.8% against 69.4% (p = 0.001).
* The hand-written Sandbox rule (the "best legal" configuration tuned on dev) gains +14 on dev but only +5
  (ORCA) and +0 (DWA) on held-out episodes.  The ranker learned from TRAIN exact labels is what generalizes.
* The user's gate (70% in the best setting, 60% in the others) is **not met under honest evaluation**.

## Held-out TRAIN episodes (143 per setting; legal inputs)

| setting | floor | Sandbox + hand rule | ranker v1 (785 states) | ranker v2 (2,174 ORCA states) | **shared ranker (4,350 states, all settings)** | shared ranker without Sandbox features |
|---|---|---|---|---|---|---|
| ORCA-native | 40.6% | 45.5% (+4.9) | 44.1% (+3.5) | 51.0% (+10.4) | **49.7% (+9.1; 26 / 13, p = 0.053)** | 43.4% (+2.8) |
| DWA-native | 42.0% | 42.0% (+0.0) | 41.3% (-0.7) | 51.7% (+9.7) | **51.7% (+9.7; 29 / 15, p = 0.049)** | 39.9% (-2.1) |
| SFM-native | 41.3% | - | 46.4% (+4.3) | 51.7% (+10.4) | **50.3% (+9.0; 25 / 12, p = 0.047)** | 43.4% (+2.1) |
| ORCA-slowturn | 31.5% | - | - | - | **43.4% (+11.9; 25 / 8, p = 0.005)** | - |
| ORCA-quick_inertia (fixed executor) | 31.5% | - | - | - | 35.7% (+4.2; 24 / 18) | - |

Shared ranker with vs without Sandbox features (held-out): ORCA 22 / 13, DWA 26 / 9 (p = 0.006), SFM 22 / 12.
v2 and the shared ranker are equivalent on the three native planners.  The shared ranker is the single policy
because it also covers slow-turn and quick-inertia.  More training data is what helps on held-out episodes:
785 states give +0 to +4, 2,000-4,000 states give +9 to +12.

## Dev (72 paired episodes per setting; noisy and selection-biased, kept for reference)

| setting | floor | Sandbox + rule | v1 | v2 | shared | v1 without Sandbox features |
|---|---|---|---|---|---|---|
| ORCA-native | 47.2% | 62.5% | 63.9% | 65.3% | 61.1% | 51.4% |
| DWA-native | 48.6% | 54.2% | 69.4% | 62.5% | 58.3% | 45.8% |
| SFM-native | 47.2% | 56.9% | 62.5% | 69.4% | 63.9% | - |
| ORCA-quick_inertia | 44.4% | 36.1% | 50.0% | - | 56.9% | - |
| ORCA-slowturn | 41.7% | 38.9% | 47.2% | - | 38.9% | - |

The dev peaks (69.4%) belong to different models in different settings and do not hold on held-out episodes.
Slow-turn shows the opposite case: the shared ranker is below the floor on dev but +11.9 on held-out.

## What ran overnight

* Exact-outcome labels for every candidate at every second decision of new TRAIN episodes, with CPU-only state
  capture (`statebank --blind`):
  * ORCA: 300 episodes, 1,457 states;
  * SFM + DWA: 320, 814;
  * slow-turn: 130, 686;
  * quick-inertia: 130, 674 (with `L3_ANTIDITHER=12 L3_MOM_SIMPLE=1`).
* Rankers:
  * v2: ORCA only, 2,174 states;
  * shared: all settings, about 4,350 states, threshold 0.374 tuned on TRAIN out-of-fold;
  * the ablation without Sandbox features: `riskfit --keep`, commit fe466e23.
* Selection protocol on held-out TRAIN episodes: floors, the Sandbox rule, v1, v2, shared, and the ablation.

Pitfall fixed: stopping a background task kills every process it started, even under `nohup`.  Queues now
launch with `setsid` and record the real pid.

## Recommendations

1. Select on held-out TRAIN episodes from now on, use dev only to confirm, and run the official test split only
   at a freeze (the user's decision).
2. Scale the ranker's data.  The trend is strong (785 states: +0 to +4; 2,000-4,000: +9 to +12).  A run of about
   10k states across settings is the cheapest next lever: 6-8 h of probes overnight, then a refit.
3. Quick-inertia gains least (+4).  Momentum-specific features or more data are the next step there.
4. Agree on where the 70% / 60% targets are measured.  On held-out episodes the honest SR is about 50% on the
   native planners against floors of about 41%.  The official test split is the eventual arbiter.
