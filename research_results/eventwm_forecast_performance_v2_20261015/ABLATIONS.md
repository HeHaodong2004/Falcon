# Controlled interventions and negative results

This table records completed experiments only. Every row uses the same nested
1,024 fitting roots, 256 development roots, 8,192 optimizer updates, batch 32,
seed 20261014, physical objective ES + .05 normalized NLL, and training-only
OU calibration. Calibration and evaluation have the same fixed Monte Carlo
budget. The matched original posterior control is an immutable v1 artifact.

| Completed mean control | ES 1s | ES 3s | NLL 3s | ADE 3s | Occupancy AP 3s |
|---|---:|---:|---:|---:|---:|
| Original posterior / .75s velocity window | .36640 | .73209 | -.74965 | 1.25849 | .081407 |
| Ungated fresh-velocity revision | .36953 | .73525 | -.74979 | 1.27246 | .079845 |
| .15s causal velocity window | .36444 | .73994 | -.74791 | 1.25089 | .080006 |
| Capacity-matched self-only interaction | .35002 | .71761 | -.75062 | 1.23206 | .078683 |
| Tracked occupancy BCE weight 2 | .36614 | .73051 | -.75036 | 1.25586 | .081692 |
| Distinct-arrival history encoding | .36163 | .73747 | -.74868 | 1.26022 | .081439 |
| Polar speed/heading dynamics | .38214 | .75170 | -.72589 | 1.27851 | .078914 |
| Tracked occupancy BCE weight 10 | .36692 | .73106 | -.75299 | 1.25371 | .082201 |
| Importance-corrected older-window proposal | .37002 | .73328 | -.74870 | 1.25176 | .080605 |

The fresh-velocity and shorter-window interventions do not earn promotion under
the declared multiobjective rule.
The shorter regression window helps one-second ES and point accuracy slightly,
but worsens three-second ES and occupancy AP. Ungated fresh-velocity revision
does not improve the measured forecasting/occupancy capabilities. These are
bounded negative results, not proof that every possible state-estimation
improvement will fail. No additional physical family is introduced in response;
all eight declared controls have now completed their finite budgets.

Each new completed pilot made 262,144 root presentations. Deterministic sampler
replay verifies the final sampler RNG, shuffled root order, cursor, presentation
count and seen-root set against the saved optimizer checkpoint. The velocity
pilot used all 5,556 windows, at least 19 times per window by completion. Its
selected checkpoint at step 4,608 had already used every window at least nine
times. Final-run and selected-checkpoint exposure are kept separate.

The completed self-only control earns a 4,096-root intermediate run: ES improves
4.47% at one second and 1.98% at three seconds, while AP declines 3.35%, within
the preregistered 5% tolerance. Paired scene-bootstrap ES differences are
-.01638 [-.03784, .00039] and -.01448 [-.05304, .01654]; neither establishes a
generalization improvement yet. NLL changes are small. This is a promotion for
more evidence, not a final architecture decision. It used all 8,192 updates and
262,144 presentations; the selected raw-physical checkpoint is step 7,168.

The weight-2 occupancy auxiliary does not earn promotion: changes in ES, NLL
and AP are smaller than the declared practical thresholds. It completed the
same full pilot budget; the selected step is 4,608.

Distinct-arrival encoding improves origin-one-second ES by 1.30% but worsens
origin-three-second ES by 0.74%. On all actual windows, root-balanced one-second
ES worsens from .653663 to .679169 (3.90%), so it does not earn promotion. In
contrast, self-only gives all-window ES1 .639109 and ES3 .807106 versus matched
control .653663 and .840350: improvements of 2.23% and 3.96%.

Polar dynamics worsen one-second ES by 4.30% and three-second ES by 2.68%; the
bounded speed/heading parameterization did not earn its complexity. Increasing
tracked occupancy BCE from weight 2 to 10 still yields no meaningful forecasting
or occupancy gain (AP improvement under 1% relative to control). The older-window
proposal does not improve origin ES or occupancy. Its completed all-window
comparison gives ES1 .658126 and ES3 .830527 versus .653663 and .840350;
the small three-second improvement comes with a one-second regression. In
windows containing an age >=3s track, the three-second difference is -.086342
[-.209163, .032555], whereas the one-second difference is +.01649 with an
interval spanning zero. This does not justify another intermediate run.
No ninth physical pilot is added. The matched two-family 4k stage is complete;
the candidate passed full-data progression and both independent full-data workers
have completed all 24,576 updates each. Their post-training calibration and full development comparisons are complete;
the final calibrated results do not preserve the intermediate advantage. See
DEVELOPMENT_BENCHMARK.md. Raw checkpoint-selection ES was preliminary.

Source recipes are in `configs/eventwm_performance_v2/pilot_*.json`; exact
calibration commands are in the corresponding `_ou_launch.json` files.
Receipts are `pilot_*/result.json`, `pilot_*_ou/result.json`, and
`sampler_audits/*.json`. Completed runs and the original control are preserved.

Before full-data promotion, the matched 1k posterior, self-only, distinct-arrival,
and older-window sampling controls also receive fixed-budget all-window evaluation.
The cohort comparison joins the existing development census by root/window,
verifies supported human counts, and averages selected windows within each root.
It reports windows containing ages >=1, >=2, or >=3 seconds, no current visible
human, dense scenes, and longer histories. These scores cover every supported
human in the selected window; they are explicitly not per-agent conditional
density ranks. This prevents fresh root-origin performance from concealing an
older-history regression in the encoder or sampler interventions.

The self-only comparison improves these older-history cohorts as well: at one
second, windows containing age >=1s have difference -.027655
[-.044226, -.011451] (147 roots, 108 scenes), and age >=3s has -.032202
[-.062721, -.004549] (108 roots, 85 scenes). At three seconds the corresponding
differences are -.042467 [-.083715, -.003544] (119 roots, 93 scenes) and -.153193
[-.304962, -.027413] (21 roots, 20 scenes). The oldest three-second cohort is
small and cannot alone establish a broad generalization claim. Distinct-arrival
encoding worsens no-current-observation one-second ES by .04377
[.00558, .08117], reinforcing its rejection. Receipt:
`physical_window_cohorts/result.json`.

At 4k roots, identical completed budgets and OU calibration give control/candidate
ES1 .356997/.345895, ES3 .710584/.702689, AP3 .081542/.081372, and NLL3
-.754260/-.766433. All-window ES improves from .641330 to .631846 at one second
and .828459 to .805391 at three seconds. The origin ES intervals include zero;
the all-window ES difference intervals exclude zero. Coverage declines 1.36
percentage points, within the declared progression tolerance, while both Gaussian
controls remain undercalibrated. The complete prospective full-training decision,
paired intervals, limits and receipt hashes are in FULL_TRAINING_DECISION.md.

## Required research questions, assessed before confirmation

| Question | Completed evidence and limits |
|---|---|
| A. Age awareness | Propagating OU state covariance from actual acquisition age plus all-window root-balanced conditional variance fitting improves old-history NLL and high-nominal coverage. Conditioning alone without age propagation provides much less of that gain. Some short-horizon central sets become too broad. |
| B. Missingness-aware training | Distinct-arrival encoding and importance-corrected older-window training completed matched pilots. Neither earns promotion; the former worsens all-window ES1 by 3.90%. Artificial observation dropout was not executed, so no claim is made that every legal missingness augmentation fails. |
| C. Probability law | Persistent stopped-motion regimes improve exact NLL and total occupancy AP, but incur short-horizon ES/central-coverage costs. One/three-second variance fitting improves horizon balance and ES, with worse three-second central coverage. These remain explicit alternatives. |
| D. Motion dynamics | Ungated fresh-velocity revision, shorter velocity regression, and polar speed/heading dynamics do not improve the full metric panel at matched pilot budgets. The new full self-only family fails to replicate its smaller-pool calibrated advantage. |
| E. Occupancy | Geometry-conditioned anonymous marginals improve AP and probability scores. The three-second stopped alternative exceeds original CV AP on development, but not CV given the same geometry prior. The horizon-balanced stopped alternative exceeds same-prior CV as a small point estimate in both seeds, while materially worsening central three-second coverage. The primary candidate does not beat CV. Tracked BCE weights 2 and 10 provide no meaningful pilot gain. |
| F. Scene context | A parameter-matched self-only control retains the scene token and shared mode mixture while removing dynamic cross-agent messages. It helps at 1k/4k but not after full-data calibration; retain the original interaction backbone. Anonymous static-map context helps; no new physical map/Transformer claim is made. |
| G. Sampling | The incumbent sampler was already root-balanced and used older windows. The importance-corrected age proposal gives mixed, small gains and does not advance. Repeated queries are not extra independent futures. |
| H. Full-data scaling | Both new seeds used all 8,965 fitting roots and all 48,434 windows. Self-only OU ES3 improves from .717606 at 1k to .702689 at 4k and .701150/.700232 at full data, with different prescribed budgets. This is not a matched-budget causal scaling estimate. Its advantage over the matched original full model disappears after the richer final probability calibration. |
| I. Recurrent belief | Both age/geometry candidate seeds pass direct/composed means, persistent sample paths and anonymous-grid composition exactly on actual development arrivals. Observation correction improves ES, NLL and point error. Stable legal IDs only; no ReID. |
| J. Replication | Both new physical seeds completed 24,576 updates. Their final calibrated negative comparison replicates. The age/geometry candidate's older-history probability and occupancy benefits replicate on the independent original physical seeds; confirmation remains separate. |

The finite physical search is closed: eight pilots, two matched 4k runs, and
two full-data seeds. Additional source code or a lower raw training-selection
ES is not grounds to extend a saturated experiment. All attempted experiments
remain available, including failures, resource-paused predecessors and controls
that did not advance. Detailed operating state is in the machine-readable
campaign status; calibration steps are not added to physical root presentations.
