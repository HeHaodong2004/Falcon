# Bounded horizon-calibration follow-up

The first one/three-second conditional-variance fit completed 512 updates on
6,180 supported roots, 26,338 unique windows and 41,258 window/horizon pairs
from the unchanged 8,965-root fitting reservation. Physical means and OU noise
parameters stayed frozen. Equal weighting is applied to roots, available
horizons within root, then supported windows within root/horizon.

| Development metric | Three-second age fit | One/three-second age fit |
|---|---:|---:|
| Origin ES1 | .348900 | .343168 |
| Origin ES3 | .702145 | .699516 |
| All-window ES1 | .624366 | .616753 |
| All-window ES3 | .818560 | .812705 |
| Origin NLL3 | -.933626 | -.932118 |
| All-window NLL1 | -.572340 | -.579911 |
| All-window NLL3 | -.979466 | -.978979 |
| Origin total occupancy AP3 | .097052 | .099749 |
| Primary origin joint 58/65 coverage | .823129 | .802721 |

Paired scene-bootstrap ES differences against the matched three-second age fit
are -.005733 [-.007918, -.003691] at origin 1s and -.002629
[-.006029, .000768] at origin 3s. All-window differences are -.007613
[-.009600, -.005779] and -.005856 [-.008759, -.002852]. Against the released
v1 checkpoint, the ES difference intervals include zero. This is evidence for
improved horizon balancing, not an established gain over the release.

There is a material calibration cost. In the separate fixed conditional-rank
diagnostic, origin 3s central coverage falls from .428571 to .353741, against
nominal 32/65=.492308. All-window 3s central coverage falls from .500000 to
.404618. Conversely, one-second all-window central coverage improves from
.576685 to .504827 and 58/65 coverage from .909836 to .882423. One-second
coverage for age >=3s is .885031, compared with v1 .308642. Three-second age
>=3s coverage is .833333 on only 21 roots. Primary and conditional-rank RNG
protocols differ; do not subtract their coverage estimates across protocols.

The law is therefore not an unqualified replacement and is not promoted to
primary on ES alone. Its common persistent variance categories couple state
and future uncertainty across horizons. The measured central-coverage shift
is a reason to investigate that limitation, not to hide the affected levels.

Execute the previously bounded complementary-law follow-up on the currently
development-leading released physical family. Reuse seed 14's completed fit,
fit the identical horizon objective on seed 15, and refit the same 18-parameter
stopped-motion gate against each new variance law. Keep stopped weight .5,
512 updates, L2 .001, the fixed geometry prior and all data identities. The
stopped branch represents behavioral alternatives rather than another global
variance inflation. Whether it resolves the coverage/ES tradeoff is an empirical
question. This is a bounded diagnostic of the already declared combination;
it does not waive final multiobjective selection or claim that the isolated
horizon law passes every calibration guardrail. No weight grid follows it.

New fits use available GPUs (seed 14 gate on GPU 0; seed 15 variance/gate on
GPU 1), with one owner per GPU and float32 scoring as before. Existing runs,
saved optimizers and scientific configurations are not changed. Final scored
comparisons use the same CPU inference and fixed Monte Carlo protocol. The
new full physical seeds' original calibration panel still completes in parallel.

Evidence: `horizon_control_comparison.json`, the horizon control's completed
evaluation/conditional-coverage receipts, and `horizon_followup/decision.json`.
The prospective scope was committed in `ba69376` before control evaluation
was inspected. The v16 queue adds 19 follow-up and nine final-verification
jobs to the existing 125. It is now active: seed 14's new gate and seed 15's
new horizon-variance fit have launched on the two GPUs. These launches are not
completed fits or release claims.
All confirmation targets remain closed. Final selection must retain the earlier
candidates if this combination does not justify its full probabilistic tradeoff.

## Completed follow-up

Both 512-step horizon fits and both 512-step stopped gates are now complete.
The seed-15 calibration recovered once from its preserved CUDA fault without
changing scientific configuration. The bounded follow-up is closed; no new
weight grid or physical architecture is added. Seed 14/15 combined ES3 is
.698830/.703276, AP3 .115125/.115078, and primary 58/65 coverage .795918/.782313.
The separate conditional diagnostic confirms the central-three-second deficit
in both seeds. This failed to resolve the calibration tradeoff, even though
occupancy point estimates now exceed same-prior CV. Final selection therefore
retains the earlier three-second age-law candidate for verification, with the
horizon/regime checkpoint retained only as an explicitly limited alternative.
