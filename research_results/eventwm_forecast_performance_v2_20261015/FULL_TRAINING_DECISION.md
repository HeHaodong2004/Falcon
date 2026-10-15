# Frozen full-data architecture decision

Promote `posterior_self_v2` to the complete 8,965-root / 48,434-window fitting
population. Train seeds **20261014** and **20261015**, each for **24,576 updates**,
batch 64, using the reference warmup, optimizer and cosine schedule. The first
seed is primary by declaration; the second is replication. Neither is chosen
using reserved results. This is an architecture progression decision, not final
model selection or evidence that full training has completed.

Both matched 4,096-root runs completed 16,384 updates and 524,288 root
presentations. After identical training-only OU calibration:

| Development metric | Matched control | Self-only candidate | Change |
|---|---:|---:|---:|
| Origin ES 1s | .356997 | .345895 | -3.11% |
| Origin ES 3s | .710584 | .702689 | -1.11% |
| All-window ES 1s | .641330 | .631846 | -1.48% |
| All-window ES 3s | .828459 | .805391 | -2.78% |
| Origin NLL 1s | -.434333 | -.469413 | -.035080 |
| Origin NLL 3s | -.754260 | -.766433 | -.012173 |
| Origin occupancy AP 3s | .081542 | .081372 | -0.21% |
| Origin 58/65 joint coverage | .680272 | .666667 | -1.36 points |

The origin ES scene-bootstrap intervals include zero: one-second difference
-.011102 [-.025340, .001006], three-second -.007895 [-.027956, .011902].
All-window differences exclude zero: -.009484 [-.019303, -.000181] and -.023067
[-.039534, -.008084]. Windows with no current observation improve 2.20% and
4.30% at one/three seconds. All preregistered older-window ES cohort point
estimates improve; the oldest three-second group still has only 21 roots.

This passes the declared practical progression thresholds without a material
regression. Coverage remains poor for both Gaussian controls; subsequent
probability calibration is necessary. Capacity is unchanged. Cross-agent
attention is replaced by agent-local interaction with the same learned scene
token and parameters; shared mode probabilities still pool legal histories.
The finding does not prove that all social context is unhelpful.

An actual CUDA training interruption at update 8 followed by resume to update 32
was bitwise equal to continuous training for model, optimizer, sampler RNG,
CPU/CUDA RNG, root order, cursor, update/presentation counts and selected state.
This is a bounded same-machine software check, not another selection experiment.

All eight physical pilots and two intermediate runs are closed. The final
probability panel remains fixed: existing agent variance law; all-window
age-conditioned law; fixed geometry anonymous prior; fixed half-weight persistent
stopped law. The separately preregistered one/three-second calibration control
and same-prior CV reference remain queued; the user's subsequent scheduling
revision defers these auxiliary jobs until both physical fits finish. No
further physical architecture search is planned. Original two-seed full-data
models provide the unchanged matched-budget controls.

Machine evidence: `full_family_selection/promotion_evidence.json`, SHA256
`33c2a3623437465fd79afdf8634301350cd6d23770b85ec2894b5a234aa350a9`. Configurations are
`configs/eventwm_performance_v2/full_self_s20261014.json` and
`full_self_s20261015.json`. Final calibration, checkpoint, primary-family and
confirmation identities will be frozen separately after complete development
evaluation. No old or new protected forecast score was used for this decision.
