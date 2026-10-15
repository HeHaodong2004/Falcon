# Forecasting ablations and negative results

These are train-internal development experiments. Except for the two full-data
initial-belief seeds, component ablations are single-seed exploratory results.
Intervals below resample scenes, with roots equally weighted. Five-second
results use supported labels but extrapolate beyond three-second training.
No planner, selector or action-ranking criterion is used.

## Learned recurrence

Holding the 1k model's encoded belief, mode weights and covariance law fixed,
replacing learned transitions with CV increases 3 s ES from 0.7786 to 0.9043
and mean ADE to 1.4451 m. The paired ES difference is +0.1257, 95% interval
[0.0757, 0.1823]. This intervention shows that predicted physical transitions
contribute to forecasting. It is not a separately fitted kinematic baseline.
Artifact: `ablation1024_recurrence/result.json`.

## Joint modes

The original full one-mode and six-mode models are not distinguishable on
3 s OU Energy Score: B minus C is -0.0101, interval [-0.0235, 0.0031]. More
modes alone did not establish an advantage.

With the initial-belief head at 1k roots, six versus one mode improves 1 s ES
from 0.3860 to 0.3664: difference -0.0196 [-0.0352, -0.0068]. The 3 s ES
difference, -0.0140 [-0.0382, 0.0107], remains inconclusive. Joint NLL improves
by -0.0972 at 1 s and -0.0310 at 3 s, with intervals excluding zero. This
supports short-horizon distributional benefit in this setting, not universal
superiority of multimodal architectures. Artifact:
`comparison1024_posterior_modes/result.json`.

Full initial-belief seed 14 has prior/posterior effective mode counts 5.65/3.51
and mean mode-center separation 0.511 m RMS at 3 s, versus 0.201 m for original
C. Neither entropy nor separation by itself establishes calibrated joint
behavior. Residual samples do not feed back through the interaction network.

## Initial belief and static geometry

The initial-belief head revises legal stale position/velocity before imagination.
The full matched batch-64 comparison uses identical roots, updates, batch,
schedule, seed and objective. Under the same training-only Gaussian OU fit:

| Full mean model | ES 1 s | ES 3 s | ADE 3 s (m) | NLL/coordinate 3 s |
|---|---:|---:|---:|---:|
| Original C, matched batch 64 | 0.3936 | 0.7099 | 1.2055 | -0.7187 |
| Initial-belief seed 14 | 0.3467 | 0.6876 | 1.1682 | -0.7656 |
| Initial-belief seed 15 | 0.3449 | 0.6908 | 1.1835 | -0.7653 |
| Directional map, seed 14 | 0.3853 | 0.6762 | 1.1398 | -0.7180 |

Initial-belief ES differences at 1 s exclude zero for both seeds. Their 3 s
differences versus C64 have intervals [-0.0562, 0.0034] and [-0.0458, 0.0034],
so the additional 3 s ES gain is not established. Map minus C64 at 3 s is
-0.0336 [-0.0632, -0.0085]. The map experiment is not replicated.
Artifact: `comparison_full_matched_b64/result.json`.

Combining map and initial-belief heads at 1k roots gives 3 s ES 0.7339,
compared with 0.7321 for initial belief alone and 0.7202 for the map alone.
The combination was not expanded to full training. Capacity was not increased
by default.

Map conditioning does not establish physical feasibility. Its full model has
27.8% mean-path and 47.4% sample-path nonwalkable fractions on the supported
eroded robot-navmesh proxy, versus 14.1% for native targets. The initial-belief
model has 28.7% and 46.4%. The map is not a human contact oracle and off-grid
points are excluded explicitly. These are geometry deficiencies, not collision
classification errors. Artifacts: `physical_full_map_patch_agent_scale/` and
`physical_full_posterior_agent_scale/`.

## Interaction layers

A capacity-matched 1k control allows attention only to the same agent. Its
3 s Gaussian-process ES is 0.7633 versus 0.7786 with cross-agent attention:
difference -0.0154 [-0.0291, -0.0021]. ADE changes 1.2608 to 1.2519 m and NLL
-0.6715 to -0.6808. This is a negative result for the current interaction
mechanism. The shared scene-mode pooling remains, so the control is not an
entirely independent-agent probability model. Full-data initial-belief results
do not prove that their cross-agent layers help. A matched full-data replication
of the simpler interaction mechanism is a concrete next experiment, rather than
a justification for a larger transformer. Artifact:
`comparison_interaction_1024_gp/result.json`.

## Probability law and calibration

Temporally independent readout noise is an inadequate trajectory distribution.
The 1k C model's coherent Gaussian residual and exact OU residual share physical
means: 3 s ES changes 0.7786 to 0.7466, while NLL changes -0.6715 to -0.7033.
Sampled displacement tails shrink, but joint density-region coverage remains
too low for the Gaussian law. CV and filtered baselines receive equivalent
training-only covariance calibration.

Persistent per-agent variance categories improve the full initial-belief
model's 3 s NLL from -0.7656 to -0.9130, while ES changes 0.6876 to 0.7032.
They are retained for improved density calibration, with the ES cost reported.
The full primary model's nominal finite-rank 89.23% density region covers 85.0%
of development trajectories; its nominal scalar 90% intervals cover 94.9%.
These are distinct calibration questions, neither a proof of reliable
uncertainty. Sampled body-center speed exceeds 3 m/s on about 1.4% of evaluated
displacements; such differences include readout noise.

A conditional variance head adds little over unconditional per-agent
probabilities at 1k roots. A scene-wide Student scale can improve likelihood
while badly distorting intermediate density ranks and generating broad physical
tails; it was not selected. The 32-step full-root covariance refinement changes
development ES 0.7678 to 0.7732 and NLL -0.8957 to -0.8967, holding all physical
parameters bitwise fixed. This small likelihood gain does not justify promotion.
Artifact: `benchmark1024_correlated_agent_refined/result.json`.

## Loss components

All three 1k controls use 8,192 updates, the same root membership, architecture,
batch, seed and optimizer schedule. Training-only OU calibration is matched.

| Objective / checkpoint | ES 3 s | NLL/coordinate 3 s | Mean ADE (m) |
|---|---:|---:|---:|
| ES + 0.05 NLL, ES-selected | 0.7466 | -0.7033 | 1.2608 |
| ES only, ES-selected | 0.7431 | -0.7056 | 1.2622 |
| NLL only, ES-selected at 512 | 0.8553 | -0.7058 | 1.4081 |
| NLL only, final at 8,192 | 0.7876 | -0.6936 | 1.3439 |

ES-only versus mixed-loss ES is inconclusive: -0.0035 [-0.0222, 0.0132]. Both
NLL-only reported checkpoints are worse on ES in this recipe. The final NLL
checkpoint is included because variance fitting changes the raw checkpoint
ordering. This does not establish that NLL is intrinsically unsuitable: its
optimizer was matched, not independently retuned, and the minimum raw-NLL
intermediate checkpoint was not retained. Artifacts:
`comparison_loss_controls_ou/`, `correlated_{energy,nll}_1024/`.

## Additional roots

With matched OU distribution families, C at 1k / 4k / full roots scores
3 s ES 0.7466 / 0.7205 / 0.7179. The 4k minus 1k interval is
[-0.0469, -0.0086]. Improvement largely plateaus after 4k for this architecture.
The matched Gaussian-process scores, 0.7786 / 0.7603 / 0.7604, show the same
plateau. Optimization budgets also grow; this is not an isolated data-only
causal effect. No 500k result exists, and collecting more queries from the same
roots would not create additional independent motion futures.

## Belief correctness and later origins

On all 256 development roots, full-model direct 3 s, 1+2 s, and repeated 0.25 s
imagination have zero maximum differences in means and covariance. Persistent
sampled paths also compose exactly with identical mode/noise identities. Robot
query perturbations change human predictions by zero. This tests computational
composition; it does not guarantee accurate dynamics.

Actual arrived stable-ID updates reduce remaining-horizon ADE 1.6366 to 0.9330 m
on 26 roots (17 contain reacquisition). The paired scene interval for the change
is [-1.0484, -0.4017] m. Cases include failures; changed-ID and birth cases are
unsupported. In the secondary later-origin evaluation, root-balanced 3 s ES
is 0.8016 for initial belief versus 0.8348 for original C and 1.0362 for CV
(166 roots, 443 supported windows). Short horizons are substantially harder
on this population because stale initial states dominate. It is not interchangeable
with the root-origin checkpoint-selection population.
