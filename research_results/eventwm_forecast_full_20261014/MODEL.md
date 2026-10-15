# Forecasting model contract

This stage forecasts physical human motion. It contains no navigation policy,
selector training, action-ranking objective or navigation evaluation.

## Inputs and legal information

The existing adapters accept human histories `[B,N,T,13]`, ego histories
`[B,T,12]`, and an optional known static map `[B,2,32,32]` with support. Exact
history and per-track acquisition timestamps are supplied separately as float64
tensors. Positive observation validity, velocity validity and observation age
remain explicit. Timestamped observed positions determine velocities; native
rigid-body velocity fields are not used as physical supervision.

Each forecast uses a fixed planar frame centered on the legal robot pose at
encoding. New observations must be transformed into that frame. The internal
clock is relative to that encoding time. Numeric tracker IDs are bookkeeping,
not learned features. Future simulator identities, target positions, candidate
outcomes, global human count and future robot queries never enter encoding.

Zero observed humans are represented by an invalid padded slot. Consumers must
apply `forecast.valid`; numbers in padded slots are not unseen-human forecasts.
The anonymous occupancy prior is a separate marginal readout.

## Physical mean models

All learned variants use a shared GRU history encoder, width 128, two masked
interaction layers, and autoregressive 0.25 s physical transitions. Six scene
modes persist across a rollout in the stochastic model; the deterministic
baseline uses one. A motion prior is corrected by learned bounded acceleration.
Mean speed is limited to 3 m/s. This is an implementation bound, not an asserted
native human speed distribution or the robot's V4 speed support.

| Variant | Stored parameters | Physically trained parameters |
|---|---:|---:|
| Original deterministic recurrent | 519,987 | 429,775 |
| Original six-mode recurrent | 520,672 | 430,460 |
| Initial-belief correction | 530,084 | 439,872 |
| Directional local map | 525,424 | 435,212 |
| Map plus initial-belief correction | 534,836 | 444,624 |

The initial-belief head revises stale position and velocity using legal hidden
features, age and velocity validity. It runs at encoding only. Exact fresh
positions are not revised. The directional map encoder samples a 5-by-5 patch
over 4 m around each imagined agent and includes explicit support. The map is
an eroded robot-floor navmesh, not an exact human-body contact map. Old event,
anonymous and learned correction heads remain stored for compatibility but are
frozen and unused by the physical forecasting objective.

The combined model did not improve the controlled 1k-root comparison and was
not expanded to full training. The release uses the initial-belief model with
the per-agent variance law below: 530,155 stored parameters, including the OU
decay parameter, seven variance logits and an inactive 63-parameter context
head. See [ABLATIONS.md](ABLATIONS.md) for measured component comparisons.

## Probability model

Let `z` be a single scene motion mode and `mu_z` its complete recurrent mean
trajectory for all legally tracked agents. The original correlated model is

```
p(Y | history, map) = sum_z pi_z Normal(Y; mu_z, K_z)
u[t+1] = F u[t] + [dt²/2, dt] a[t]
a[t] ~ Normal(0, q²)
Y[t] = mu_z[t] + u_position[t] + Normal(0, r²)
```

The residual state contains position and velocity for each agent/axis. Initial
position/velocity covariance, process noise and readout noise define the entire
trajectory covariance. Exact Kalman innovations compute the mixture likelihood,
marginalizing unsupported coordinates. The Energy Score samples the same law.
NLL is reported in nats per supported spatial coordinate; negative values are
valid for a continuous density.

The OU variant replaces the residual transition with the exact discretization
of `dv = -lambda*v*dt + q_c*dW`, `dp = v*dt`, with
`q_c = q*sqrt(0.25)`. It adds one scalar decay parameter. Mean reversion applies
to the velocity residual, not to the physical mean. It does not clip samples.

The finite scene-scale experiment draws one persistent variance category shared
by every human, coordinate and time. The per-agent experiment instead uses

```
p(Y | h) = sum_z pi_z(h) product_n sum_l rho_nl(h)
                                      Normal(Y_n; mu_zn, v_l K_zn)
```

Each agent's variance category persists through all imagination segments. The
seven fixed standard-deviation multipliers are 0.1, 0.25, 0.5, 1, 2, 4 and 8.
Only their probabilities are fitted in the controlled per-agent experiment;
physical means and Gaussian covariance parameters are held fixed. The optional
63-parameter context head reads legal age, lost flags, velocity validity,
estimated speed, mode velocity spread, nearest legal belief distance and legal
track count. It never reads physical target count. A seven-parameter scene-scale
control receives the same 512 full-batch optimization steps.

A separate bounded refinement experiment holds all physical means and scene
weights fixed while jointly fitting the five OU noise parameters and seven
unconditional variance logits. Each Adam update uses the normalized joint
likelihood over the entire supported fitting-root population; it does not
optimize on development. Completed optimizer updates are resumable. This is a
calibration-convergence diagnostic, not a change to the physical architecture.
The executed 32-step experiment slightly improved fitting likelihood but
worsened development Energy Score and was not promoted.

Scene-mode dependence and explicit attention make these joint mean forecasts.
Conditional residual noise is independent across agents. Sampled residuals do
not feed back into the nonlinear interaction network; this is a mixture of
recurrent means and coherent residual processes, not a fully stochastic
interaction simulator. Mode entropy alone does not demonstrate distinct physical
futures. Mode separation, posterior usage and proper scores are measured.

## Reusable belief and observations

The belief stores physical state, recurrent hidden features, clock, legal track
support, observation age, persistent mode weights and residual covariance. The
scale variants additionally store persistent variance probabilities. Imagination
does not re-encode history or consume future detections.

```python
import torch
from eventforecastwm.benchmark import load_model
from eventforecastwm.composable_sampling import sample_transition

wm = load_model(checkpoint).eval()
belief = wm.encode_history(
    legal_batch, static_map, sensor_config, robot_geometry,
    history_timestamps=history_times,
    track_acquisition_times=acquisition_times,
)
first = wm.imagine(belief, horizon_s=1.0)
second = wm.imagine(first.next_belief, horizon_s=2.0)
generator = torch.Generator().manual_seed(7)  # CPU example
paths1, sample_state = sample_transition(first, k=8, generator=generator)
paths2, _ = sample_transition(second, k=8, generator=generator, state=sample_state)
# Actual arrived positions, expressed in the original fixed frame:
corrected = wm.observe(first.next_belief, actual_observation_at_time_1)
revised = wm.imagine(corrected, horizon_s=2.0)
```

`actual_observation_at_time_1` supplies `positions[B,N,2]`, `observed[B,N]`,
`acquisition_times[B,N]`, and `timestamp_s=1.0`. Acquisition times use the same
relative clock. Existing slots must align; optional `track_ids` must match.
Masked slots may contain NaNs. Actual detected positions/timestamps must be
finite. Future observations and delays exceeding one canonical tick are rejected.
The delayed observation operator uses a local linear approximation.

Observation correction performs a Gaussian residual update and updates mode
likelihoods. Scale variants update variance probabilities with the same
measurement model. The corrected physical state is then used by the nonlinear
transition. This is approximate nonlinear filtering; it is not an exact
posterior over every future nonlinear trajectory. Hidden features persist.

For pathwise composition, use `sample_transition` with its returned sample
state and the same generator, or carry `forecast.sample()`'s conditional noise
state. Do not draw a new scene mode at every segment. After a real observation,
draw new posterior samples. Composition tests compare equal physical times and
identical sampled noise, including covariance and all persistent scale choices.

## Supported readouts and limits

Outputs include component positions/velocities, scene weights, covariance,
coherent samples, exact marginal human-center occupancy, validity and horizon
support. Occupancy integrates the forecast distribution; it is not a blurred
trajectory image. Full-scene occupancy combines tracked occupancy with a common
training-only anonymous marginal prior, fitted with complete native exposure.
Absence of a detection never supplies a free-space label.

Changed-ID association and new track births are not supported by this filtering
interface. Identity/existence probabilities are not calibrated. The recorded
physical populations provide no demonstrated nontrivial birth/death supervision.
Never-observed humans have an occupancy prior, not decoded identity trajectories
or a joint anonymous dynamics model. These limitations prevent a claim of a
complete calibrated belief model for arbitrary changing populations.

This is an observed limitation: the full compact-cache census finds excess
positively observed tracker slots in 593 of 9,221 roots, including 153 initial
origins, and 1,348 switched-ID occurrences across overlapping windows. Forecast
slots must be interpreted as legal track hypotheses, not guaranteed distinct
physical people. The audit uses simulator population counts only as diagnostic
targets; the model never receives those counts. See `association_census/` in
the experiment output directory for root-level provenance.

Training free-runs to 3 s. Five-second results are supported-target extrapolation,
reported separately. Later imagined states remain reusable but may be outside
the trained recurrent depth. Geometry and speed diagnostics qualify sampled
plausibility; no sample-level static feasibility guarantee is made.
