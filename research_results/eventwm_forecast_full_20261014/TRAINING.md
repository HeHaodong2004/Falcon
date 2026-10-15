# Executed training

All runs below use the immutable 350k snapshot and fixed train-internal
development protocol described in [DATA.md](DATA.md). None uses reserved
confirmation episodes for fitting or checkpoint selection. The full fitting
population is 8,965 roots, not 304,293 independent query futures.

## Completed full-pool runs

| Run directory | Seed | Batch | Updates | Root presentations | Root epochs | Selected update | Loop seconds | Reported peak GPU GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `deterministic_gp_full_cached` | 20261014 | 32 | 32,768 | 1,045,444 | 116.61 | 31,232 | 6,825.7 | 1.638 |
| `correlated_full_cached` | 20261014 | 32 | 32,768 | 1,045,444 | 116.61 | 26,624 | 6,885.9 | 1.638 |
| `stochastic_full_cached` (IID output noise) | 20261014 | 32 | 32,768 | 1,045,444 | 116.61 | 24,576 | 5,617.0 | 1.638 |
| `correlated_full_b64_s20261014` | 20261014 | 64 | 24,576 | 1,562,598 | 174.30 | 22,016 | 5,029.9 | 2.814 |
| `posterior_full_b64_s20261014` | 20261014 | 64 | 24,576 | 1,562,598 | 174.30 | 17,408 | 5,679.6 | 3.409 |
| `posterior_full_b64_s20261015` | 20261015 | 64 | 24,576 | 1,562,598 | 174.30 | 16,640 | 5,527.0 | 3.379 |
| `map_patch_full_b64_s20261014` | 20261014 | 64 | 24,576 | 1,562,598 | 174.30 | 22,272 | 5,950.0 | 3.409 |

Directories are relative to
`/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_full_20261014`.
Loop duration accumulates optimization, validation and checkpointing, excluding
resource pauses and process startup/preparation. GPU peaks are decimal GB and
refer to the final resumed process session; they are not proven maxima over
every process session. Pauses do not reset optimizer or sampler progress.

Every full run completed its declared budget. Exact sampler audits for the
original stochastic run and both initial-belief seeds confirm all fitting roots
and all 48,434 available windows were visited. Other full runs have the same
root/window sampler configuration and recorded full root counts; their window
coverage is not substituted for an unexecuted replay audit.

## Optimization and selection

The main objective is joint trajectory Energy Score plus 0.05 times exact
trajectory NLL normalized per supported coordinate. Training free-runs up to
three seconds from actual legal histories; no future target enters the imagined
state. Missing target coordinates are masked. This is not best-of-K training.

AdamW uses physical learning rate `1e-4`, weight decay `0.01`, and gradient
clipping at norm 1. The four uncertainty parameters use ten times the learning
rate and no weight decay. The schedule warms up and then decays by cosine to
10% of its initial rate. Batch-32 runs use the default 256 warmup steps;
batch-64 runs use 128. Physical and uncertainty gradients remain finite in the
completed logs. Recorded gradient norms are measured before clipping.

Checkpoint selection minimizes supported development 3 s proper Energy Score
under the training distribution. It is not based on navigation, action ranking
or minADE. Training-only covariance calibration subsequently fits the chosen
checkpoint, holding physical means and scene weights fixed. This separation
can change checkpoint ordering; the NLL-only control explicitly reports both
its ES-selected and final checkpoints rather than hiding that effect.

Each directory contains `identity.json`, `architecture.json`, `training.jsonl`,
`validation.jsonl`, per-validation `dev_step*.json`, `best.pt`, `last.pt`, and
`result.json`. Identity pins config, dataset, protocol and implementation hashes.
The result records exact best/final SHA256 values. Calibrated checkpoint hashes
and development result hashes are additionally frozen in
[confirmation_decision.json](confirmation_decision.json).

The two initial-belief seeds have similar late learning curves and outperform
the old small pilot. This is evidence of substantial optimization, not a proof
of global convergence. Mode entropy remains nonzero, but mode separation and
proper held-out scores are required to interpret it. The full-model curves are
`curves_full_posterior_completed/learning_curves.png` and `.pdf`; original full
baseline curves are in `curves_full_original/`.

## Root scaling and ablation budgets

The new 128-root references run 1,024 steps on 711 windows. The 1,024-root
comparisons run 8,192 steps on 5,556 windows, with 262,144 root presentations
(256 root epochs). The 4,096-root correlated model runs 16,384 steps on 22,175
windows, with 524,288 root presentations (128 epochs). These are convergence-
budget comparisons, not an isolated causal effect of more data at fixed compute.

The full batch-64 C / initial-belief / map runs match root membership, batch,
update budget, seed 20261014, schedule and objective. The second initial-belief
seed is a replication. The original B/C batch-32 runs also match each other.
Fixed-step original full-pool checkpoint copies are in `checkpoint_archive/`:
at update 4,096, raw development ES was 0.7704 for B and 0.7578 for C; final
selected raw ES was 0.7439 and 0.7476. A 4,096-update pilot was not treated as
full-data convergence.

Completed 1k controls include deterministic versus stochastic recurrence,
observation-age covariance, scalar and directional maps, initial-belief
correction, their combination, capacity-matched self-only attention, IID versus
temporally correlated residuals, and Energy Score-only / NLL-only losses.
Loss-control budgets, architecture and seed match the mixed-loss control.
See [ABLATIONS.md](ABLATIONS.md) for positive and negative outcomes.

## Calibration and exact recovery

The Gaussian/OU calibration fits only global noise parameters, using every
eligible fitting root with complete root-origin 3 s identity support. Full-data
calibration has 5,245 supported roots and at least eight minibatch passes
(2,248 updates). The persistent per-agent variance stage fits seven global
probability logits using 512 full-batch updates. It does not refit physical
trajectories or use development labels.

A 32-update full-root joint refinement of noise and variance probabilities was
tested on the 1k pool. Training NLL improved from -0.889666 to -0.890520, but
development Energy Score worsened and likelihood barely changed. It was not
promoted. All frozen physical parameters were verified bitwise unchanged.

Training resume stores model, optimizer, root order/cursor, sampling RNG and
CPU/CUDA RNG states. `resume_audit/result.json` demonstrates bitwise equality
between uninterrupted 32 updates and an 8+24 interrupted run on this system.
The refinement also resumes completed Adam updates; a partial interrupted
gradient is discarded and replayed. Production collector state is never written.

The historical C256/C4096 diagnosis supports overfitting: training-root 3 s ES
improved 0.8374 to 0.2436 while development ES worsened 0.8871 to 1.0087.
That result does not isolate learning rate, objective or limited diversity as
the sole cause. The new full-data runs are a separate controlled protocol.
