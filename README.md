# Unified Layer 2 world model — 2026-10-27

Stage 1 (`hwm_lite_e12`, human worlds) and Stage 2 (`v17_a`, trajectory-conditioned hypothetical map increments) behind one
legal decision-state API for Layer 2.5: [docs/layer2/INTERFACE.md](docs/layer2/INTERFACE.md) (how to call it) and
[docs/layer2/RESULTS.md](docs/layer2/RESULTS.md) (validation on all 2,110 development Roots, K=1/4/6/8 latency, throughput).
Stage 2's accuracy gate (> 0.50 label-aware IoU) is still unmet (0.4831), so the system as a whole is not accepted.

# Claude Code Stage 2 handoff — 2026-10-26

Start with [the complete technical handoff](docs/handoff/CLAUDE_STAGE2_HANDOFF_20261026.md). It supersedes the older status below, records the legal sensor-input extension, distinguishes oracle diagnostics, and identifies local data, checkpoints, running jobs, and executable reproduction commands. This is a source-only research handoff; Stage 2 remains below its acceptance targets.

# Earlier Stage 2 development progress

See [the current progress report](docs/l2/stage2_hf_20261025/PROGRESS.md) for the 2026-10-25 implementation, configurations, completed pilots and pending work. This branch contains source and compact reports only; it is not a final trained release.

# Falcon research source review

Clean source snapshot of the final Stage1 research checkpoint implementation and Stage2 Trajectory-Conditioned GraphDelta Predictor. Start with [SOURCE_REVIEW_GUIDE.md](SOURCE_REVIEW_GUIDE.md).

Source only: datasets, weights, generated arrays and original Git history are intentionally excluded. Frozen reported results are included for review, not reproduced by this publication.

Original source commit: `b0e662e37b5517cfff455ac7f6146d4df5c21f7a`. See [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) for individual source identities, related component commits and location redactions. See [LICENSE](LICENSE).
