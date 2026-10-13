# EventWM v1: LEGACY / DEBUG / REGRESSION ONLY

Dataset: `datasets/raw/eventwm_prod_v1` (symlink resolves under `/home/haodong/workspace/datasets/eventwm-falcon/raw/`). The 4.5 GB raw dataset stays in place and is read-only for v2 work. V2 must never append into this directory, use it by default, or mix its roots into v2.

Integrity audit: PASS for all 48 completed roots. SR audit: 1,504 primary branches; 155 safe; 1,349 collision (180 human, 1,169 static-only). There were 52 separate failed roots. See `EVENTWM_SR_DATA_AUDIT.md` and dataset `audits/sr_summary.json`.

Retirement is a distribution/contract decision, not corruption: v1 proposals frequently execute statically invalid motion. V2 predicts human events conditional on statically feasible fixed queries. Static collision branches must not be used as either positive or negative human-event supervision. V1 remains useful for regression, tensorizer tests and carefully selected micro-overfit debugging; it is not the production training dataset.

Inventory at freeze:
- cwd: `/home/haodong/workspace/projects/eventwm-falcon`
- branch: `research/structured-human-observation`
- HEAD before checkpoint: `ed63aff` (overnight launcher guard)
- recent commits: `0d57610`, `8762be5`, `64dcf29`, `b041762`, `a706993`.
- git status: only untracked `EVENTWM_SR_DATA_AUDIT.md`, `scripts/audit_eventwm_sr.py`, `scripts/training/`.
- collector entry points: `scripts/eventwm_collect.py`, `scripts/eventwm_scale.py`, `scripts/eventwm_overnight.py`, `scripts/run_eventwm_overnight.sh`.
- canonical source/tests/documentation retained: collector, controlled runtime, all collector tests, existing phase-1/structured-observation/scale documents, SR report and `scripts/audit_eventwm_sr.py`.
- useful unfinished event-training preparation archived: `scripts/training/{audit_production,production_index}.py` moved to `scripts/legacy/eventwm_v1/`; repository roots/default output paths corrected. No training was executed. The 174-sample completed-only event index is NOT an SR index or a v2 input.
- generated index/micro16/earlier integrity report moved from `output/eventwm_tiny_overfit` to ignored `output/legacy/eventwm_tiny_overfit`; no raw data copied.
- disposable material: Python bytecode and `/tmp` command logs left ignored; no uncertain artifacts deleted.

The intentional source, tests, audit tooling and documents are checkpointed before v2 edits. Git history preserves the original v1 entry points. V2 disables the historical production launcher and guards the legacy dataset path. The new protocol uses physically separate storage and explicit versions.
