# EventWM mass-collection version and provenance audit

**Answer: the mass worktree is not missing any newer locally available collector fix.** At audit time both worktree HEADs equal `bf46195517d33367af061c2982bfbf2ed6f58000`. All 41 scoped tracked files have identical SHA-256 bytes in the base commit, original working tree, and mass working tree. No local branch or remote-tracking ref contains a commit beyond that base.

**The latest model and audit work is newer, uncommitted work in the original checkout.** It has not been copied into the collection-only checkout. This is not a missing collector upgrade: the model/adapter imports the same canonical `collector.core.load_wm_input`, whose hash matches the overfit diagnostic record. No collector semantics were changed by this audit.

## Worktrees and branches

Original: `/home/haodong/workspace/projects/eventwm-falcon`; branch `research/eventquerywm-v0`; HEAD `bf46195517d33367af061c2982bfbf2ed6f58000`.
Mass: `/home/haodong/workspace/projects/eventwm-falcon/output/mass_collection_checkout`; branch `research/eventwm-v3-mass-collection`; HEAD `bf46195517d33367af061c2982bfbf2ed6f58000`.

| Local ref | Tip | Commits outside base | Tip ancestor of base |
|---|---|---:|---|
| refs/heads/research/controlled-matrix | d15ef512c4b1 | 0 | True |
| refs/heads/research/eventquerywm-v0 | bf46195517d3 | 0 | True |
| refs/heads/research/eventwm-data-collector | b041762b368c | 0 | True |
| refs/heads/research/eventwm-human-event-v2 | 36e2872d09cc | 0 | True |
| refs/heads/research/eventwm-social-outcome-v3 | 82097ac974cb | 0 | True |
| refs/heads/research/eventwm-v3-mass-collection | bf46195517d3 | 0 | True |
| refs/heads/research/eventwm-v3-natural-sampling | bf46195517d3 | 0 | True |
| refs/heads/research/structured-human-observation | 4a72eaa692dd | 0 | True |
| refs/remotes/origin/HEAD | d15ef512c4b1 | 0 | True |
| refs/remotes/origin/main | d15ef512c4b1 | 0 | True |
| refs/remotes/origin/research/controlled-matrix | d15ef512c4b1 | 0 | True |

`git log --all bf46195.. -- scripts/eventwm_collect.py scripts/collector` is empty. `git diff bf46195` in the original checkout reports only the legacy SR audit script and its document. The original untracked files contain EventQueryWM, its adapter/diagnostics, regression tests, exact-input audit, diagnostic reports and checkpoints. No `scripts/collector/` file is modified or untracked. No index assume-unchanged/skip-worktree flags were found in the mass checkout.

## Collector, semantics and storage checks

| File | SHA-256 (base = original = mass) |
|---|---|
| `scripts/eventwm_collect.py` | `a18b534c6cf790a857285065002a489c1b68e1cf76397dadcd3de63e1ea6c4a0` |
| `scripts/collector/core.py` | `d8e043f2f530e7a87a67cce7d5712758b5f97460ea83c2f5a79582265f9a0dc9` |
| `scripts/collector/v2.py` | `eac8b9e869e855bdc71718792deb4349f529905c30ff26403a4bde5e4630a2a5` |
| `scripts/collector/v3.py` | `be1025999a1fa1a5f294f125c8612c95b827eb1a5dc52eb2bf7b952708bb9ae1` |
| `scripts/collector/runtime.py` | `ad485a52779bbdb5d2149b6aec6f10bf5b12fad91f701aca3072c2dd7840ede4` |
| `scripts/collector/wm_sampling.py` | `fa919cebf4a18b0eb1d393156ae2c1df4a7ec63e4cac529d0a20ccb17842d92b` |
| `scripts/collector/root_sampling.py` | `36edb643d91934f68221c603ef9efa31fad30dc41b0fd5757570fbdc87c8b43f` |
| `scripts/collector/natural_sampling.py` | `aec9c631914ab32fd67bbb4182f5ff5280a0885151d0c5ba82a5457191305af7` |
| `scripts/collector/observation.py` | `7212ddf3610b6f6a9174b9d9a3c35d3ad65c2c96f4c88a19962066999b0f0ce9` |
| `scripts/collector/observation_audit.py` | `c426f51225b77da0bd5bab74d616f5cd8685633213a1a754051902720fcad260` |
| `scripts/collector/static_feasibility.py` | `93be0dc1821c277d9276485084730b386ff938a154b237253a904dc309d12228` |
| `scripts/collector/static_habitat.py` | `e10406110dbef8c38f4e615663aa3b247d09d84d65ae4c313dca5d978c9de4ed` |
| `scripts/collector/storage.py` | `7b920d181bfdb54526b2e9daf9c86334905c0a5f74ffed190e4a0b8a285eb290` |

`scripts/collector/queries.py` does not exist in the base commit or either worktree. Query compilation is in `collector/core.py` (`compile_motion`); statically filtered generation is in `collector/v2.py` (`generate_queries_v2`). An absent file was not treated as a matching implementation.

- Root selection: `root_sampling.py`, `natural_sampling.py`, `wm_sampling.py` and `runtime.py` match exactly.
- Legal observation/tracking: `observation.py`, `observation_audit.py`, `runtime.py` and the legal loader in `core.py` match exactly.
- Queries and labels: `core.py`, `v2.py`, `v3.py` and `social_habitat.py` match exactly. The v3 binary view remains class 2 → collision 1; classes 0/1 → collision 0. No legacy static-collision supervision was introduced.
- Static enforcement: `static_feasibility.py` declares `eventwm-static-swept-v2.2`; its source and `static_habitat.py`/`v2.py` match exactly.
- Replay/integrity: the canonical collector `validate`, worker replay/fingerprints, observation audits and simulator support match exactly.
- Storage: the canonical profile remains named `production`. The new orchestration name `structured_production` is an explicit alias to `--protocol v3 --storage-profile production --no-render`, not a new native serialization implementation. Debug static visuals are disabled in recipes. Completed C1 outputs were checked for forbidden visual archives.
- Native human/controller support: all tracked `falcon/` files and `scripts/controlled/` files included in the 41-file scope match exactly.

## Relevant historical commits

- `64dcf298f66cbe1db13144d0dccec7d82222ecd9`: structured legal observations.
- `cbfe0a907af6af0bb5f265e524647d5aa8f174c1`: static-feasibility v2.2 implementation.
- `82097ac974cba4f8aa604d3319238f7a2fe75a50`: v3 social-outcome labels and structured/no-visual support.
- `bf46195517d33367af061c2982bfbf2ed6f58000`: validated legal targeted/natural root-sampling additions and collision training index.

All these commits are ancestors of the current base. The model/overfit work was created later without a newer commit. Branch names alone would not reveal those uncommitted additions; this audit checks working-tree status and source hashes explicitly.

## Pilot and diagnostic hash evidence

The 40-root training pilot manifest records HEAD `82097ac...` plus working-source hashes. Those bytes, rather than the recorded HEAD alone, identify the code actually used while the next commit was being developed. **All 17 canonical collector entry/module hashes and all three recorded controlled-simulator support hashes match the mass checkout.** The stored `eventwm_scale.py` hash also matches.

Of 22 recorded code files available in this scope, 21 match. The sole mismatch is the historical bounded launcher `scripts/eventwm_v3_sampling_pilot.py`: stored SHA-256 `20b37009cdc0f6f744201171d3a9f4b6381fe79c81b3ee0191c2ca6226398dad`, current/base SHA-256 `53cd3354b7ab34ce0dd4234693bddcd5be9595867f355472d276806bc06bf945`. It was first committed at bf46195; no earlier version of that file exists in reachable Git history. The exact historical launcher diff is therefore unresolved from retained source evidence. This is a pre-base launcher provenance gap, not a post-base collector difference. The new mass runner does not call this historical launcher. Its frozen recipes and canonical collector hashes are retained independently.

All eight frozen inputs/sources listed in `reports/eventwm_overfit/source_integrity.json` still match their recorded hashes, including model, tensorizer, canonical loader, v3 labels, training index, pilot manifest and exact-input artifacts. This verifies that the current original-worktree model/loader inputs match the validated diagnostic record. It does not re-run or claim new training validation.

## New mass orchestration boundary

The uncommitted mass additions implement scene inventory, new initial episode recipes and region coverage, round-robin scheduling, role assignment, process/GPU/thread management, resumability, resource guards, monitoring and benchmark/report tooling. Their sampling distribution and scene/episode inputs deliberately differ from the Castle-only pilot, as requested. They call the unchanged legal root selectors and one-root collector. This audit establishes source provenance; it does not equate new scheduling code with previously validated collector code or prove all 952 scenes will yield accepted roots.

At audit time C1 was allowed to finish; the C24 supervisor was paused before launching any high-concurrency workers. C2/C4/C6 remain disabled. No long production run has started and no release tag has been created. Once this provenance check is closed, only the already-authorized single bounded C24 test remains before release decisions.

## Evidence and limits

- [Machine-readable audit](reports/eventwm_mass_collection/provenance_audit/audit.json)
- [Full source hashes](reports/eventwm_mass_collection/provenance_audit/source_hashes.csv)
- [Original tracked diff](reports/eventwm_mass_collection/provenance_audit/original_tracked_changes.json)
- [Git history](reports/eventwm_mass_collection/provenance_audit/history.txt)
- [Reflog](reports/eventwm_mass_collection/provenance_audit/reflog.txt)

Scope includes all locally available branch tips and remote-tracking refs; no remote fetch was performed. The original worktree and existing datasets were not modified. The new report/evidence files are the only audit writes.

**Collector version parity: PASS. Newer local collector fixes omitted: NONE FOUND. Model/audit work exists only in original uncommitted files: YES. Historical pilot-launcher hash fully reconciled: NO; exact older source unavailable. Production readiness remains subject to the bounded concurrency result.**
