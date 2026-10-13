# EventWM V4 300k production collector archive

This release freezes the collector that produced **300,663 retained primary
queries across 9,111 roots and 530 physical scenes**. The nominal target was
300,000; publication of whole 33-query roots during the final drain explains
the 663-query overshoot. **The dataset, scene assets, model checkpoints and
runtime logs are not distributed in this release.**

The final scientific source is `eac1c27813a753d26fc1756671d8d0bd198a65c6`.
All **479 recorded implementation files** at the repository root retain their
exact production bytes. The separate operational snapshot at
[`archive/eventwm_v4_300k/operations`](archive/eventwm_v4_300k/operations)
preserves all 44 hashed operational files from
`6c869911e0c232bc309e1394eee8265a55ec7ce1`. Its final launch receipt identifies
`80b906c53f1bd1f5b809a7f403bb05207d07ad21`; the later operational commit adds
verification evidence. The scientific and operational launchers have the same
filename but different contents, so their snapshots are kept separately.
Both original histories and all contributing commits are reachable from this
archive branch. No collector implementation, optimization or schema migration
was added for publication.

## Collection and scientific contract

A root is a saved native Spot state with a legally arrived observation history.
Static geometry and NavMesh information generate eight distinct movement paths.
Each path has four fixed variants: immediate normal speed, wait 0.5 seconds,
wait 1 second, and immediate half speed. A stationary WAIT query spans the
largest movement-query duration: **8 × 4 + 1 = 33 primary queries** per accepted
root. Three additional native audit jobs check replay and checkpoint integrity;
they are excluded from the retained count.

Execution uses COMBINED / `COMBO_NO_RENDERER`, fresh native worlds and COW
prefix checkpoints. Native clocks and executable controls determine duration.
The saved pool has 36 workers, while the effective native limit is **C24**;
4 preparers, 2 validators, prefetch 8 and backlog 36 remain distinct settings.
Earlier sessions used backlog 72; the recorded operational transition reduced
it to 36 to bound staging space. These are recorded settings, not new defaults.

**SR=1** means a safely completed executable query: the full control sequence
and native horizon completed without declared physical collision. Social-distance
violations can coexist with SR=1. **SR=0** means a declared collision; early
native robot–human contact determines SR=0 even before the planned horizon.
An unexpected static contact fails the static-feasibility integrity gate and
is never retained. **Non-collision premature termination is invalid and must
never be converted into SR=0.** Research-disc overlap is a diagnostic, rather
than the v3 physical-collision detector. See
[`collector/v3.py`](scripts/collector/v3.py) and the unchanged validator.

Social Outcome is separate: 0=`SAFE_SUCCESS`, 1=`SOCIAL_VIOLATION_SUCCESS`,
2=`HUMAN_COLLISION`. The production PSC distance uses geodesic distance with
Euclidean fallback when no path exists, and a strict distance below 1 metre.

## Legal inputs, ground truth and schemas

The checkpoint/dataset schema is **4**, retained query schema **3**, protocol
`eventwm-social-outcome-v3`, and SR contract `binary-execution-safety-v1`.
Diagnostic v2 audit queries can have schema 2; they are not retained primaries.
The schema definitions are the frozen serializers/dataclasses in
[`collector/core.py`](scripts/collector/core.py),
[`collector/v3.py`](scripts/collector/v3.py),
[`collector/observation.py`](scripts/collector/observation.py), and
[`eventwm_pipeline_io.py`](scripts/eventwm_pipeline_io.py).

| Artifact | Fields / role |
| --- | --- |
| Legal history | `timestamp`, `robot_position`, `robot_yaw`, `previous_command`, `human_tracks`, `tracking_available`, `sensor_keys` |
| Legal human tracks | Observed position, estimated velocity and validity, visible-now flag, first/last observation times, age and observation count; legal `track_id` replaces simulator identity |
| Query | `root_id`, `query_id`, local target, spatial path, nominal timed trajectory, executable controls, control/integration dt, wait, effective wait, speed scale, planned duration, static certificate and robot/generation configuration |
| Labels | `sr_valid`, `sr_label`, native contacts and event times, termination/completion flags, Social Outcome, static-feasibility validity and censoring evidence |
| Native branch truth | Recorded future robot/human states, contacts, timestamps, controls and future arrived observations; supervision only |
| Combined SQLite index | `roots(root_id,dataset,path,role,physical_scene,seed,queries,stats)`, `examples(query_id,root_id,branch_id,sr,social)`, `scenes(physical_scene,scene_role)`, `meta` |

`collector.core.load_wm_input` opens only legal history, query definitions and
image-retention metadata. It enumerates allowed fields and never loads privileged
root state or branch truth. Native human positions are exposed to the model only
through the legal observer: camera FOV/range checks, static scene/rigid-object
ray occlusion, tracking and arrived history. Tracks expire after 2 seconds;
association gate is 0.8 m, prediction horizon 0.5 s, velocity maximum gap 0.5 s.
Camera FOV and range come from the frozen depth-sensor configuration. Renderer
images are not retained in this production profile; `history_images` is `None`.
The final full input-boundary audit checked all 300,663 inputs and passed.

Successor sidecars use `eventwm-successor-horizons-v1` at **0.25, 0.5, 1, 2 and
3 seconds**. Targets contain actual human poses/yaws, timestamp-derived symmetric
finite-difference velocities, future arrived legal observations, physical-event
indicators and terminal state. The nearest actually recorded sample is selected;
its actual timestamp is retained. Pose/velocity/observation masks are false beyond
the observed trace. A determined early collision can still provide an event
target. `horizon_sr` is the observed collision-free horizon, whereas `sr_label`
is full-query SR. Native reported zero velocities are not used as velocity truth.

| Horizon (s) | Pose valid | Censored | Determined early collision while censored |
| --- | ---: | ---: | ---: |
| 0.25 | 297,496 | 3,167 | 3,167 |
| 0.5 | 293,760 | 6,903 | 6,903 |
| 1 | 285,254 | 15,409 | 15,409 |
| 2 | 243,802 | 56,861 | 32,049 |
| 3 | 157,217 | 143,446 | 41,471 |

## Queries, scenes and humans

Candidates interleave deterministic headings and radii 1.5/1/0.65 m, with at most
128 proposals, 0.3 m endpoint separation, projection at most 0.25 m and path length
below 4 m. Nominal translation speed is 0.7 m/s; yaw speed is 1.5 rad/s. Waits and
motion are quantized to the root's control dt. Controls are scaled by the ratio of
control dt to integration dt. Native timestamps remain authoritative; clock
tolerance is 0.008001 seconds. Static certification preserves the nominal dense
sweep and, in the final epoch, exact native float integration plus repeated Bullet
contact refresh. A failed candidate is rejected through the original generation
law; integrity gates remain enabled.

There are 952 compatible scenes in the new collection's frozen inventory; 530
physical scenes have retained combined samples. Families include HM3D, MP3D and
the other native fixture family. Human counts are 0, 1, 2, 4 or 6, with six allocated
human slots, episode-defined start/waypoint configurations and native humanoid
controllers. Native human motion is exogenous to the robot.

V4 TRAIN proposals use 87.5% static geometry-feasible support and 12.5% original
V3 support, with clearance threshold 0.38965485334396366 m. Separate V3 shadow
coverage supplies unchanged natural-reference/unseen recipes. Scene-role splits
are fixed before outcomes: 12.5% held-out physical scenes per large family and
one eighth of eligible recipe opportunities assigned to natural reference.
Variants never cross root/scene splits. TRAIN importance weights were not estimated.
[`scene_coverage.json`](archive/eventwm_v4_300k/scene_coverage.json) contains exact
aggregate retained coverage, not trajectories or episodes.

## Dataset layout and loading for WM training

```text
<new-dataset>/
  pipeline.sqlite3                    # durable WAL checkpoint: meta, roots
  manifest.json, aggregate.json       # exported metadata; SQLite is authoritative
  catalogue/, recipes/               # saved replay inputs
  root_seals/, primary_indices/       # immutable accepted-root proofs and IDs
  roots/<root_id>/attempt_XX/
    legal_observations/history.json, image_storage.json
    queries/queries.json, labels/labels.json
    roots/legal_root.json, privileged_root.json, prefix_truth.json.gz, replay.json
    branch_truth/branches.jsonl.gz    # native future truth: supervision only
    audits/, VALIDATED.json, COMPLETE.json
<combined-index>/verified.sqlite3, integrity.json
<supervision>/<root_id>.jsonl.gz, <root_id>.provenance.json
```

Some root directories are symlinks into the spill archive. Loading requires the
raw dataset, its archive links, legacy roots, combined index and supervision.
The repository contains none of these data files. Select training roots by
`role='train'`; retain natural-reference and unseen roles for evaluation.

```python
import sys, sqlite3, json, gzip
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
from collector.core import load_wm_input
index = Path('/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_300k_combined_index/verified.sqlite3')
with sqlite3.connect(index.as_uri() + '?mode=ro', uri=True) as db:
    dataset, path, root_id, query_id, sr = db.execute(
        'SELECT r.dataset,r.path,r.root_id,e.query_id,e.sr FROM roots r '
        'JOIN examples e USING(root_id) WHERE r.role=? LIMIT 1', ('train',)).fetchone()
inputs = load_wm_input(Path(dataset) / path, query_id)
with gzip.open(Path('/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_300k_supervision') / (root_id + '.jsonl.gz'), 'rt') as f:
    targets = next(json.loads(line) for line in f if json.loads(line)['query_id'] == query_id)
# Feed only inputs to the WM; sr and targets are supervision.
```

## Production start, resume, progress and validation

The original launcher intentionally pins host `cv-w06`, the production interpreter,
Git branch/commit, assets and literal workstation paths. This archive preserves
those checks. Use the exact operational worktree, rather than invoking the
scientific checkout's historical launcher or running the nested snapshot directly.
On the completed workstation the following read-only commands work:

```bash
cd /home/haodong/workspace/projects/eventwm-falcon/production/eventwm-v4-300k/recovery/eventwm-v4-ops-recovery
./scripts/run_eventwm_300k.sh check
./scripts/run_eventwm_300k.sh status
./scripts/run_eventwm_300k.sh verify
./scripts/run_eventwm_300k.sh logs
```

On a fresh deployment matching the recorded configuration, with an empty configured
output and all external assets available, collection starts with
`./scripts/run_eventwm_300k.sh start`. An interrupted compatible checkpoint resumes
with `./scripts/run_eventwm_300k.sh check` followed by
`./scripts/run_eventwm_300k.sh resume`. Graceful stop is
`./scripts/run_eventwm_300k.sh stop`; `recovery-status` inspects the bounded recovery
guard. The launcher refuses duplicate collectors, incompatible checkpoints,
nonempty fresh destinations and an already reached combined target. Therefore
start/resume on this completed dataset correctly refuse. No archival validation
starts or stops production.

For an independent clone, the original commits remain available. These commands
materialize the two identities into NEW checkout destinations:

```bash
git worktree add --detach .archive-runtime/science eac1c27813a753d26fc1756671d8d0bd198a65c6
git worktree add -b research/eventwm-v4-300k-recovery-ops-20261010 .archive-runtime/operations 6c869911e0c232bc309e1394eee8265a55ec7ce1
```

That branch must be unused in the clone. Running the original launchers also
requires the literal paths in the preserved production configuration, the frozen
reference `dd1400e0121a8353d79be12d18f98c15bb17e43f`, parent/recovery checkouts,
recorded native gates, legacy verification proof and external assets. Merely
relocating a checkpoint does not make its path/release digests compatible.
Another workstation needs an explicit reviewed path/environment binding release;
none is silently synthesized here. Original installation guidance is retained in
[`docs/FALCON_SETUP.md`](docs/FALCON_SETUP.md); the observed native build is
Habitat-Sim 0.3.1 headless Bullet, Python 3.9.19, NumPy 1.23.5 and PyTorch 1.12.1.
[`environment_observed.json`](archive/eventwm_v4_300k/environment_observed.json)
records package versions observed at archival, rather than claiming a historical
complete environment lock.

The archive supplies portable read-only checks; no native generation is launched:

```bash
python3 scripts/eventwm_archive.py verify
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_pipeline_runner.py --help
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_archive.py inspect \
  --index /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_300k_combined_index/verified.sqlite3
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_archive.py load-smoke \
  --index /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_300k_combined_index/verified.sqlite3 \
  --supervision /home/haodong/workspace/projects/eventwm-falcon/output/eventwm_300k_supervision --limit 12
```

Production validation seals every accepted root after original audit, canonical/raw
equality, label validity and compressed/decoded hash checks. The combined index
enforces unique IDs/seeds and physical-scene roles. Its index helper and full
input/Successor audits are preserved in the operational snapshot. Archived final
audit receipts cover all 300,663 queries. Full reruns can be expensive; this
publication reuses those completed full audits and runs bounded read-only checks.

## Exact completed-dataset provenance

| Source commit | Retained roots | Retained queries |
| --- | ---: | ---: |
| `4b01a7a699462967172c9d61485dea4b40a41f78` — verified legacy | 910 | 30,030 |
| `06e6afe0fa334265337631c1df833c78977a9a86` — initial pipeline | 40 | 1,320 |
| `44f0f95b96323f83a59728cab7766f0a5f52ad49` — validated audit cache | 332 | 10,956 |
| `58b4b0a17c90c5c7de420834b8ef62c44194eee7` — TRAIN geometry/shadow support | 1,856 | 61,248 |
| `2e792aa26f677e4d6b83bbbefebb733c0efa8e63` — native float certification | 1,639 | 54,087 |
| `eac1c27813a753d26fc1756671d8d0bd198a65c6` — Bullet contact refresh | 4,334 | 143,022 |

Completed new roots record their own `source_commit`; legacy roots inherit their
immutable manifest's frozen commit. Historical roots were not replayed or relabelled
at source transitions. Reproducing the final source alone does not reproduce the
six-epoch collection history. The two static repairs are documented in the frozen
operational recovery documents and gates; no individual invalid branch was retained.

Role counts are TRAIN **259,545**, natural reference **19,866**, unseen evaluation
**21,252**. SR positives are **253,688**, negatives **46,975**. Social classes
0/1/2 have **216,529 / 37,159 / 46,975** queries. Successor coverage is complete.
Integrity, legal input boundary and Successor audits all PASS; duplicate query IDs,
root IDs, accepted seeds, scene-role conflicts and retained static violations are zero.

The first verified target observation was **13 October 2026, 07:12:54.074260 London**;
full final audits completed at **07:22:18.299449 London**. The original 12 October
09:20 deadline was missed. Collector state is `stopped_by_user`; its final session
summary has no fatal exception. An exact stop wall-clock timestamp was not recorded
there. The formal standalone dataset-version field is unavailable; the production
protocol/schema and archival identifier are recorded explicitly.

[`production_manifest.json`](archive/eventwm_v4_300k/production_manifest.json)
contains exact aggregate statistics, source/config hashes, completion observations,
hotfixes and limitations. Its supporting records, six epoch hash manifests and
checkpoint configuration contain no sample histories, episodes, future trajectories,
dataset database, model weights or production logs. Native assets and weights remain
external; native dynamics/build sensitivity and the original machine/path bindings
limit byte-identical regeneration. See
[`validation.json`](archive/eventwm_v4_300k/validation.json) for this archival run.
