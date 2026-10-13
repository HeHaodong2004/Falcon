# Native static certification recovery, 10 October 2026

Parent scientific epoch: `58b4b0a17c90c5c7de420834b8ef62c44194eee7`.
The parent checkouts remain unchanged. This branch versions the repaired checker
as `eventwm-static-swept-v2.3-native-controller`.

The stopped production root is `v3m_e7c6987aa64211d2f63c`, seed 731030807,
scene `hm3d_KJxdMPgweZG`, attempt 01, branch b027, query q_b85b8f97f044.
Original-code COW and fresh execution both reproduce static contact at 2.448 s,
control 50; the branch terminates at 2.480 s. Robot joints never change.
The contact is Spot foot link 13 against the stage, with signed distance
+0.005716154 m, normal y=0.774835050, and normal force zero. The unchanged
ground-support exclusion requires abs(normal.y)>0.8, so this contact trips the
existing static integrity contract even though it is below foot level.

The nominal checker reconstructs transforms using double precision yaw and
positions. Native BaseVelAction repeatedly converts single precision matrices
through VelocityControl. At the failing pose, native yaw is -2.093582050 versus
nominal -2.093577780; position differs by about 1 micrometre. The native float
transform produces the offending Bullet contact; the nominal transform does not.
A static-only native controller replay reproduces its exact pose and contact
without any physics step, time advance, RNG, human motion or future label.

The repair adds native transform certification after the existing dense nominal
sweep. It uses the original native VelocityControl integration and a separate
static pathfinder, preserves the entire float matrix, and applies the exact
existing static contact/support predicate. Failed certification rejects the
whole movement candidate through the existing bounded 8-path generation law.
SR, Social Outcome, native collision recording, censoring, legal histories,
root selection, TRAIN proposal mixtures and collection-stopping gates remain
unchanged. No failed branch is retained or relabelled.

The original failed query is rejected at control 50. The original proposal recipe
then produces 33 queries which pass native execution, fresh/COW comparison and
the original full validator. A bounded panel covers this root plus TRAIN,
natural-reference and unseen-evaluation roots: 132 retained diagnostic queries,
zero static violations, eight exact scientific fresh/COW trace/history matches.
The panel artifacts are under `/tmp/eventwm_static_recovery_20261010/native_gate`;
original reproduction evidence is under the sibling `original` directory, with
untouched original evidence also in production attempt 01. These are diagnostics,
not a new production dataset.

Historical byte-freeze tests now check repaired static files at the preserved
parent checkout; unchanged protocol, label, human, controller and observation
files are still compared at the current checkout. Full collector tests retain
the fatal static violation and input-boundary assertions.

Production resumes through the isolated operational recovery branch. Its explicit
transition backs up SQLite and the manifest, finishes already validated old-epoch
roots with the original publisher/checker, versions unfinished retries, and leaves
completed records and files untouched. The failed attempt is preserved; the
runner retries the saved episode/recipe through its existing attempt policy.
All saved C24 settings and seeds are retained. The launcher requires both the
old scientific evidence and the new committed recovery gate.
