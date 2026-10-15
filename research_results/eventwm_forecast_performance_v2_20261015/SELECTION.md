# Frozen development selection

The preferred candidate retains the original fully trained recurrent physical
backbone and adds age-propagated OU covariance, all-window root-balanced
three-second conditional variance fitting, and the fixed geometry-conditioned
anonymous occupancy prior. Seed 20261014 is primary; seed 20261015 is the
independent replication. The exact checkpoint identities and evidence are in
configs/eventwm_performance_v2/final_development_selection.json.

Both seeds improve origin-three-second exact NLL and foreground occupancy AP
with paired scene-bootstrap intervals excluding zero. ES3 changes are small
and inconclusive: −.001047 [−.011214, .007973] and −.001812
[−.012259, .007722]. AP gains are .011688 [.008384, .015557] and
.012224 [.008803, .016124]. All-window old-history proper likelihood and
high-nominal coverage improve. Point forecasts remain unchanged.

This is an explicit multiobjective tradeoff, not an unqualified winner.
All-window ES1 increases 1.39%/2.70%; seed15's paired interval excludes zero.
All-window one-second central coverage is approximately .577 versus nominal
.4923. Its roughly eight-point excess relative to the old aggregate exceeds
the five-point progression heuristic in a secondary population. We accept
and disclose that cost for the large reduction in older-history undercoverage
and improved proper likelihood; we do not claim uniformly calibrated sets.
Primary-origin three-second coverage remains around .82 at nominal .8923.
Foreground occupancy remains below CV. Geometric sample proxies worsen even
while extreme speed/acceleration tails decrease.

The horizon-balanced stopped alternative has lower ES and AP .115125/.115078,
slightly above same-prior CV .113897 as point estimates, but origin-three-second
central coverage is only .374/.361 at nominal .4923. Its 75%-level coverage
is .612/.592 at nominal .7385. This material primary-population calibration
regression prevents selection on ES/AP alone. The earlier stopped alternative
is also retained for its likelihood/occupancy tradeoff, with larger ES1 costs.
Neither alternative can become primary based on confirmation ranking.

The matched new self-only physical family completed both 24,576-update full
schedules but does not retain its smaller-pool advantage after calibration.
All 48,434 fitting windows were used by both selected and final checkpoints.
Raw development ES was preliminary. The finite architecture search is closed.

Before this freeze, both preferred seeds passed recurrent composition,
persistent path sampling, anonymous occupancy composition, actual-arrival
correction, finite deterministic standalone inference and legal-file access
checks. All 97 forecasting tests passed on the frozen Python sources. The
matched six-track inference p95 is 154 ms for both candidate and incumbent.

New confirmation membership is already fixed at 128 roots per bank, excluding
all 350k physical identities. No confirmation target or forecast score was
opened for this decision. The frozen panel includes both preferred seeds, both
v1 fallback seeds, two named probability alternatives, both new full physical
controls, CV and same-prior CV. Confirmation may reject the whole candidate
family only to the predeclared v1 primary/replicate; it cannot choose a test
winning alternative or seed. All fitting used the frozen 350k dataset.
