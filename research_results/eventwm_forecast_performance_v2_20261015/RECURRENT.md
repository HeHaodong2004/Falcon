# Actual-arrival recurrent verification

The bounded audit compares forecasts before and after actual arrived observations
in consecutive development windows. It uses existing legal track IDs only, with
legal acquisition timestamps and legal robot-frame transforms. It neither matches
changed IDs nor uses target positions as observations. Arrivals are advanced to
the next canonical .25s boundary; the existing <=.25s delayed-update approximation
is explicit. Future targets are read only after both forecasts are constructed.

There are 1,167 considered window pairs and 151 supported pairs from 71 roots / 65
scenes. Each root averages its supported pairs before scene-clustered inference.
The following scores concern the next one second after the update.

| Distribution | Observation update | ES | NLL | ADE |
|---|---|---:|---:|---:|
| Released v1 | No new observation | .74635 | -.42158 | 1.37984 |
| Released v1 | Actual arrival assimilated | .52606 | -.67809 | .90815 |
| Aged covariance + conditional scales | No new observation | .73197 | -.45404 | 1.37984 |
| Aged covariance + conditional scales | Actual arrival assimilated | .52051 | -.70071 | .91133 |

For the aged-covariance candidate, assimilation reduces ES by .21146 (paired
scene 95% interval [.16096,.27137]), NLL by .24668 ([.18879,.30487]), and ADE
by .46851m ([.36339,.58657]). Its post-update ADE is slightly worse than v1's;
no improved point-correction claim is made. Both distributions retain useful
actual-observation correction.

Maximum direct-versus-composed position discrepancy and maximum persistent-path
discrepancy are both exactly zero for both models on these real histories. Unit
tests additionally cover all proposed physical variants, query independence,
missing observations, covariance validity, and serialization. The combined
geometry occupancy adapter preserves both trajectory and anonymous clocks and
round-trips through the standalone loader.

Receipts are `incumbent_recurrent_audit/result.json` and
`cal_all_age_cond_recurrent_audit/result.json`. These are development engineering
audits, not new confirmation results. A final selected physical model must run
this audit again; passing the incumbent/probability controls does not verify an
untrained or merely launched candidate.

## Both age/geometry candidate seeds

Final development audits cover 151 supported actual-arrival pairs from 71 roots
and 65 scenes, among 1,167 considered pairs. Both seeds have zero maximum
direct/composed position difference, zero persistent-sample path difference,
and zero anonymous-occupancy composition difference. Primary ES before/after
an observation is .731973/.520513; replicate is .746992/.530846. Primary NLL
is -.454038/-.700713 and replicate -.480071/-.736221. Point-error correction
also improves in both seeds. These are paired stable-ID updates with the
documented <=.25 s delayed-observation approximation, not cross-ID association.

The source-bound complete forecasting suite passes 97 tests. Both candidate
standalone checks reproduce identical arrays for identical random seeds, keep
all numeric outputs finite, and report no forbidden file access. The final
source archive will receive a separate inference check after packaging.

## Final retained roles

The v1 primary and replicate also completed the final recurrent audit, with zero direct/composed position and persistent-path discrepancy. The source-bound release standalone checks for all four v1/age-aware roles pass after the supplementary diagnostic correction. Both seeds of each family remain independent; no test-set seed swap was made. The retained primary is an LDM with a recurrent forecasting belief interface, not a complete Layer-2 WM or an implemented Query-Conditioned Belief Adapter.
