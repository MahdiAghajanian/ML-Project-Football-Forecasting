# P2 reimplementation status

Branch: `p2-reimplementation`

This branch separates two roles clearly:

- **P2 project reimplementation** — `p2_reimplementation/`, prepared and evaluated by the new P2 workflow.
- **Official NGBoost library baseline** — historical `ngboost==0.5.11` notebook/output, retained only for an explicit paper-vs-library comparison.

## Static implementation status

| Requirement | Status before full run |
|---|---|
| Verified P2 source reconstruction | Ready; checksum + 28-file manifest fail closed |
| No external `ngboost` runtime dependency in P2 path | Ready |
| sklearn clone/Pipeline compatibility | Patched and testable |
| Task C full metrics + class P/R/F1 | Implemented |
| Task C reliability + confusion matrices | Implemented |
| Task C row predictions + ten worst | Implemented |
| Task R MAE/RMSE/correlation/NLL | Implemented |
| Task R raw/calibrated 50/80/95% coverage | Implemented |
| Task R row predictions + intervals + ten worst | Implemented |
| Task L classification raw/calibrated/frozen pre-match | Implemented |
| Task L classification metrics by minute and phase | Implemented |
| Task L phase reliability + row predictions + worst cases | Implemented |
| Task L regression live/frozen metrics by minute and phase | Implemented |
| Task L regression NLL/coverage + row predictions | Implemented |
| Multivariate validation selection | Implemented |
| Multivariate NLL/Energy/coverage/eigenvalues/correlation | Implemented |
| Multivariate derived H/D/A probabilities | Implemented |
| Validation-only covariance dispersion correction | Implemented |
| Background peak-RSS sampling | Implemented |
| Prediction p50/p95/p99 timing | Implemented |
| Official-library comparison table | Implemented if historical results are present |
| Feature contract independent of old model run_config | Implemented from data-pipeline audit + P1 schema |
| Complete pinned dependencies | Implemented |
| Appendix A derivation source | Present; expand/review before final PDF |

## Not complete until execution

No metric in this branch should be reported as a result until the full P2 run succeeds. The following evidence must be generated and committed or included in the final submission bundle:

- executed P2 notebook;
- `outputs/p2_reimplementation_full/` metric tables, row predictions, figures and fitted models;
- resource/latency evidence;
- `P2_REPORT_INSERT.md` populated by the held-out run;
- final unified Task C/R/L tables containing both the official-library baseline and the P2 reimplementation;
- updated report PDF and submission evidence archive.

TA paper sign-off and any authorship/provenance attestation are external requirements and are not manufactured by this branch.
