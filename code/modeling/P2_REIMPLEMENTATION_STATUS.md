# P2 reimplementation status

Branch: `p2-reimplementation`

## Canonical roles

- **Required P2 implementation:** `code/modeling/p2_source/`, used through the repository-local `code/modeling/ngboost/` facade.
- **Official NGBoost library baseline:** historical `ngboost==0.5.11` notebook/output, read only for labelled comparison after the P2 run.

## Static implementation status

| Requirement | Status before full run |
|---|---|
| Visible direct P2 source tree | Ready: `p2_source/` |
| 28-file source integrity verification | Ready: fail-closed SHA-256 manifest |
| External `ngboost` excluded from P2 requirements | Ready |
| Old-style `from ngboost import ...` integration | Ready via local facade |
| sklearn clone/Pipeline compatibility | Ready via facade |
| Task C full metrics + per-class P/R/F1 | Implemented |
| Task C reliability + confusion + row predictions + worst 10 | Implemented |
| Task R MAE/RMSE/correlation/NLL + 50/80/95% coverage | Implemented |
| Task R intervals + row predictions + worst 10 | Implemented |
| Task L outcome live/raw/calibrated/frozen pre-match | Implemented |
| Task L outcome metrics by minute and phase | Implemented |
| Task L phase reliability figures | Implemented by finalizer |
| Task L margin live/frozen metrics by minute and phase | Implemented |
| Multivariate validation selection + NLL/Energy/coverage/eigenvalues/correlation | Implemented |
| Derived multivariate H/D/A probabilities | Implemented |
| Validation-only covariance dispersion correction | Implemented |
| Sampled peak RSS + p50/p95/p99 latency | Implemented |
| Unified official-library comparison C/R/L outcome/L margin | Implemented by finalizer |
| Feature contract independent of historical model run_config | Implemented |
| Explanatory Colab workflow | Expanded |
| Strict evidence-derived completion checklist | Implemented by finalizer |
| Expanded P2 report insert | Implemented by finalizer |

## Not complete until execution

No new P2 metric should be reported until the full notebook succeeds. A complete run must generate `code/modeling/outputs/p2_reimplementation_full/`, including metric tables, row-level predictions, reliability/interval figures, fitted models, compute/latency evidence, the unified official-library comparison, the strict completion checklist, and the report insert.

The executed notebook, final report PDF, final evidence archive, and any TA paper/sign-off or authorship attestation remain submission-stage requirements outside the static code changes.
