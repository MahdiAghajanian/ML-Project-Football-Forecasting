# Validation

The extracted package was syntax-compiled and executed end-to-end in `synthetic_demo` mode with 16 matches.

Observed validation result:

- matches: 16
- pre-match rows: 16
- snapshot rows: 336
- readiness gate: all checks passed
- evidence archive: created successfully

The validation runtime did not provide `pyarrow`, so the smoke-test harness substituted a test-only pickle-backed Parquet shim. The delivered Colab runner installs and uses real `pyarrow`; the package itself retains the original Parquet I/O.
