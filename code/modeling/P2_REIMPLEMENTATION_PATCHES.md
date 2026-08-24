# P2 project reimplementation: direct-source integration

The required P2 implementation is the visible source tree under `code/modeling/p2_source/`. The historical PyPI `ngboost==0.5.11` workflow remains only as a separately labelled library baseline.

## Runtime integration

The project now uses the P2 source the same way the earlier modeling workflow used the `ngboost` package: modeling code imports `from ngboost import NGBClassifier, NGBRegressor`, but `code/modeling/ngboost/__init__.py` is a repository-local facade whose submodules resolve directly to `p2_source/`. The P2 requirements file does not install external `ngboost`.

The source files in `p2_source/` are not reconstructed from an archive and are not copied into another implementation tree. `prepare_p2_reimplementation.py` is now a verifier only: it checks the exact 28-file Python source list and SHA-256 manifest, compiles the source, and confirms that the local `ngboost` facade imports successfully.

## Compatibility layer

The facade adds only integration behavior needed by the existing sklearn-based evaluator:

1. It exposes `p2_source` under the project-local `ngboost` package name so existing imports continue to work without PyPI NGBoost.
2. It supplies a local version string instead of requiring installed-package metadata.
3. Its `NGBClassifier` facade exposes `validation_fraction` and `early_stopping_rounds` so `sklearn.clone()` and Pipelines see a constructor consistent with inherited estimator parameters.
4. It adds `__sklearn_is_fitted__` to the shared NGBoost class for modern sklearn fitted-state checks.

These integration changes do not replace or rewrite the natural-gradient loop, Fisher-information calculations, distribution equations, base-tree fitting, subsampling, line search, or boosting update in `p2_source/`.

## Source integrity

`p2_source_manifest_sha256.json` records the expected SHA-256 digest of every Python source file. Verification is fail-closed: missing, extra, or modified source files stop the P2 run before tests or training.

Authorship statements in the coursework should reflect the real development history; the repository does not infer authorship merely from packaging or hashes.
