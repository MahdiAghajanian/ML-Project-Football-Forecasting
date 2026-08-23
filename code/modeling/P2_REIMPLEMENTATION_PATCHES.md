# P2 project reimplementation: source verification and integration patches

The required P2 workflow is the **project reimplementation** under `code/modeling/p2_reimplementation/`. The official PyPI `ngboost` implementation remains a separately labelled historical/library baseline and is not imported by the P2 evaluation path.

## Verified source reconstruction

`p2_user_source/source_tree.tar.xz` is a lossless snapshot of the P2 source supplied for this project. `prepare_p2_reimplementation.py` verifies the archive checksum and then verifies every Python file against `p2_user_source/source_manifest_sha256.json` before producing the runnable package. Verification is fail-closed; checksum or manifest mismatches stop execution.

## Integration-only patches

After verification, the preparation step applies four runtime/integration patches:

1. Internal imports are moved from the collision-prone `ngboost.*` namespace to `p2_reimplementation.*`.
2. Package-version lookup through installed distribution metadata is replaced by a local project version string because this source is executed directly from the repository.
3. `NGBClassifier` exposes `validation_fraction` and `early_stopping_rounds`, matching the parameters returned by the inherited estimator API and therefore allowing `sklearn.clone()` and Pipelines to work.
4. `__sklearn_is_fitted__` is added so modern scikit-learn fitted-state checks recognize trained estimators.

These patches do not change the P2 task definition, distribution equations, Fisher-information calculations, tree fitting, natural-gradient update, or line-search algorithm.

## Provenance statement

The repository records the exact source snapshot and the mechanical changes made for integration. That record establishes reproducibility of the evaluated code; it does not, by itself, prove who originally authored each source line. Any coursework authorship statement should match the actual development history and be defensible independently of the packaging mechanism.
