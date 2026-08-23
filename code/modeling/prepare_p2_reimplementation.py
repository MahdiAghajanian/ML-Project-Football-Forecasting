#!/usr/bin/env python
"""Reconstruct the project P2 implementation and apply integration-only patches.

The committed source archive is a lossless snapshot of the supplied P2 source.
Every Python file is verified against source_manifest_sha256.json before patching.
The preparation step does not disable or bypass verification.
"""
from __future__ import annotations

import hashlib
import json
import py_compile
import shutil
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "p2_user_source" / "source_tree.tar.xz"
MANIFEST = HERE / "p2_user_source" / "source_manifest_sha256.json"
DEST = HERE / "p2_reimplementation"
EXPECTED_ARCHIVE_SHA256 = "fa2ebe1ea387f9680085f1f719c01a86300703175a8d8983c72ff6a6999f80be"
ORIGINAL_UPLOAD_SHA256 = "654208cf8dc70070cd8657a1f8c7c693e98e0dc66beb5a5ed7e0af71885f8076"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_source(source: Path) -> None:
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = sorted(str(p.relative_to(source)).replace("\\", "/") for p in source.rglob("*.py"))
    if files != sorted(expected):
        raise RuntimeError("P2 source file list differs from committed manifest")
    bad = []
    for rel, wanted in expected.items():
        got = sha256(source / rel)
        if got != wanted:
            bad.append(f"{rel}: expected {wanted}, got {got}")
    if bad:
        raise RuntimeError("P2 source manifest mismatch:\n" + "\n".join(bad))


def patch_namespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace("from ngboost.", "from p2_reimplementation.")
    text = text.replace("import ngboost.", "import p2_reimplementation.")
    path.write_text(text, encoding="utf-8")


def patch_init(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    old = '''try:\n    from importlib.metadata import version\nexcept ImportError:\n    # before python 3.8\n    from importlib_metadata import version\n\n'''
    if text.count(old) != 1 or '__version__ = version(__name__)' not in text:
        raise RuntimeError("Expected package-version metadata block not found")
    text = text.replace(old, "", 1)
    text = text.replace('__version__ = version(__name__)', '__version__ = "project-p2-reimplementation"', 1)
    path.write_text(text, encoding="utf-8")


def patch_classifier_clone(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    old = '''        tol=1e-4,\n        random_state=None,\n    ):\n        assert issubclass(\n            Dist, ClassificationDistn\n        ), f"{Dist.__name__} is not useable for classification."\n        super().__init__(\n            Dist,\n            Score,\n            Base,\n            natural_gradient,\n            n_estimators,\n            learning_rate,\n            minibatch_frac,\n            col_sample,\n            verbose,\n            verbose_eval,\n            tol,\n            random_state,\n        )\n'''
    new = '''        tol=1e-4,\n        random_state=None,\n        validation_fraction=0.1,\n        early_stopping_rounds=None,\n    ):\n        assert issubclass(\n            Dist, ClassificationDistn\n        ), f"{Dist.__name__} is not useable for classification."\n        super().__init__(\n            Dist,\n            Score,\n            Base,\n            natural_gradient,\n            n_estimators,\n            learning_rate,\n            minibatch_frac,\n            col_sample,\n            verbose,\n            verbose_eval,\n            tol,\n            random_state,\n            validation_fraction,\n            early_stopping_rounds,\n        )\n'''
    if text.count(old) != 1:
        raise RuntimeError("Expected NGBClassifier constructor block not found exactly once")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_fitted_state(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    needle = '''    def score(self, X, Y):  # for sklearn\n        return self.Manifold(self.pred_dist(X)._params).total_score(Y)\n'''
    replacement = '''    def __sklearn_is_fitted__(self):\n        """Allow modern sklearn Pipeline/check_is_fitted to detect fitted state."""\n        return self.init_params is not None and len(self.base_models) > 0\n\n    def score(self, X, Y):  # for sklearn\n        return self.Manifold(self.pred_dist(X)._params).total_score(Y)\n'''
    if text.count(needle) != 1:
        raise RuntimeError("Expected sklearn score block not found exactly once")
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")


def compile_all() -> None:
    failures = []
    for py in sorted(DEST.rglob("*.py")):
        try:
            py_compile.compile(str(py), doraise=True)
        except Exception as exc:
            failures.append(f"{py.relative_to(DEST)}: {exc}")
    if failures:
        raise RuntimeError("Prepared P2 package failed Python compilation:\n" + "\n".join(failures))


def main() -> None:
    actual = sha256(ARCHIVE)
    if actual != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"P2 source archive checksum mismatch: expected {EXPECTED_ARCHIVE_SHA256}, got {actual}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with tarfile.open(ARCHIVE, "r:xz") as tf:
            tf.extractall(td, filter="data")
        source = td / "ngboost"
        if not source.is_dir():
            raise RuntimeError("Expected ngboost/ source directory missing")
        verify_source(source)
        if DEST.exists():
            shutil.rmtree(DEST)
        shutil.copytree(source, DEST)

    for py in DEST.rglob("*.py"):
        patch_namespace(py)
    patch_init(DEST / "__init__.py")
    patch_classifier_clone(DEST / "api.py")
    patch_fitted_state(DEST / "ngboost.py")
    compile_all()

    print("Prepared verified P2 reimplementation:", DEST)
    print("Verified archive SHA-256:", EXPECTED_ARCHIVE_SHA256)
    print("Verified all pre-patch Python files against:", MANIFEST.name)
    print("Compiled all prepared Python modules successfully")
    print("Recorded original uploaded ZIP SHA-256:", ORIGINAL_UPLOAD_SHA256)


if __name__ == "__main__":
    main()
