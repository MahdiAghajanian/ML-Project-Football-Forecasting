#!/usr/bin/env python
"""Verify the visible project P2 source and local runtime facade.

Canonical implementation: ``code/modeling/p2_source``.
No archive is unpacked and no source file is rewritten.

Integrity is checked against the committed per-file SHA-256 manifest after
normalizing text line endings to LF.  This makes verification invariant to Git
checkout settings (LF on Linux/Colab versus CRLF on Windows) while remaining
fail-closed for substantive source changes.
"""
from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "p2_source"
MANIFEST = HERE / "p2_source_manifest_sha256.json"
EXPECTED_PYTHON_FILES = 28


def normalized_sha256(path: Path) -> str:
    """SHA-256 over UTF-8 source normalized to LF line endings."""
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def main() -> None:
    if not SOURCE.is_dir():
        raise RuntimeError(f"Missing canonical P2 source directory: {SOURCE}")
    if not MANIFEST.is_file():
        raise RuntimeError(f"Missing P2 source manifest: {MANIFEST}")

    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = sorted(SOURCE.rglob("*.py"))
    actual_rel = [str(p.relative_to(SOURCE)).replace("\\", "/") for p in files]
    expected_rel = sorted(expected)

    if len(files) != EXPECTED_PYTHON_FILES or len(expected_rel) != EXPECTED_PYTHON_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_PYTHON_FILES} P2 Python files; "
            f"working tree has {len(files)} and manifest has {len(expected_rel)}"
        )

    missing = sorted(set(expected_rel) - set(actual_rel))
    extra = sorted(set(actual_rel) - set(expected_rel))
    if missing or extra:
        raise RuntimeError(f"P2 source file-list mismatch. missing={missing}, extra={extra}")

    bad = []
    for rel in expected_rel:
        got = normalized_sha256(SOURCE / rel)
        want = expected[rel]
        if got != want:
            bad.append(f"{rel}: expected {want}, got {got}")
    if bad:
        raise RuntimeError(
            "P2 source checksum mismatch after LF normalization:\n" + "\n".join(bad)
        )

    for py in files:
        py_compile.compile(str(py), doraise=True)

    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(HERE)
    code = (
        "import ngboost,inspect; "
        "from ngboost import NGBClassifier,NGBRegressor; "
        "from ngboost.distns import Normal,MultivariateNormal,k_categorical; "
        "from ngboost.scores import LogScore; "
        "print('version=',ngboost.__version__); "
        "print('module=',inspect.getfile(ngboost)); "
        "print('source_path=',list(ngboost.__path__))"
    )
    subprocess.check_call([sys.executable, "-c", code], env=env)

    print(f"Verified {len(files)} P2 Python files against normalized SHA-256 manifest")
    print("Line endings are normalized to LF for verification (Windows CRLF is accepted)")
    print("Project-local ngboost facade imports successfully")


if __name__ == "__main__":
    main()
