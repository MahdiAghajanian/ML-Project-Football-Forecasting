#!/usr/bin/env python
"""Verify the visible project P2 source and its local ngboost runtime facade.

Canonical implementation: ``code/modeling/p2_source``.
No archive is used and no source tree is copied or rewritten.  The script is
fail-closed: missing, extra, or modified Python files stop execution.
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not SOURCE.is_dir():
        raise RuntimeError(f"Missing canonical P2 source directory: {SOURCE}")
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = sorted(str(p.relative_to(SOURCE)).replace("\\", "/") for p in SOURCE.rglob("*.py"))
    wanted = sorted(expected)
    missing = sorted(set(wanted) - set(actual))
    extra = sorted(set(actual) - set(wanted))
    if missing or extra:
        raise RuntimeError(f"P2 source file-list mismatch. missing={missing}, extra={extra}")

    bad = []
    for rel, digest in expected.items():
        got = sha256(SOURCE / rel)
        if got != digest:
            bad.append(f"{rel}: expected {digest}, got {got}")
    if bad:
        raise RuntimeError("P2 source checksum mismatch:\n" + "\n".join(bad))

    for py in sorted(SOURCE.rglob("*.py")):
        py_compile.compile(str(py), doraise=True)

    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(HERE)
    code = (
        "import ngboost; "
        "from ngboost import NGBClassifier,NGBRegressor; "
        "from ngboost.distns import Normal,MultivariateNormal,k_categorical; "
        "from ngboost.scores import LogScore; "
        "print(ngboost.__version__)"
    )
    subprocess.check_call([sys.executable, "-c", code], env=env)
    print(f"Verified {len(actual)} P2 source files from {SOURCE}")
    print("Project-local ngboost facade imports successfully")


if __name__ == "__main__":
    main()
