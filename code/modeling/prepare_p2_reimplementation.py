#!/usr/bin/env python
"""Verify the visible project P2 source and local runtime facade.

Canonical implementation: ``code/modeling/p2_source``.
No archive is unpacked and no source file is rewritten.

Integrity is checked against the committed Git tree object for ``p2_source``.
This is fail-closed: if the source tree differs from the reviewed commit, the
verification step stops before tests or model training.
"""
from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = HERE / "p2_source"
EXPECTED_SOURCE_TREE = "d18d5d35b679eb3d98830db2d45903e432206926"
EXPECTED_PYTHON_FILES = 28


def git_source_tree() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD:code/modeling/p2_source"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Cannot verify the committed p2_source Git tree. Run this notebook from "
            "the checked-out p2-reimplementation repository, not from a copied folder "
            "without .git metadata.\n" + exc.output
        ) from exc


def main() -> None:
    if not SOURCE.is_dir():
        raise RuntimeError(f"Missing canonical P2 source directory: {SOURCE}")

    files = sorted(SOURCE.rglob("*.py"))
    if len(files) != EXPECTED_PYTHON_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_PYTHON_FILES} P2 Python files, found {len(files)}"
        )

    actual_tree = git_source_tree()
    if actual_tree != EXPECTED_SOURCE_TREE:
        raise RuntimeError(
            "P2 source Git-tree mismatch: "
            f"expected {EXPECTED_SOURCE_TREE}, got {actual_tree}. "
            "Do not bypass this check; review the source changes and update the "
            "recorded tree only after approval."
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
        "print('module=',inspect.getfile(ngboost))"
    )
    subprocess.check_call([sys.executable, "-c", code], env=env)

    print(f"Verified {len(files)} P2 Python files")
    print("Verified p2_source Git tree:", actual_tree)
    print("Project-local ngboost facade imports successfully")


if __name__ == "__main__":
    main()
