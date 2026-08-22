"""Execute the NGBoost notebook on the local CPU and retain its cell outputs."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SOURCE = HERE / "Football_Forecasting_NGBoost_Colab.ipynb"


def prepare_local_runtime() -> None:
    """Keep runtime caches writable and contained inside the ignored environment."""
    runtime_root = PROJECT_ROOT / ".venv" / "runtime"
    runtime_paths = {
        "IPYTHONDIR": runtime_root / "ipython",
        "MPLCONFIGDIR": runtime_root / "matplotlib",
        "NUMBA_CACHE_DIR": runtime_root / "numba",
        "JOBLIB_TEMP_FOLDER": runtime_root / "joblib",
    }
    for variable, path in runtime_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["MPLBACKEND"] = "Agg"

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def notebook_run_mode(notebook: nbformat.NotebookNode) -> str:
    """Read RUN_MODE from the configuration cell without executing user code."""
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        match = re.search(r'^RUN_MODE\s*=\s*["\'](quick|standard|full)["\']', cell.source, re.MULTILINE)
        if match:
            return match.group(1)
    raise RuntimeError("Could not find RUN_MODE in the source notebook.")


def main() -> None:
    prepare_local_runtime()
    notebook = nbformat.read(SOURCE, as_version=4)
    run_mode = notebook_run_mode(notebook)
    output_dir = HERE / "outputs" / f"ngboost_p1_corrected_{run_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    executed_path = output_dir / f"Football_Forecasting_NGBoost_executed_{run_mode}.ipynb"

    cell_started: dict[int, float] = {}

    def on_cell_start(*, cell: nbformat.NotebookNode, cell_index: int) -> None:
        if cell.cell_type == "code":
            cell_started[cell_index] = time.perf_counter()
            first_line = next((line.strip() for line in cell.source.splitlines() if line.strip()), "")
            print(f"Starting code cell {cell_index + 1}/{len(notebook.cells)}: {first_line[:90]}", flush=True)

    def on_cell_complete(
        *, cell: nbformat.NotebookNode, cell_index: int, **_: object
    ) -> None:
        if cell.cell_type == "code":
            elapsed = time.perf_counter() - cell_started.pop(cell_index, time.perf_counter())
            print(f"Finished code cell {cell_index + 1}/{len(notebook.cells)} in {elapsed:.1f}s", flush=True)

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="ngboost-local",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
        on_cell_start=on_cell_start,
        on_cell_executed=on_cell_complete,
    )

    try:
        client.execute()
    finally:
        # A partial notebook is useful for diagnosing the exact failing cell.
        nbformat.write(notebook, executed_path)

    # The notebook creates the evidence archive before the runner can save this
    # executed copy. Rebuild it here so the final archive includes every output.
    archive_base = output_dir.parent / f"{output_dir.name}_evidence"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=output_dir)
    print(f"Completed notebook: {executed_path}")
    print(f"Refreshed evidence archive: {archive_path}")


if __name__ == "__main__":
    main()
