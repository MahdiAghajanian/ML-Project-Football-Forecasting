# Smoke-test protocol

The delivered implementation was tested in four layers.

## 1. Syntax/import regression

All package `.py` modules are compiled with `py_compile` before execution.

## 2. Original MD1 synthetic regression

P1 disabled. Expected verified contracts:

- 16 pre-match rows;
- 336 snapshots;
- 128 pre-match model features;
- 189 snapshot model features;
- 12/12 original leakage tests;
- 15/15 original readiness checks.

This proves the P1 additions do not alter the default baseline behavior.

## 3. Hybrid P1 synthetic smoke test

The synthetic hybrid fixture contains:

- three complete historical double-round-robin seasons;
- one StatsBomb-shaped target season;
- a team that continues across seasons;
- a team that leaves and returns after a same-league gap;
- cross-provider team identity resolution;
- raw Football-Data-shaped extra odds/statistics that are discarded before canonical P1 history.

The test-only recency grid is `6..10`, so all five candidates can be meaningfully evaluated under the minimum-six rule. This is **not** the paper run. Real mode remains `9..100`.

Expected contracts:

- exact 6-feature `total` representation;
- exact 18-feature `homeaway` representation;
- Pearson selection uses train labels only;
- 13/13 hybrid leakage/source tests;
- all P1 readiness checks.

## 4. Full production-shape offline smoke test

Because the execution sandbox used to build the package has no outbound DNS access and no `pyarrow`, a production-shape test was executed with:

- 15 locally seeded Football-Data-format complete EPL seasons;
- 5,700 historical matches total;
- one locally seeded StatsBomb-format complete EPL 2015/16 target season;
- 380 target matches;
- 6,080 combined same-league history rows;
- the **full real recency range 9..100 (92 candidates)**;
- real-mode 20-team/380-match completeness gates;
- actual long/short EPL alias patterns such as `Manchester United` ↔ `Man United`, `Manchester City` ↔ `Man City`, `Tottenham Hotspur` ↔ `Tottenham`, `West Bromwich Albion` ↔ `West Brom`, etc.

The production-shape test passed all historical completeness gates, all 13 hybrid leakage/source tests, and all readiness checks.

A runtime-only pickle-backed Parquet shim was used **only in the sandbox tests** because `pyarrow` is unavailable there. The delivered Colab requirements install `pyarrow`; the package itself was not modified to use pickle.

## 5. Optional k-NN smoke

Both recency modes were structurally exercised on the synthetic data with a reduced test-only `k` set:

- `project_temporal`;
- `paper_diagnostic`.

The primary final P1 method remains training-only Pearson selection.

## Real Colab validation still required

The builder sandbox cannot access `football-data.co.uk` or `raw.githubusercontent.com` directly. Therefore the real public files could not be downloaded in that environment. The provided Colab runner performs this final network-backed run and will hard-block if any real historical season or the StatsBomb target season fails completeness/readiness.
