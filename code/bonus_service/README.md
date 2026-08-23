# Bonus: Real-Time Prediction Service

This directory implements Section 11 of the course brief on the isolated `bonus-service` branch. It does not modify the frozen core experiment.

## Design

The service deliberately consumes the existing modeling artifacts instead of reimplementing feature engineering:

- the exact `gold_snapshots.parquet` table produced by the leakage-audited training pipeline is loaded once at startup;
- the final pre-match feature list and live feature list are read from the saved `run_config.json`;
- all fitted estimators are loaded once from the final modeling output directory;
- a request for minute `t` always selects the latest saved snapshot whose `snapshot_rank <= t`, so a request cannot pull a future snapshot;
- snapshot groups are indexed in memory and repeated predictions are cached.

Served models:

- pre-match outcome: final trained Task-C LightGBM;
- pre-match margin: final trained Task-R Nyström Kernel Ridge;
- live outcome: final trained Task-L LightGBM (selected here because it supports direct TreeSHAP explanations in the live dashboard);
- live margin: final trained Task-L Gradient Boosting regressor, which had the best held-out RMSE in the final experiment.

The production choice is intentionally separate from the report ranking: the strongest aggregate live outcome result was NGBoost, while LightGBM is served for outcome probabilities because the bonus requires a live SHAP panel and TreeSHAP applies directly to it.

## Install

Reuse the project modeling environment, then install the small serving layer:

```bash
pip install -r code/bonus_service/requirements_bonus.txt
```

## Start the API

From the repository root:

```bash
uvicorn app:app --app-dir code/bonus_service --host 127.0.0.1 --port 8000
```

Interactive API docs are available at `/docs`.

Endpoints:

- `GET /health`
- `GET /matches?split=test`
- `GET /prematch/{match_id}` — Models 1 and 2 before kick-off
- `GET /predict/{match_id}?minute=60` — Model 3 at the latest causal snapshot at/before minute 60
- `POST /predict` — accepts `match_id` plus either `minute` or exact `snapshot_id`
- `GET /replay/{match_id}` — returns every saved snapshot for dashboard replay

Example request:

```bash
curl "http://127.0.0.1:8000/predict/3754000?minute=60"
```

Use a match ID returned by `/matches`; the example ID above is only illustrative.

## Start the dashboard

With the API running:

```bash
streamlit run code/bonus_service/dashboard.py
```

The dashboard:

- selects a held-out match;
- replays its snapshots;
- plots Home/Draw/Away probabilities through time;
- plots expected final goal margin;
- shows current score;
- infers goal and red-card markers from changes in the same snapshot state;
- displays the top live TreeSHAP attributions for the latest shown snapshot.

## Correctness smoke test

From `code/bonus_service`:

```bash
python smoke_test.py
```

The test checks service health, pre-match probability normalization, causal `snapshot_rank <= requested minute`, live probability normalization, and replay length.

## Latency evidence

Run the API first, obtain a held-out `match_id` from `/matches`, then run:

```bash
python code/bonus_service/benchmark_latency.py --match-id <MATCH_ID> --minute 60 --requests 200
```

The benchmark performs warm-up calls followed by repeated real HTTP requests and reports mean, p50, p95, p99 and maximum end-to-end latency. The output also reports whether p99 is below the project's 200 ms budget.

Do not claim the `<200 ms` requirement in the report until this benchmark has been run on the actual final-defence machine and the measured p50/p95/p99 values have been recorded.

## Final-defence workflow

1. Start FastAPI.
2. Start Streamlit.
3. Choose the TA-selected held-out match.
4. Show `/prematch/{match_id}` for Models 1/2.
5. Move the replay slider through successive snapshots.
6. Narrate changes in probabilities, expected final margin, score/red-card markers and SHAP drivers.
7. Send a hand-crafted `/predict/{match_id}?minute=t` request and explain the JSON response.
8. Show the saved latency benchmark from the same machine.
