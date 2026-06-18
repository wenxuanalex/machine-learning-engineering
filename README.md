# Customer Churn Prediction — MLOps Pipeline

Binary churn classifier for a UK online retailer. Flags at-risk customers weekly so the business can intervene before they leave.

**Dataset:** UCI Online Retail (541K transactions, Dec 2010 – Dec 2011, 3,309 modelable customers)

---

## Quickstart

**Prerequisites:** Docker Desktop running.

```bash
# 1. Start MLflow tracking server and Jupyter
docker compose up jupyter mlflow -d

# 2. Run the full data pipeline (Bronze → Silver → Gold)
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 jupyter python main.py

# 3. Train models and register the best in MLflow (LR, RF, GBM+Optuna)
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 jupyter python src/train.py

# 4. Promote Staging → Production if AUC improves
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 jupyter python src/promote.py

# 5. Run weekly batch inference + monitoring
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 jupyter python src/predict.py
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 jupyter python src/monitor.py
```

**MLflow UI:** http://localhost:5000 (or 5001 on macOS — AirPlay occupies 5000: `MLFLOW_HOST_PORT=5001 docker compose up mlflow -d`)

**Jupyter:** http://localhost:8888

**Online inference endpoint** (requires a trained Production model):
```bash
docker compose --profile optional up serve -d
# POST http://localhost:5001/invocations
```

**Airflow orchestration:**
```bash
docker compose --profile optional up airflow -d
# UI: http://localhost:8080  (user: admin / pass: admin)
# Two DAGs: churn_data_pipeline (monthly) · churn_weekly_inference (weekly)
```

---

## Architecture

```
Raw CSVs → Bronze (parquet) → Silver (cleaned) → Gold (feature store)
                                                       │
                                              train / val / test splits
                                                       │
                                              MLflow experiment tracking
                                              LR · RF · GBM (Optuna HPT)
                                                       │
                                              Model Registry (Staging → Production)
                                              Champion/challenger AUC gate
                                                       │
                                    ┌──────────────────┴──────────────────┐
                               Batch inference                     Online inference
                               predict.py                          mlflow models serve
                               weekly CSV + parquet                REST /invocations
                                    │
                               Evidently monitoring
                               drift_report · bias_drift · shap_attribution
```

---

## Run Airflow (Local)

This project defines Airflow under the optional Compose profile.

1. Build Airflow image (one-time, or when dependencies change):
- `docker-compose --profile optional build airflow`
2. Start Airflow:
- `docker-compose --profile optional up -d airflow`
2. Open Airflow UI:
- `http://127.0.0.1:8080`
3. Log in with:
- Username: `admin`
- Password: `admin`

The Airflow service startup command enforces `admin`/`admin` on each start,
so credentials stay deterministic across restarts.

Note: Airflow dependencies are baked into `Dockerfile.airflow`, so restarts are
much faster and avoid the long startup install phase that can temporarily return
`ERR_EMPTY_RESPONSE`.

If you changed Airflow environment variables in `docker-compose.yaml`, recreate the service:
- `docker-compose --profile optional up -d --force-recreate airflow`

Useful commands:
- `docker-compose ps`
- `docker-compose logs airflow --tail 120`
- `docker-compose --profile optional down`


## Repository Structure

```
data/
  bronze/        parquet ingestion layer
  silver/        cleaned, timestamped layer
  gold/          feature store + OOT train/val/test splits
dags/
  pipeline_dag.py          monthly: ingest → train → promote
  weekly_inference_dag.py  weekly: predict → monitor
src/
  train.py     train LR / RF / GBM+Optuna, register @Staging
  promote.py   champion/challenger AUC gate, promote to @Production
  predict.py   batch inference, risk tier CSV, SHAP attribution
  monitor.py   Evidently drift + bias drift segment report
utils/
  bronze.py · silver.py · gold.py · timestamps.py
eda/
  eda.ipynb · eda_gold.ipynb
tests/           pytest suite (34 tests)
reports/         drift_report · bias_drift · shap_attribution outputs
```

---

## Gold Layer Feature Store

One row per modelable customer. Written to `data/gold/feature_store.parquet`.

| Window | Dates |
|---|---|
| Observation | 2010-12-01 → 2011-08-31 |
| Label | 2011-09-01 → 2011-12-31 |

`is_churn_label = 1` if no purchase in label window. Macro features lagged by 1 month to prevent leakage.

**Features:** recency, frequency, monetary, avg_basket_size, avg_orderinterarrival_days, product_diversity, cancellation_rate, rolling_30/60/90d_spend, CRM dimensions (company_size, vertical, region, onboard_channel, account_manager), credit/payment terms, macro indicators (FTSE, GDP, CPI, etc.)

---

## Deployment Strategy

Batch inference is the appropriate deployment type for this use case — churn intervention is a weekly business process, not a real-time decision. `promote.py` implements a **champion/challenger** pattern: the Staging model is evaluated on a held-out OOT test set and promoted to Production only if AUC improves by ≥ 1 point (`MIN_DELTA = 0.01`). The promotion decision and AUC delta are logged as MLflow metrics.

An online inference endpoint (`mlflow models serve`) is wired into `docker-compose.yaml` under the `optional` profile for use cases requiring real-time scoring.

---

## Monitoring

Three artefacts are written to `reports/` on every weekly run:

| File | What it covers |
|---|---|
| `drift_report_YYYYMMDD.html` | Evidently: data drift, target drift, data quality |
| `bias_drift_YYYYMMDD.html` | Segment-level churn rate shift across company_size, region, vertical, onboard_channel — alerts at ±10% |
| `shap_attribution_YYYYMMDD.png` | Mean \|SHAP\| bar chart for feature attribution tracking |

---

## Tests

```bash
docker compose run --rm jupyter pytest tests/ -q
```

34 tests covering bronze ingestion, silver cleaning, gold feature engineering, timestamp handling, and smoke imports.
