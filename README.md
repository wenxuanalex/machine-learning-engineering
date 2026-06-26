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
                               PSI monitoring
                               drift_summary · bias_drift · fairness · shap_attribution
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
  gold/          point-in-time feature snapshots + train/val/test splits
dags/
  pipeline_dag.py          monthly: ingest → train → promote
  weekly_inference_dag.py  weekly: predict → monitor
src/
  train.py     train LR / RF / GBM+Optuna, register @Staging
  promote.py   champion/challenger AUC gate, promote to @Production
  predict.py   batch inference, risk tier CSV, SHAP attribution
  monitor.py   PSI drift summary + bias drift + model fairness
utils/
  bronze.py · silver.py · gold.py · timestamps.py
eda/
  eda.ipynb · eda_gold.ipynb
tests/           pytest suite (40 tests)
reports/         drift_summary · bias_drift · fairness · shap_attribution outputs
```

---

## Tests

```bash
docker compose run --rm jupyter pytest tests/ -q
```

40 tests covering bronze ingestion, silver cleaning, gold feature engineering, point-in-time snapshots (cohort distinctness + leakage), timestamp handling, and smoke imports.
