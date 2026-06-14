# machine-learning-engineering
# Customer Churn Prediction

## Project Overview
Predicting customer churn for a UK online retailer using
transactional data, to flag customers at risk so the business can
intervene before they leave.

## Business Problem
The retailer has a high one time customer rate. This project builds
a binary classifier that produces a churn probability score per customer.

## Dataset
- **Anchor dataset:** UCI Online Retail (transactions, Dec 2010 – Dec 2011)
- **Auxiliary datasets:** to be added

## Repository Structure
- `data/` — raw and processed data
- `eda/` — exploratory data analysis notebooks
to be added: data engineering, backend, frontend, llm eng

## Gold Layer (G1) Feature Store

Gold output is written to `data/gold/feature_store.parquet` with one row per modelable customer.

### Windowing and labels
- Observation window: `2010-12-01` to `2011-08-31`
- Label window: `2011-09-01` to `2011-12-31`
- Label definition:
	- `is_churn_label = 1` if customer has no purchase in label window
	- `is_active_label = 1` if customer has at least one purchase in label window

### Leakage control
- Customer behavior features use only observation-window transactions.
- Macro enrichment is lagged by `macro_lag_months` (default `1`) to simulate publication delay.

### Core G1 feature list
- `recency`
- `frequency`
- `monetary`
- `avg_basket_size`
- `avg_order_interarrival_days`
- `product_diversity`
- `cancellation_rate`
- `rolling_30d_spend`
- `rolling_60d_spend`
- `rolling_90d_spend`

---

## Running the pipeline

### Prerequisites

```bash
pip install -r requirements.txt
# or use the fyp conda env
```

Place the raw CSVs in `data/raw/`:
- `data.csv` (UCI Online Retail, latin-1 encoded)
- `ancillary_20101201_to_20111231.csv` (market + macro + holidays)
- `bronze_customer_metadata_synthetic.csv`

---

### Option A — Docker (recommended)

```bash
# Start MLflow + Jupyter
docker compose up -d mlflow jupyter

# Run the full pipeline inside the container
docker compose exec -e MLFLOW_TRACKING_URI=http://mlflow:5000 jupyter python main.py
docker compose exec -e MLFLOW_TRACKING_URI=http://mlflow:5000 jupyter python src/data_quality.py
docker compose exec -e MLFLOW_TRACKING_URI=http://mlflow:5000 jupyter python src/train.py
docker compose exec -e MLFLOW_TRACKING_URI=http://mlflow:5000 jupyter python src/monitor.py
docker compose exec -e MLFLOW_TRACKING_URI=http://mlflow:5000 jupyter python src/shap_monitor.py

# View MLflow UI
open http://localhost:5000
```

---

### Option B — Local

```bash
# 1. Start MLflow tracking server
python -m mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlruns.db

# 2. In a separate terminal, set the tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000   # Mac/Linux
# $env:MLFLOW_TRACKING_URI = "http://localhost:5000"  # Windows PowerShell

# 3. Bronze → Silver → Gold
python main.py

# 4. Data quality gate (validates Silver + Gold before training)
python src/data_quality.py

# 5. Train models + register best as ChurnModel@Staging
python src/train.py

# 6. (optional) Promote Staging → Production if AUC delta >= 1%
python src/promote.py

# 7. Batch score customers
python src/predict.py

# 8. Feature drift report (Evidently) — saved to reports/
python src/monitor.py

# 9. SHAP attribution report — saved to reports/
python src/shap_monitor.py

# 10. Run tests
pytest tests/
```

---

## Pipeline overview

```
Raw CSVs
   │
   ▼
main.py ──► Bronze (raw parquet) ──► Silver (cleaned) ──► Gold (ML-ready)
                                                              │
                                             src/data_quality.py (gate)
                                                              │
                                                      src/train.py
                                                    (LR / RF / GBM+Optuna)
                                                              │
                                                      src/promote.py
                                                  (Staging → Production)
                                                              │
                                                      src/predict.py
                                                   (batch churn scores)
                                                              │
                                          src/monitor.py   src/shap_monitor.py
                                         (feature drift)  (SHAP attribution)
```

### Key outputs
| Path | Description |
|------|-------------|
| `data/gold/feature_store.parquet` | 3,309 customers × 40 features |
| `data/gold/train_labeled.parquet` | 70% split (2,315 rows) |
| `data/gold/val_labeled.parquet` | 15% split (497 rows) |
| `data/gold/test_labeled.parquet` | 15% split (497 rows) |
| `reports/drift_report_*.html` | Evidently drift report (open in browser) |
| `reports/shap_importance_*.png` | SHAP feature importance bar chart |
| MLflow UI `localhost:5000` | All runs, metrics, registered models |

