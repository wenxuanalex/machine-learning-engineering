# MLOps A+ Implementation Plan
## Customer Churn Prediction — Gap Analysis & Agile Roadmap

> **Grading premise:** Points are awarded for rigorous engineering, monitoring, and MLOps practice — not model accuracy. Every epic below is scoped to close a graded pillar gap.

---

## Current-State Scorecard

| Pillar | Maturity | Gap Severity |
|---|---|---|
| Environment & Containers | PARTIAL — Dockerfile + compose exist, no `.dockerignore`, Airflow profile defined but empty | Medium |
| Data Pipelines | HIGH — Bronze→Silver→Gold fully coded and tested | Low |
| Workflow Orchestration | STUB — Airflow service in compose, **zero DAG files** | Critical |
| Model Registry & Deployment | STUB — MLflow server configured, **zero training/serving code** | Critical |
| Continuous Monitoring | ABSENT — No drift, data quality, or alerting code anywhere | Critical |

---

## Epic 1 — High Priority: Workflow Orchestration (Airflow DAGs)

### Current State
`docker-compose.yaml` declares an Airflow 2.9.3 service under the `optional` profile. The compose volume mounts `./dags:/opt/airflow/dags` but the `dags/` directory does not exist and contains no DAG files. `main.py` currently runs the entire Bronze→Silver→Gold pipeline as a sequential Python script with no scheduling, retries, or dependency graph.

### User Stories

**US-1.1** — *As a Data Engineer, I want a DAG that runs the full ingestion pipeline on a schedule so that the data layers are refreshed automatically without manual intervention.*

**Acceptance Criteria:**
- [ ] `dags/pipeline_dag.py` exists and is importable by Airflow without errors
- [ ] DAG has tasks for: `ingest_bronze`, `clean_silver`, `build_gold`, each mapping to the existing `utils/` functions
- [ ] Task dependencies enforce `bronze >> silver >> gold` execution order
- [ ] DAG is set to run on a `@monthly` schedule (matching the dataset's monthly grain)
- [ ] `docker compose --profile optional up airflow` starts the Airflow webserver and the DAG appears in the UI
- [ ] A triggered manual run completes all three tasks with status `success`
- [ ] Failed tasks trigger at least 2 automatic retries with a 5-minute delay

**Implementation Steps:**

```bash
mkdir -p dags
```

Create `dags/pipeline_dag.py`:

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils.bronze import ingest
from utils.silver import clean_transactions, clean_customer_metadata, clean_macro_monthly, date_dim
from utils.gold import build_gold_feature_store, build_gold_train_test_split

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="churn_data_pipeline",
    default_args=default_args,
    description="Bronze → Silver → Gold data pipeline for churn prediction",
    schedule_interval="@monthly",
    start_date=datetime(2011, 1, 1),
    catchup=False,
    tags=["data-pipeline", "churn"],
) as dag:

    t_bronze = PythonOperator(task_id="ingest_bronze", python_callable=ingest)

    t_silver_tx      = PythonOperator(task_id="clean_silver_transactions", python_callable=clean_transactions)
    t_silver_crm     = PythonOperator(task_id="clean_silver_crm",          python_callable=clean_customer_metadata)
    t_silver_macro   = PythonOperator(task_id="clean_silver_macro",        python_callable=clean_macro_monthly)
    t_silver_datedim = PythonOperator(task_id="build_date_dim",            python_callable=date_dim)

    t_gold_fs     = PythonOperator(task_id="build_gold_feature_store", python_callable=build_gold_feature_store)
    t_gold_splits = PythonOperator(task_id="build_gold_splits",        python_callable=build_gold_train_test_split)

    t_bronze >> [t_silver_tx, t_silver_crm, t_silver_macro, t_silver_datedim]
    [t_silver_tx, t_silver_crm, t_silver_macro, t_silver_datedim] >> t_gold_fs
    t_gold_fs >> t_gold_splits
```

Update `docker-compose.yaml` — add `AIRFLOW__CORE__LOAD_EXAMPLES: "false"` and `PYTHONPATH: /opt/airflow` to the Airflow service so `utils/` is importable.

---

## Epic 2 — Critical Priority: Model Training & Registry

### Current State
`data/gold/train_labeled.parquet`, `val_labeled.parquet`, and `test_labeled.parquet` are fully prepared. `requirements.txt` includes `scikit-learn`, `optuna`, and the MLflow server is configured. **No training script, model notebook, or registered model exists.** The MLflow UI at `localhost:5000` has no experiments.

### User Stories

**US-2.1** — *As an ML Engineer, I want a training script that logs all experiments to MLflow so that every model run is reproducible and comparable.*

**Acceptance Criteria:**
- [ ] `src/train.py` exists and runs end-to-end without errors inside the Docker container
- [ ] At least three model variants are trained (LogisticRegression, RandomForest, GradientBoosting) within one script invocation
- [ ] MLflow experiment `churn_prediction` is created; each run logs: params, metrics (AUC-ROC, F1, precision, recall, average_precision), and the serialized model artifact
- [ ] Best model by AUC-ROC on the validation set is registered in the MLflow Model Registry as `ChurnModel` with stage `Staging`
- [ ] `docker compose up mlflow` followed by `docker compose run jupyter python src/train.py` completes without errors and the MLflow UI shows the experiment

**US-2.2** — *As an ML Engineer, I want Optuna hyperparameter tuning integrated so that the best hyperparameters are found systematically and logged.*

**Acceptance Criteria:**
- [ ] An Optuna study runs ≥ 20 trials for the best-performing model class
- [ ] Each trial is logged as a child MLflow run under the parent experiment
- [ ] Best trial params are logged on the parent run
- [ ] The optimized model outperforms the default-param baseline on AUC-ROC (or is documented as equivalent)

**US-2.3** — *As a Data Scientist, I want a batch inference script so that churn predictions can be generated for a new cohort of customers.*

**Acceptance Criteria:**
- [ ] `src/predict.py` accepts a parquet path as input, loads the `Production`-staged model from MLflow registry, and outputs predictions + probabilities to a parquet file
- [ ] Prediction job runs inside the Docker container without network access to external services
- [ ] Output schema is validated: columns `customer_id`, `churn_probability`, `churn_prediction`, `predicted_at` are present

**Implementation Steps:**

```bash
mkdir -p src mlruns
```

Create `src/train.py`:

```python
import mlflow
import mlflow.sklearn
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MLFLOW_URI = "http://mlflow:5000"
EXPERIMENT = "churn_prediction"
TARGET     = "churned"
SEED       = 42

def load_splits():
    X_train = pd.read_parquet("data/gold/train_labeled.parquet").drop(columns=[TARGET, "customer_id"])
    y_train = pd.read_parquet("data/gold/train_labeled.parquet")[TARGET]
    X_val   = pd.read_parquet("data/gold/val_labeled.parquet").drop(columns=[TARGET, "customer_id"])
    y_val   = pd.read_parquet("data/gold/val_labeled.parquet")[TARGET]
    return X_train, y_train, X_val, y_val

def log_model(name, estimator, X_train, y_train, X_val, y_val, params):
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        estimator.fit(X_train, y_train)
        proba = estimator.predict_proba(X_val)[:, 1]
        pred  = estimator.predict(X_val)
        metrics = {
            "auc_roc":       roc_auc_score(y_val, proba),
            "f1":            f1_score(y_val, pred),
            "avg_precision": average_precision_score(y_val, proba),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(estimator, "model")
        return run.info.run_id, metrics["auc_roc"]

def tune_gbm(X_train, y_train, X_val, y_val, n_trials=20):
    def objective(trial):
        params = {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
            "max_depth":     trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
        }
        with mlflow.start_run(run_name=f"gbm_trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            m = Pipeline([("scaler", StandardScaler()),
                          ("clf", GradientBoostingClassifier(**params, random_state=SEED))])
            m.fit(X_train, y_train)
            auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
            mlflow.log_metric("auc_roc", auc)
        return auc

    with mlflow.start_run(run_name="gbm_optuna_study"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_auc_roc", study.best_value)
        best_model = Pipeline([("scaler", StandardScaler()),
                                ("clf", GradientBoostingClassifier(**study.best_params, random_state=SEED))])
        best_model.fit(X_train, y_train)
        mlflow.sklearn.log_model(best_model, "model")
        return mlflow.active_run().info.run_id, study.best_value

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    X_train, y_train, X_val, y_val = load_splits()

    results = []
    results.append(log_model("logistic_regression",
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=SEED))]),
        X_train, y_train, X_val, y_val, {"model": "logistic_regression"}))
    results.append(log_model("random_forest",
        RandomForestClassifier(n_estimators=100, random_state=SEED),
        X_train, y_train, X_val, y_val, {"model": "random_forest", "n_estimators": 100}))

    gbm_run_id, gbm_auc = tune_gbm(X_train, y_train, X_val, y_val)
    results.append((gbm_run_id, gbm_auc))

    best_run_id = max(results, key=lambda x: x[1])[0]
    client    = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    mv        = mlflow.register_model(model_uri, "ChurnModel")
    client.transition_model_version_stage("ChurnModel", mv.version, "Staging")
    print(f"Best model registered: ChurnModel v{mv.version} → Staging (AUC={max(r[1] for r in results):.4f})")
```

Create `src/predict.py`:

```python
import mlflow.pyfunc
import pandas as pd
from datetime import datetime, timezone

MLFLOW_URI  = "http://mlflow:5000"
MODEL_NAME  = "ChurnModel"
MODEL_STAGE = "Production"

def batch_predict(input_path: str, output_path: str):
    mlflow.set_tracking_uri(MLFLOW_URI)
    model        = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    df           = pd.read_parquet(input_path)
    customer_ids = df["customer_id"]
    features     = df.drop(columns=["customer_id"], errors="ignore")
    proba        = model.predict(features)
    out = pd.DataFrame({
        "customer_id":       customer_ids,
        "churn_probability": proba,
        "churn_prediction":  (proba >= 0.5).astype(int),
        "predicted_at":      datetime.now(timezone.utc).isoformat(),
    })
    out.to_parquet(output_path, index=False)
    print(f"Predictions written to {output_path} ({len(out)} rows)")
```

Add a training task to the Airflow DAG after `build_gold_splits`:

```python
from airflow.operators.bash import BashOperator
t_train = BashOperator(
    task_id="train_and_register",
    bash_command="python /app/src/train.py",
)
t_gold_splits >> t_train
```

---

## Epic 3 — Critical Priority: Continuous Monitoring

### Current State
**Nothing exists.** No Evidently AI, no Prometheus metrics, no Grafana dashboards, no data quality checks beyond the static unit tests that run against fixed parquet files. The grading rubric explicitly calls out bias drift and feature attribution drift.

### User Stories

**US-3.1** — *As an ML Engineer, I want automated data drift detection so that I am alerted when the feature distribution in production diverges from the training baseline.*

**Acceptance Criteria:**
- [ ] `src/monitor.py` generates an Evidently HTML report and JSON summary comparing the training set to a reference "production" dataset
- [ ] Report covers: dataset drift (with per-feature p-values), data quality (nulls, range violations), and target drift
- [ ] Reports are saved to `reports/` with a timestamp in the filename
- [ ] A drift flag (`drift_detected: bool`) is returned and logged as an MLflow metric on the monitoring run
- [ ] Running `python src/monitor.py` inside the Docker container produces a report without errors

**US-3.2** — *As an ML Engineer, I want SHAP-based feature attribution drift tracking so that I can detect silent model degradation before it impacts business metrics.*

**Acceptance Criteria:**
- [ ] `src/shap_monitor.py` computes SHAP values for the production-staged model on both train and a "current" dataset
- [ ] A bar chart of mean absolute SHAP values is saved to `reports/shap_<timestamp>.png`
- [ ] Relative SHAP attribution change per feature is logged as MLflow metrics
- [ ] Any feature with >20% relative attribution shift is flagged in the console output

**US-3.3** — *As a Data Engineer, I want automated data quality checks integrated into the Airflow pipeline so that bad data is caught before it corrupts the feature store.*

**Acceptance Criteria:**
- [ ] A `PythonOperator` wrapper validates each Silver layer output before Gold tasks run
- [ ] Checks include: row count within expected range, no nulls in key columns, value ranges for numeric features, cardinality checks for categoricals
- [ ] If any check fails, the DAG task fails and downstream Gold tasks are blocked
- [ ] A JSON quality report is written to `reports/data_quality_<timestamp>.json`

**Implementation Steps:**

Add to `requirements.txt`:
```
evidently==0.4.33
shap==0.45.1
```

Create `src/monitor.py`:

```python
import mlflow
import pandas as pd
from datetime import datetime
from pathlib import Path
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset

MLFLOW_URI   = "http://mlflow:5000"
REPORTS_DIR  = Path("reports")
TARGET       = "churned"

def run_monitoring(reference_path: str = "data/gold/train_labeled.parquet",
                   current_path:   str = "data/gold/val_labeled.parquet") -> bool:
    REPORTS_DIR.mkdir(exist_ok=True)
    ref     = pd.read_parquet(reference_path)
    cur     = pd.read_parquet(current_path)
    col_map = ColumnMapping(target=TARGET, prediction=None)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=col_map)

    ts        = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    html_path = REPORTS_DIR / f"drift_report_{ts}.html"
    json_path = REPORTS_DIR / f"drift_report_{ts}.json"
    report.save_html(str(html_path))
    report.save_json(str(json_path))

    result        = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    n_drifted      = result["metrics"][0]["result"]["number_of_drifted_columns"]

    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name=f"monitoring_{ts}"):
        mlflow.log_metric("dataset_drift_detected", int(drift_detected))
        mlflow.log_metric("n_drifted_features", n_drifted)
        mlflow.log_artifact(str(json_path))
        mlflow.log_artifact(str(html_path))

    print(f"Drift detected: {drift_detected} | Drifted features: {n_drifted}")
    return drift_detected

if __name__ == "__main__":
    run_monitoring()
```

Create `src/shap_monitor.py`:

```python
import shap
import mlflow
import mlflow.pyfunc
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

MLFLOW_URI   = "http://mlflow:5000"
MODEL_NAME   = "ChurnModel"
MODEL_STAGE  = "Staging"
TARGET       = "churned"
REPORTS_DIR  = Path("reports")
DRIFT_THRESH = 0.20

def compute_shap_drift():
    REPORTS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    model        = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    sklearn_model = model._model_impl

    ref = pd.read_parquet("data/gold/train_labeled.parquet").drop(columns=[TARGET, "customer_id"])
    cur = pd.read_parquet("data/gold/val_labeled.parquet").drop(columns=[TARGET, "customer_id"])

    explainer = shap.Explainer(sklearn_model.predict_proba, ref, max_evals=500)
    shap_ref  = explainer(ref).values[:, :, 1]
    shap_cur  = explainer(cur).values[:, :, 1]

    mean_ref  = pd.Series(abs(shap_ref).mean(axis=0), index=ref.columns)
    mean_cur  = pd.Series(abs(shap_cur).mean(axis=0), index=cur.columns)
    drift_pct = ((mean_cur - mean_ref) / (mean_ref + 1e-9)).abs()

    ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(figsize=(10, 6))
    drift_pct.sort_values().plot.barh(ax=ax)
    ax.axvline(DRIFT_THRESH, color="red", linestyle="--", label=f"{DRIFT_THRESH*100:.0f}% threshold")
    ax.set_title("SHAP Feature Attribution Drift (reference vs current)")
    ax.set_xlabel("Relative |SHAP| change")
    ax.legend()
    chart_path = REPORTS_DIR / f"shap_{ts}.png"
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    flagged = drift_pct[drift_pct > DRIFT_THRESH]
    with mlflow.start_run(run_name=f"shap_monitor_{ts}"):
        for feat, val in drift_pct.items():
            mlflow.log_metric(f"shap_drift_{feat}", val)
        mlflow.log_artifact(str(chart_path))
        if not flagged.empty:
            print(f"ALERT: {len(flagged)} features exceed drift threshold:")
            for f, v in flagged.items():
                print(f"  {f}: {v*100:.1f}%")

if __name__ == "__main__":
    compute_shap_drift()
```

Create `src/data_quality.py`:

```python
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

REPORTS_DIR = Path("reports")

SILVER_RULES = {
    "transactions": {
        "min_rows":      300_000,
        "no_null_cols":  ["CustomerID", "InvoiceDate", "UnitPrice", "Quantity"],
        "range_checks":  {"UnitPrice": (0, 10_000), "Quantity": (-10_000, 10_000)},
    },
    "customer_metadata": {
        "min_rows":           4_000,
        "no_null_cols":       ["CustomerID"],
        "cardinality_checks": {"company_size": ["Small", "Medium", "Large", "Enterprise"]},
    },
}

def validate_dataframe(name: str, df: pd.DataFrame, rules: dict) -> list[dict]:
    failures = []
    if len(df) < rules.get("min_rows", 0):
        failures.append({"check": "min_rows", "expected": rules["min_rows"], "actual": len(df)})
    for col in rules.get("no_null_cols", []):
        nulls = df[col].isna().sum()
        if nulls > 0:
            failures.append({"check": "no_nulls", "column": col, "null_count": int(nulls)})
    for col, (lo, hi) in rules.get("range_checks", {}).items():
        violations = ((df[col] < lo) | (df[col] > hi)).sum()
        if violations > 0:
            failures.append({"check": "range", "column": col, "violations": int(violations)})
    for col, valid_vals in rules.get("cardinality_checks", {}).items():
        bad = df[~df[col].isin(valid_vals)][col].unique().tolist()
        if bad:
            failures.append({"check": "cardinality", "column": col, "unexpected_values": bad})
    return failures

def run_quality_checks(silver_dir: str = "data/silver") -> bool:
    REPORTS_DIR.mkdir(exist_ok=True)
    all_failures = {}
    for name, rules in SILVER_RULES.items():
        df       = pd.read_parquet(f"{silver_dir}/{name}.parquet")
        failures = validate_dataframe(name, df, rules)
        if failures:
            all_failures[name] = failures

    ts     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report = {"timestamp": datetime.utcnow().isoformat(), "passed": not bool(all_failures), "failures": all_failures}
    (REPORTS_DIR / f"data_quality_{ts}.json").write_text(json.dumps(report, indent=2))

    if all_failures:
        raise ValueError(f"Data quality checks FAILED: {list(all_failures.keys())}")
    print("All data quality checks passed.")
    return True

if __name__ == "__main__":
    run_quality_checks()
```

Add quality gate and monitoring tasks to the Airflow DAG:

```python
t_quality = PythonOperator(
    task_id="data_quality_gate",
    python_callable=lambda: __import__("src.data_quality", fromlist=["run_quality_checks"]).run_quality_checks(),
)
t_monitor = PythonOperator(
    task_id="run_drift_monitoring",
    python_callable=lambda: __import__("src.monitor", fromlist=["run_monitoring"]).run_monitoring(),
)

[t_silver_tx, t_silver_crm, t_silver_macro, t_silver_datedim] >> t_quality >> t_gold_fs
t_train >> t_monitor
```

---

## Epic 4 — Medium Priority: Container & Environment Hardening

### Current State
A working `Dockerfile` and `docker-compose.yaml` exist. Missing: `.dockerignore` (causes `data/*.csv` and `mlruns/` to be baked into the image), `reports/` directory is not volume-mounted, MLflow's artifact store is inside the container (not persisted), and the CI pipeline does not run tests inside Docker.

### User Stories

**US-4.1** — *As a DevOps Engineer, I want a `.dockerignore` so that large data files and local artifacts are excluded from the Docker build context.*

**Acceptance Criteria:**
- [ ] `.dockerignore` exists at the repo root
- [ ] `docker build .` completes without copying `data/`, `mlruns/`, `reports/`, or `.git/` into the image
- [ ] Build time is measurably reduced compared to the build without `.dockerignore`

**US-4.2** — *As a DevOps Engineer, I want `reports/` and `mlruns/` to be persisted as Docker volumes so that monitoring artifacts and experiment logs survive container restarts.*

**Acceptance Criteria:**
- [ ] `docker-compose.yaml` mounts `./reports:/app/reports` and `./mlruns:/app/mlruns` for all relevant services
- [ ] After running `docker compose down` and `docker compose up`, the MLflow UI still shows previous experiment runs

**Implementation Steps:**

Create `.dockerignore`:

```
data/
mlruns/
reports/
.git/
.github/
eda/
__pycache__/
*.pyc
.DS_Store
*.csv
*.parquet
.env
```

Update `docker-compose.yaml` — add to the `jupyter` and `airflow` service volumes:

```yaml
    volumes:
      - ./reports:/app/reports
      - ./mlruns:/app/mlruns
      - ./dags:/app/dags
      - ./src:/app/src
```

---

## Epic 5 — Medium Priority: CI/CD Pipeline Enhancement

### Current State
Two GitHub Actions workflows exist. `python-ci.yml` runs `ruff` lint and `pytest` on push/PR to `main`. `dockerfile-ci.yml` builds changed Dockerfiles. Neither workflow runs tests inside Docker, pushes to a container registry, or validates the monitoring scripts.

### User Stories

**US-5.1** — *As a DevOps Engineer, I want the CI pipeline to run the full test suite inside the Docker container so that the tested environment matches production exactly.*

**Acceptance Criteria:**
- [ ] A new job in `python-ci.yml` runs `pytest` via `docker compose run` or `docker run`
- [ ] The job passes on the `main` branch
- [ ] Test results are uploaded as a GitHub Actions artifact

**US-5.2** — *As a DevOps Engineer, I want the CI pipeline to validate the monitoring scripts run without errors so that monitoring regressions are caught before merge.*

**Acceptance Criteria:**
- [ ] CI runs `python src/data_quality.py` against the test fixtures inside the container
- [ ] CI runs `python src/monitor.py` against the gold layer splits inside the container
- [ ] Both jobs must pass before a PR can merge to `main`

**Implementation Steps:**

Add to `.github/workflows/python-ci.yml`:

```yaml
  docker-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t churn-pipeline:ci .
      - name: Run tests in container
        run: docker run --rm -v ${{ github.workspace }}/data:/app/data churn-pipeline:ci pytest tests/ -q
      - name: Run data quality checks
        run: docker run --rm -v ${{ github.workspace }}/data:/app/data churn-pipeline:ci python src/data_quality.py
```

---

## Epic 6 — Low Priority: Advanced Deployment Strategy

### Current State
The MLflow `Staging` → `Production` promotion lifecycle is configured but not exercised. No canary, shadow, or champion/challenger deployment pattern is wired up.

### User Stories

**US-6.1** — *As an ML Engineer, I want a model promotion script so that the best Staging model is automatically evaluated against the Production model before being promoted.*

**Acceptance Criteria:**
- [ ] `src/promote.py` loads both the `Staging` and `Production` ChurnModel versions
- [ ] Both are evaluated on the held-out test set; AUC-ROC is the comparison metric
- [ ] If the Staging model beats Production by ≥ 1% AUC, it is promoted; otherwise promotion is skipped with a warning
- [ ] Promotion decision and delta metrics are logged to MLflow

**Implementation Steps:**

Create `src/promote.py`:

```python
import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.metrics import roc_auc_score

MLFLOW_URI = "http://mlflow:5000"
MODEL_NAME = "ChurnModel"
TARGET     = "churned"
MIN_DELTA  = 0.01

def evaluate(model, df: pd.DataFrame) -> float:
    features = df.drop(columns=[TARGET, "customer_id"], errors="ignore")
    return roc_auc_score(df[TARGET], model.predict(features))

def promote_if_better():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client  = mlflow.tracking.MlflowClient()
    test_df = pd.read_parquet("data/gold/test_labeled.parquet")

    staging_versions    = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    production_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

    if not staging_versions:
        print("No Staging model found. Nothing to promote.")
        return

    staging_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Staging")
    staging_auc   = evaluate(staging_model, test_df)

    if production_versions:
        prod_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
        prod_auc   = evaluate(prod_model, test_df)
        delta      = staging_auc - prod_auc
        print(f"Staging AUC: {staging_auc:.4f} | Production AUC: {prod_auc:.4f} | Delta: {delta:+.4f}")
        if delta >= MIN_DELTA:
            client.transition_model_version_stage(MODEL_NAME, staging_versions[0].version, "Production")
            print(f"Promoted ChurnModel v{staging_versions[0].version} to Production.")
        else:
            print(f"Staging does not beat Production by {MIN_DELTA*100:.0f}%. Promotion skipped.")
    else:
        client.transition_model_version_stage(MODEL_NAME, staging_versions[0].version, "Production")
        print(f"No existing Production model. Promoted ChurnModel v{staging_versions[0].version} to Production.")

if __name__ == "__main__":
    promote_if_better()
```

Add to the Airflow DAG:

```python
t_promote = PythonOperator(
    task_id="promote_model_if_better",
    python_callable=lambda: __import__("src.promote", fromlist=["promote_if_better"]).promote_if_better(),
)
t_monitor >> t_promote
```

---

## Delivery Sequence (Sprint Plan)

| Sprint | Duration | Epics | Deliverable |
|---|---|---|---|
| Sprint 1 | 3 days | Epic 1 + Epic 4 | Airflow DAG runs full pipeline; `.dockerignore` and volume mounts in place |
| Sprint 2 | 4 days | Epic 2 | Training script registers best model; MLflow UI shows 3+ experiments with metrics |
| Sprint 3 | 3 days | Epic 3 | Monitoring + SHAP reports generated; quality gate blocks bad data in DAG |
| Sprint 4 | 2 days | Epic 5 + Epic 6 | CI runs tests in Docker; promotion script wired into DAG end-to-end |

**Total estimated effort: 12 development days.** The data pipeline (Bronze→Silver→Gold) is already the strongest pillar — all sprint value is in the upper stack: orchestration, ML, and monitoring.

---

## Final DAG Execution Order

```
ingest_bronze
    └── clean_silver_transactions ──┐
    └── clean_silver_crm           ├──> data_quality_gate ──> build_gold_feature_store ──> build_gold_splits ──> train_and_register ──> run_drift_monitoring ──> promote_model_if_better
    └── clean_silver_macro         ┘
    └── build_date_dim ────────────┘
```
