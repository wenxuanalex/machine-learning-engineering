"""Train churn prediction models and register the best one in MLflow Model Registry.

Trains three model variants (LogisticRegression, RandomForest, GradientBoosting)
against the gold train/val splits. GBM is further tuned with Optuna (20 trials,
each logged as a nested MLflow run). The best model by AUC-ROC on the validation
set is registered as ChurnModel in the Staging stage.

Usage (local):
    python src/train.py

Usage (Docker):
    docker compose run -e MLFLOW_TRACKING_URI=http://mlflow:5000 jupyter python src/train.py
"""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "churn_prediction"
TARGET = "is_churn_label"
SEED = 42
N_OPTUNA_TRIALS = 20

# Columns present in the gold parquet that are not model features
NON_FEATURE_COLS = {
    "customer_id",
    "is_churn_label",
    "is_active_label",
    "scoring_date",
    "macro_lag_months",
}

# CRM columns backed by synthetic data — one-hot encoded
CATEGORICAL_COLS = ["company_size", "vertical", "onboard_channel", "region", "account_manager"]


def load_splits() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    train = pd.read_parquet("data/gold/train_labeled.parquet")
    val = pd.read_parquet("data/gold/val_labeled.parquet")

    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS and c not in ("scoring_date",)]
    # Drop datetime column missed by set check (dtype guard)
    feature_cols = [c for c in feature_cols if str(train[c].dtype) != "datetime64[ms]"]

    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_val = val[feature_cols]
    y_val = val[TARGET]

    return X_train, y_train, X_val, y_val, feature_cols


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [("num", numeric_pipe, numeric_cols)]
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers)


def make_pipeline(classifier, feature_cols: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(feature_cols)),
        ("clf", classifier),
    ])


def find_optimal_threshold(y_true: pd.Series, y_proba) -> float:
    """Return the threshold on y_proba that maximises F1 on the given split."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    return float(thresholds[f1s.argmax()])


def compute_metrics(y_true: pd.Series, y_pred, y_proba) -> dict:
    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "average_precision": average_precision_score(y_true, y_proba),
    }


def train_and_log(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    feature_cols: list[str],
) -> tuple[str, float]:
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        y_pred = pipeline.predict(X_val)
        metrics = compute_metrics(y_val, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        opt_thresh = find_optimal_threshold(y_val, y_proba)
        y_pred_opt = (y_proba >= opt_thresh).astype(int)
        mlflow.log_param("optimal_threshold", round(opt_thresh, 4))
        mlflow.log_metrics({
            "f1_at_optimal_threshold": f1_score(y_val, y_pred_opt),
            "precision_at_optimal_threshold": precision_score(y_val, y_pred_opt),
            "recall_at_optimal_threshold": recall_score(y_val, y_pred_opt),
        })

        signature = mlflow.models.infer_signature(X_train, y_proba)
        mlflow.sklearn.log_model(pipeline, "model", signature=signature)
        mlflow.log_param("feature_cols", ",".join(feature_cols))
        return run.info.run_id, metrics["auc_roc"]


def tune_gbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
) -> tuple[str, float]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
        with mlflow.start_run(run_name=f"gbm_trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            pipe = make_pipeline(GradientBoostingClassifier(**params, random_state=SEED), feature_cols)
            pipe.fit(X_train, y_train)
            auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
            mlflow.log_metric("auc_roc", auc)
        return auc

    with mlflow.start_run(run_name="gbm_optuna_study") as run:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

        best_params = {**study.best_params, "model": "gradient_boosting_tuned"}
        mlflow.log_params(best_params)
        mlflow.log_metric("best_trial_auc_roc", study.best_value)
        mlflow.log_metric("n_trials", N_OPTUNA_TRIALS)

        best_pipe = make_pipeline(
            GradientBoostingClassifier(**study.best_params, random_state=SEED), feature_cols
        )
        best_pipe.fit(X_train, y_train)
        y_proba = best_pipe.predict_proba(X_val)[:, 1]
        y_pred = best_pipe.predict(X_val)
        metrics = compute_metrics(y_val, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        opt_thresh = find_optimal_threshold(y_val, y_proba)
        y_pred_opt = (y_proba >= opt_thresh).astype(int)
        mlflow.log_param("optimal_threshold", round(opt_thresh, 4))
        mlflow.log_metrics({
            "f1_at_optimal_threshold": f1_score(y_val, y_pred_opt),
            "precision_at_optimal_threshold": precision_score(y_val, y_pred_opt),
            "recall_at_optimal_threshold": recall_score(y_val, y_pred_opt),
        })

        signature = mlflow.models.infer_signature(X_train, y_proba)
        mlflow.sklearn.log_model(best_pipe, "model", signature=signature)
        mlflow.log_param("feature_cols", ",".join(feature_cols))

        return run.info.run_id, metrics["auc_roc"]


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_train, y_train, X_val, y_val, feature_cols = load_splits()
    print(f"Training on {len(X_train)} samples | {len(feature_cols)} features | churn rate: {y_train.mean():.1%}")

    results: list[tuple[str, float, str]] = []

    run_id, auc = train_and_log(
        "logistic_regression",
        make_pipeline(LogisticRegression(max_iter=500, C=1.0, random_state=SEED), feature_cols),
        X_train, y_train, X_val, y_val,
        {"model": "logistic_regression", "max_iter": 500, "C": 1.0},
        feature_cols,
    )
    results.append((run_id, auc, "logistic_regression"))
    print(f"  LogisticRegression   AUC={auc:.4f}")

    run_id, auc = train_and_log(
        "random_forest",
        make_pipeline(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED), feature_cols),
        X_train, y_train, X_val, y_val,
        {"model": "random_forest", "n_estimators": 100, "max_depth": 10},
        feature_cols,
    )
    results.append((run_id, auc, "random_forest"))
    print(f"  RandomForest         AUC={auc:.4f}")

    print(f"  GBM Optuna ({N_OPTUNA_TRIALS} trials)...")
    run_id, auc = tune_gbm(X_train, y_train, X_val, y_val, feature_cols)
    results.append((run_id, auc, "gradient_boosting_tuned"))
    print(f"  GBM (tuned)          AUC={auc:.4f}")

    best_run_id, best_auc, best_name = max(results, key=lambda x: x[1])
    model_uri = f"runs:/{best_run_id}/model"
    client = mlflow.tracking.MlflowClient()
    mv = mlflow.register_model(model_uri, "ChurnModel")
    client.set_registered_model_alias("ChurnModel", "Staging", mv.version)

    print(f"\nBest model: {best_name} (AUC={best_auc:.4f})")
    print(f"Registered ChurnModel v{mv.version} @Staging")
