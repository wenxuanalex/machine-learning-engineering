"""SHAP feature attribution monitor.

Loads the registered ChurnModel from MLflow, computes SHAP values on the
validation set using TreeExplainer (for tree-based models) or LinearExplainer
(for logistic regression), then saves a bar chart of mean absolute SHAP values
and logs everything to MLflow.

If a baseline SHAP file exists in reports/, also computes attribution drift:
features whose mean |SHAP| shifted by more than DRIFT_THRESHOLD are flagged.

Usage:
    python src/shap_monitor.py                    # uses @Staging alias
    python src/shap_monitor.py --alias Production
    python src/shap_monitor.py --no-mlflow        # skip MLflow logging
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS_DIR = Path("reports")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "churn_monitoring"

NON_FEATURE_COLS = {
    "customer_id", "is_churn_label", "is_active_label",
    "scoring_date", "macro_lag_months",
}

# Flag feature if its mean |SHAP| share shifts by more than this fraction
DRIFT_THRESHOLD = 0.20


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and str(df[c].dtype) != "datetime64[ms]"
    ]


def _get_feature_names_from_pipeline(pipeline, feature_cols: list[str]) -> list[str]:
    """Extract post-encoding feature names from the fitted sklearn pipeline."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return feature_cols

    names: list[str] = []
    for tf_name, transformer, cols in preprocessor.transformers_:
        if tf_name == "num":
            names.extend(cols)
        elif tf_name == "cat":
            try:
                ohe = transformer.named_steps["encoder"]
                names.extend(ohe.get_feature_names_out(cols).tolist())
            except Exception:
                names.extend(cols)
    return names if names else feature_cols


def compute_shap_values(
    pipeline,
    X: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return (shap_values, feature_names) for the positive-class output."""
    import shap

    X_input = X[feature_cols].copy()
    preprocessor_steps = pipeline[:-1]
    X_transformed = preprocessor_steps.transform(X_input)
    clf = pipeline.named_steps["clf"]
    feature_names = _get_feature_names_from_pipeline(pipeline, feature_cols)

    clf_type = type(clf).__name__
    if clf_type in ("GradientBoostingClassifier", "RandomForestClassifier",
                    "XGBClassifier", "LGBMClassifier", "HistGradientBoostingClassifier"):
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_transformed)
        # Binary tree models may return list [neg, pos] or 2D array
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
    elif clf_type in ("LogisticRegression", "LinearSVC"):
        explainer = shap.LinearExplainer(clf, X_transformed)
        sv = explainer.shap_values(X_transformed)
        if isinstance(sv, list):
            sv = sv[1]
    else:
        # Fallback: use sklearn feature_importances_ or coef_ directly
        print(f"  No SHAP explainer for {clf_type}; using built-in importances")
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            importances = np.ones(X_transformed.shape[1])
        sv = np.outer(np.ones(len(X_transformed)), importances)

    return sv, feature_names


def save_shap_bar_chart(mean_abs_shap: pd.Series, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = mean_abs_shap.sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
    ax.barh(range(len(top)), top.values, color="#1f77b4")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Attribution (SHAP) — Top 20")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def check_attribution_drift(
    current: pd.Series,
    baseline_path: Path,
) -> dict:
    """Compare current SHAP attribution shares against a saved baseline."""
    if not baseline_path.exists():
        return {"baseline_found": False}

    baseline = pd.read_json(baseline_path, typ="series")
    common = current.index.intersection(baseline.index)
    if len(common) == 0:
        return {"baseline_found": True, "common_features": 0, "drifted": []}

    cur_share = (current[common] / current[common].sum()).fillna(0)
    base_share = (baseline[common] / baseline[common].sum()).fillna(0)
    delta = (cur_share - base_share).abs()
    drifted = delta[delta > DRIFT_THRESHOLD].sort_values(ascending=False)

    return {
        "baseline_found": True,
        "common_features": len(common),
        "drifted": [
            {"feature": f, "delta_share": round(float(d), 4)}
            for f, d in drifted.items()
        ],
    }


def run(alias: str = "Production", val_path: str = "data/gold/val_labeled.parquet") -> dict:
    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/ChurnModel@{alias}"
    print(f"  Loading {model_uri} ...")
    pipeline = mlflow.sklearn.load_model(model_uri)

    val_df = pd.read_parquet(val_path)
    feature_cols = _get_feature_cols(val_df)
    X_val = val_df[feature_cols]

    print(f"  Computing SHAP values on {len(X_val):,} samples × {len(feature_cols)} features ...")
    shap_vals, feat_names = compute_shap_values(pipeline, X_val, feature_cols)

    # Map transformed feature names back to mean abs values
    n_feat = min(len(feat_names), shap_vals.shape[1])
    mean_abs = pd.Series(
        np.abs(shap_vals[:, :n_feat]).mean(axis=0),
        index=feat_names[:n_feat],
    ).sort_values(ascending=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    chart_path = REPORTS_DIR / f"shap_importance_{ts}.png"
    save_shap_bar_chart(mean_abs, chart_path)

    shap_json_path = REPORTS_DIR / f"shap_importance_{ts}.json"
    mean_abs.to_json(shap_json_path)

    baseline_path = REPORTS_DIR / "shap_importance_baseline.json"
    drift_info = check_attribution_drift(mean_abs, baseline_path)

    # Save current as the new baseline if none exists
    if not baseline_path.exists():
        mean_abs.to_json(baseline_path)
        print(f"  Baseline saved to {baseline_path}")

    top5 = mean_abs.head(5).to_dict()
    result = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "model_alias": alias,
        "val_path": val_path,
        "n_samples": len(X_val),
        "n_features": len(feature_cols),
        "top5_features": {k: round(float(v), 6) for k, v in top5.items()},
        "attribution_drift": drift_info,
        "chart_path": str(chart_path),
        "json_path": str(shap_json_path),
    }
    return result


def log_to_mlflow(result: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="shap_attribution"):
        mlflow.log_param("model_alias", result["model_alias"])
        mlflow.log_param("n_samples", result["n_samples"])
        mlflow.log_param("n_features", result["n_features"])

        drift = result["attribution_drift"]
        if drift.get("baseline_found"):
            mlflow.log_metric("shap_drifted_features", len(drift.get("drifted", [])))

        for feat, val in result["top5_features"].items():
            safe_name = feat.replace(" ", "_").replace("/", "_")[:40]
            mlflow.log_metric(f"shap_{safe_name}", val)

        mlflow.log_artifact(result["chart_path"])
        mlflow.log_artifact(result["json_path"])
        print(f"  MLflow run logged to experiment '{EXPERIMENT}'")


def main() -> None:
    import mlflow

    parser = argparse.ArgumentParser(description="SHAP feature attribution monitor")
    parser.add_argument("--alias", default="Production", help="Model alias (Staging or Production)")
    parser.add_argument("--val", default="data/gold/val_labeled.parquet")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)
    print("Computing SHAP attribution report...")
    result = run(alias=args.alias, val_path=args.val)

    print(f"  Top features by mean |SHAP|:")
    for feat, val in result["top5_features"].items():
        print(f"    {feat:<40} {val:.5f}")

    drift = result["attribution_drift"]
    if drift.get("baseline_found") and drift.get("drifted"):
        print(f"\n  WARNING: {len(drift['drifted'])} feature(s) with attribution drift > {DRIFT_THRESHOLD:.0%}:")
        for d in drift["drifted"]:
            print(f"    {d['feature']}: Δshare={d['delta_share']:.3f}")
    elif drift.get("baseline_found"):
        print("  Attribution drift: none detected vs baseline")

    print(f"\n  Chart: {result['chart_path']}")

    if not args.no_mlflow:
        try:
            log_to_mlflow(result)
        except Exception as exc:
            print(f"  MLflow logging skipped: {exc}")


if __name__ == "__main__":
    main()
