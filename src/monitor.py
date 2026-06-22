"""Evidently drift monitoring report + bias drift segment analysis.

Compares training data (reference) against the weekly inference batch (current)
and generates two artefacts:

1. drift_report_YYYYMMDD.html  — Evidently report: data drift, target drift, data quality
2. bias_drift_YYYYMMDD.html    — Segment-level churn rate comparison across CRM dimensions
                                  (company_size, region, vertical, onboard_channel)
                                  to detect whether the target is drifting unevenly
                                  for specific customer subgroups.
3. fairness_report_YYYYMMDD.html — Per-segment model performance (selection rate, recall,
                                  precision, FPR) comparing predictions vs true labels;
                                  flags groups the model serves worse than average. A
                                  retrospective audit (needs matured labels).

Usage:
    python src/monitor.py

    python src/monitor.py \\
        --train data/gold/train_labeled.parquet \\
        --feature-store data/gold/feature_store_2011-08-31.parquet \\
        --predictions reports/predictions.parquet \\
        --output-dir reports
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.report import Report

# Columns present in both parquets that are not model features.
NON_FEATURE_COLS: set[str] = {
    "customer_id",
    "is_churn_label",
    "is_active_label",
    "scoring_date",
    "macro_lag_months",
    # Snapshot identifiers — differ by construction between snapshots taken at
    # different scoring dates, so excluded to avoid flagging them as drift.
    "auxiliary_lag_months",
    "auxiliary_snapshot_month",
}

# All macro_* columns are a single point-in-time snapshot broadcast to every row,
# so variance = 0 across customers. Evidently would flag them as drifted (or produce
# degenerate test results) even when nothing has changed — exclude them entirely.
MACRO_PREFIX = "macro_"

CATEGORICAL_COLS: list[str] = [
    "company_size",
    "vertical",
    "onboard_channel",
    "region",
    "account_manager",
]


def _strip_non_features(df: pd.DataFrame, retain: set[str]) -> pd.DataFrame:
    """Drop non-feature columns except those in `retain`, cast Int64 → float64."""
    macro_cols = [c for c in df.columns if c.startswith(MACRO_PREFIX)]
    datetime_cols = [c for c in df.columns if str(df[c].dtype) == "datetime64[ms]"]
    drop = (NON_FEATURE_COLS | set(macro_cols) | set(datetime_cols)) - retain
    df = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    # Evidently passes columns to scipy statistical tests via np.asarray().
    # Pandas nullable Int64 can silently misbehave there; float64 is safe.
    for col in df.select_dtypes(include="Int64").columns:
        df[col] = df[col].astype("float64")
    return df


def _load_reference(train_path: str) -> pd.DataFrame:
    df = pd.read_parquet(train_path)
    return _strip_non_features(df, retain={"is_churn_label"})


# Segments used for bias drift analysis — CRM dimensions that could mask
# unfair prediction shifts across customer subgroups.
BIAS_SEGMENTS: list[str] = ["company_size", "region", "vertical", "onboard_channel"]

# Delta threshold above which a segment's churn rate shift is flagged as a potential bias drift.
BIAS_ALERT_THRESHOLD = 0.10

# Fairness check: max acceptable gap (vs the overall batch) in recall or selection
# rate before a segment group is flagged. Groups smaller than MIN_GROUP_N are
# reported but never flagged — too few rows for a stable estimate.
FAIRNESS_THRESHOLD = 0.10
MIN_GROUP_N = 20


def _load_current(feature_store_path: str, predictions_path: str) -> pd.DataFrame:
    """Load the weekly inference batch.

    predictions.parquet is read and joined so the DAG task dependency
    (run_predict >> run_monitor) enforces correct ordering. The churn_prediction
    column is logged for transparency but dropped before passing to Evidently:
    Evidently requires the prediction column to be present in BOTH reference and
    current, and the training reference has no stored model predictions.
    """
    fs = pd.read_parquet(feature_store_path)
    preds = pd.read_parquet(predictions_path)[["customer_id", "churn_prediction"]]
    df = fs.merge(preds, on="customer_id", how="inner")

    n_churn = int(df["churn_prediction"].sum())
    print(f"Inference batch: {len(df)} customers, {n_churn} predicted churners ({n_churn/len(df):.1%})")

    df = df.drop(columns=["churn_prediction"])
    return _strip_non_features(df, retain={"is_churn_label"})


def _build_column_mapping(df: pd.DataFrame) -> ColumnMapping:
    feature_cols = [c for c in df.columns if c != "is_churn_label"]
    numerical = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    return ColumnMapping(
        target="is_churn_label",
        numerical_features=numerical,
        categorical_features=CATEGORICAL_COLS,
    )


def generate_bias_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: str,
) -> str:
    """Compare churn rates per segment between reference and current.

    For each CRM segment column, computes the churn rate (mean of is_churn_label)
    per group in both reference and current, then flags groups whose rate has shifted
    by more than BIAS_ALERT_THRESHOLD. Writes a standalone HTML table to output_dir.
    """
    rows = []
    for segment in BIAS_SEGMENTS:
        if segment not in reference.columns or segment not in current.columns:
            continue
        ref_rates = reference.groupby(segment)["is_churn_label"].mean().rename("ref_churn_rate")
        cur_rates = current.groupby(segment)["is_churn_label"].mean().rename("cur_churn_rate")
        merged = pd.concat([ref_rates, cur_rates], axis=1).dropna()
        merged["delta"] = merged["cur_churn_rate"] - merged["ref_churn_rate"]
        merged["alert"] = merged["delta"].abs() > BIAS_ALERT_THRESHOLD
        merged["segment"] = segment
        merged.index.name = "group"
        rows.append(merged.reset_index())

    if not rows:
        return ""

    results = pd.concat(rows, ignore_index=True)[
        ["segment", "group", "ref_churn_rate", "cur_churn_rate", "delta", "alert"]
    ]

    def _fmt_pct(v: float) -> str:
        return f"{v:.1%}"

    def _row_style(row: pd.Series) -> list[str]:
        colour = "background-color: #ffe0e0" if row["alert"] else ""
        return [colour] * len(row)

    styled = (
        results.style
        .format({"ref_churn_rate": _fmt_pct, "cur_churn_rate": _fmt_pct, "delta": _fmt_pct})
        .apply(_row_style, axis=1)
        .set_caption(
            f"Bias Drift — segment churn rate shift (reference vs current) | "
            f"alert threshold: ±{BIAS_ALERT_THRESHOLD:.0%}"
        )
    )

    n_alerts = int(results["alert"].sum())
    summary = (
        f"<p><strong>{n_alerts} segment(s) exceed the ±{BIAS_ALERT_THRESHOLD:.0%} alert threshold.</strong> "
        "Highlighted rows indicate potential bias drift — verify whether the shift reflects "
        "genuine customer behaviour change or model degradation on that subgroup.</p>"
    )

    output_path = Path(output_dir) / f"bias_drift_{date.today():%Y%m%d}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("<html><head><meta charset='utf-8'>")
        f.write("<style>body{font-family:Arial,sans-serif;margin:2em} table{border-collapse:collapse;width:100%} "
                "th,td{border:1px solid #ccc;padding:6px 12px;text-align:left} "
                "th{background:#f5f5f5}</style></head><body>")
        f.write("<h2>Bias Drift Report</h2>")
        f.write(summary)
        f.write(styled.to_html())
        f.write("</body></html>")

    print(f"Bias drift report saved: {output_path} ({n_alerts} alert(s))")
    return str(output_path)


def _safe_div(num: float, den: float) -> float:
    return float(num) / den if den else float("nan")


def generate_fairness_report(
    feature_store_path: str,
    predictions_path: str,
    output_dir: str,
) -> str:
    """Per-segment model-performance (fairness) audit on the current batch.

    Unlike the bias-drift report — which compares churn *rates* between reference
    and current — this joins the current batch's TRUE labels with the model's
    PREDICTIONS and computes per-segment model performance: selection rate
    (predicted-positive share), recall (equal-opportunity), precision, and FPR.
    A group is flagged when its recall or selection rate deviates from the overall
    batch by more than FAIRNESS_THRESHOLD.

    This is feasible only because the current batch is the held-out test split,
    whose labels are known. In live operation the churn label is unavailable for
    ~4 months (label-window latency), so in production this runs as a RETROSPECTIVE
    audit once labels mature, not at scoring time.
    """
    fs = pd.read_parquet(feature_store_path)
    preds = pd.read_parquet(predictions_path)[["customer_id", "churn_prediction"]]
    df = fs.merge(preds, on="customer_id", how="inner")

    y = df["is_churn_label"].astype(int)
    p = df["churn_prediction"].astype(int)
    overall_recall = _safe_div(int(((y == 1) & (p == 1)).sum()), int((y == 1).sum()))
    overall_selection = float(p.mean())

    rows = []
    for segment in BIAS_SEGMENTS:
        if segment not in df.columns:
            continue
        for group, g in df.groupby(segment):
            gy = g["is_churn_label"].astype(int)
            gp = g["churn_prediction"].astype(int)
            tp = int(((gy == 1) & (gp == 1)).sum())
            fp = int(((gy == 0) & (gp == 1)).sum())
            fn = int(((gy == 1) & (gp == 0)).sum())
            tn = int(((gy == 0) & (gp == 0)).sum())
            n = len(g)
            recall = _safe_div(tp, tp + fn)
            selection = _safe_div(tp + fp, n)
            recall_gap = abs(recall - overall_recall) if recall == recall else float("nan")
            sel_gap = abs(selection - overall_selection)
            flag = n >= MIN_GROUP_N and (
                (recall == recall and recall_gap > FAIRNESS_THRESHOLD)
                or sel_gap > FAIRNESS_THRESHOLD
            )
            rows.append({
                "segment": segment,
                "group": str(group),
                "n": n,
                "actual_churn": float(gy.mean()),
                "selection_rate": selection,
                "recall": recall,
                "precision": _safe_div(tp, tp + fp),
                "fpr": _safe_div(fp, fp + tn),
                "recall_gap": recall_gap,
                "alert": bool(flag),
            })

    results = pd.DataFrame(rows)

    def _fmt_pct(v: float) -> str:
        return "n/a" if pd.isna(v) else f"{v:.1%}"

    def _row_style(row: pd.Series) -> list[str]:
        colour = "background-color: #ffe0e0" if row["alert"] else ""
        return [colour] * len(row)

    pct_cols = ["actual_churn", "selection_rate", "recall", "precision", "fpr", "recall_gap"]
    styled = (
        results.style
        .format({c: _fmt_pct for c in pct_cols})
        .format({"n": "{:d}"})
        .apply(_row_style, axis=1)
        .set_caption(
            f"Model fairness — per-segment performance on the current batch | "
            f"overall recall {overall_recall:.1%}, overall selection rate {overall_selection:.1%} | "
            f"flag if recall or selection-rate gap > {FAIRNESS_THRESHOLD:.0%} (groups with n≥{MIN_GROUP_N})"
        )
    )

    n_alerts = int(results["alert"].sum())
    summary = (
        f"<p><strong>{n_alerts} segment group(s) exceed the ±{FAIRNESS_THRESHOLD:.0%} fairness gap.</strong> "
        "This compares the model's predictions against true labels per customer segment — "
        "<em>selection rate</em> is the share predicted to churn (demographic-parity view) and "
        "<em>recall</em> is the share of true churners caught (equal-opportunity view). "
        "Highlighted rows are groups the model serves measurably better or worse than average.</p>"
        "<p><em>Possible only because the current batch is the held-out test split with known "
        "labels; in production this is a retrospective audit once the ~4-month label window closes.</em></p>"
    )

    output_path = Path(output_dir) / f"fairness_report_{date.today():%Y%m%d}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("<html><head><meta charset='utf-8'>")
        f.write("<style>body{font-family:Arial,sans-serif;margin:2em} table{border-collapse:collapse;width:100%} "
                "th,td{border:1px solid #ccc;padding:6px 12px;text-align:left} "
                "th{background:#f5f5f5}</style></head><body>")
        f.write("<h2>Model Fairness Report</h2>")
        f.write(summary)
        f.write(styled.to_html())
        f.write("</body></html>")

    print(f"Fairness report saved: {output_path} ({n_alerts} alert(s))")
    return str(output_path)


def generate_drift_report(
    train_path: str = "data/gold/train_labeled.parquet",
    # Reference = the training snapshot (earlier scoring date). Current = a later
    # point-in-time snapshot, so the comparison is a genuine earlier-vs-later drift
    # check rather than two samples of one distribution. See GOLD_SNAPSHOTS /
    # SCORING_SNAPSHOT in utils/gold.py for the snapshot definitions.
    feature_store_path: str = "data/gold/feature_store_2011-08-31.parquet",
    predictions_path: str = "reports/predictions.parquet",
    output_dir: str = "reports",
) -> str:
    reference = _load_reference(train_path)
    current = _load_current(feature_store_path, predictions_path)
    column_mapping = _build_column_mapping(current)

    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    output_path = Path(output_dir) / f"drift_report_{date.today():%Y%m%d}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    print(f"Drift report saved: {output_path}")

    generate_bias_drift_report(reference, current, output_dir)
    generate_fairness_report(feature_store_path, predictions_path, output_dir)

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Evidently drift monitoring report")
    parser.add_argument("--train", default="data/gold/train_labeled.parquet")
    parser.add_argument("--feature-store", default="data/gold/feature_store_2011-08-31.parquet")
    parser.add_argument("--predictions", default="reports/predictions.parquet")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    generate_drift_report(
        train_path=args.train,
        feature_store_path=args.feature_store,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
    )
