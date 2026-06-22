"""Drift monitoring: PSI drift summary + bias drift + model fairness.

Compares the training snapshot (reference) against the latest snapshot (current)
and generates three artefacts:

1. drift_summary_YYYYMMDD.html — One PSI table: each model feature + the target gets a
                                  single Population Stability Index and a stable/moderate/
                                  significant verdict (thresholds 0.10 / 0.25). Macro
                                  features are excluded (snapshot-level, shown as context).
                                  Also written as drift_summary_YYYYMMDD.csv.
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

import numpy as np
import pandas as pd

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
# so variance = 0 across customers and they differ by construction between snapshots
# taken in different months — they would always read as drifted. Excluded from the
# per-feature drift table and reported separately as snapshot-level context.
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
    # Cast pandas nullable Int64 to float64 — numpy ops on Int64 can misbehave
    # (e.g. in the PSI binning); float64 is safe.
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
    (run_predict >> run_monitor) enforces correct ordering and so the predicted
    churn rate can be logged. churn_prediction is dropped before the drift
    comparison, which is computed on features + the true label only.
    """
    fs = pd.read_parquet(feature_store_path)
    preds = pd.read_parquet(predictions_path)[["customer_id", "churn_prediction"]]
    df = fs.merge(preds, on="customer_id", how="inner")

    n_churn = int(df["churn_prediction"].sum())
    print(f"Inference batch: {len(df)} customers, {n_churn} predicted churners ({n_churn/len(df):.1%})")

    df = df.drop(columns=["churn_prediction"])
    return _strip_non_features(df, retain={"is_churn_label"})


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


# PSI (Population Stability Index) thresholds — the industry-standard interpretation.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
PSI_COLOURS = {"Significant": "#ffd6d6", "Moderate": "#fff0cc", "Stable": "#e3f4e1"}


def _psi(reference: pd.Series, current: pd.Series, categorical: bool, bins: int = 10) -> float:
    """Population Stability Index between a reference and current distribution.

    Numeric features are binned on reference deciles; categorical features use their
    levels. A small epsilon avoids div-by-zero on empty bins.
    """
    eps = 1e-6
    if categorical:
        levels = set(reference.dropna().astype(str)) | set(current.dropna().astype(str))
        ref_dist = reference.astype(str).value_counts(normalize=True)
        cur_dist = current.astype(str).value_counts(normalize=True)
        psi = 0.0
        for lvl in levels:
            r = float(ref_dist.get(lvl, 0.0)) + eps
            c = float(cur_dist.get(lvl, 0.0)) + eps
            psi += (c - r) * np.log(c / r)
        return float(psi)
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.nunique() <= 1:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = pd.cut(ref, edges).value_counts(sort=False) / len(ref) + eps
    cur_pct = pd.cut(cur, edges).value_counts(sort=False) / len(cur) + eps
    return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())


def _psi_status(psi: float) -> str:
    if psi > PSI_SIGNIFICANT:
        return "Significant"
    if psi >= PSI_MODERATE:
        return "Moderate"
    return "Stable"


def _distribution_summary(series: pd.Series, categorical: bool) -> str:
    if categorical:
        vc = series.astype(str).value_counts(normalize=True)
        return f"{vc.index[0]} ({vc.iloc[0]:.0%})" if not vc.empty else "n/a"
    return f"{pd.to_numeric(series, errors='coerce').mean():.2f}"


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
    """One-table PSI drift summary: training snapshot (reference) vs latest snapshot
    (current). Each model feature + the target gets a single comparable PSI score and
    a stable/moderate/significant verdict. Also writes bias + fairness reports.
    """
    reference = _load_reference(train_path)
    current = _load_current(feature_store_path, predictions_path)
    feature_cols = [c for c in current.columns if c != "is_churn_label" and c in reference.columns]

    rows = [{
        "Signal": "churn (target)",
        "Type": "target",
        "Reference": f"{reference['is_churn_label'].mean():.1%}",
        "Current": f"{current['is_churn_label'].mean():.1%}",
        "PSI": _psi(reference["is_churn_label"], current["is_churn_label"], categorical=True),
    }]
    for col in feature_cols:
        cat = col in CATEGORICAL_COLS
        rows.append({
            "Signal": col,
            "Type": "categorical" if cat else "numeric",
            "Reference": _distribution_summary(reference[col], cat),
            "Current": _distribution_summary(current[col], cat),
            "PSI": _psi(reference[col], current[col], categorical=cat),
        })

    table = pd.DataFrame(rows)
    table["Status"] = table["PSI"].apply(_psi_status)
    # Target pinned on top; features below, most-drifted first.
    feats = table[table["Type"] != "target"].sort_values("PSI", ascending=False)
    table = pd.concat([table[table["Type"] == "target"], feats], ignore_index=True)

    # Context: predicted churn rate (current) + macro snapshot months (excluded from drift).
    pred_rate = float(pd.read_parquet(predictions_path)["churn_prediction"].mean())
    macro_ref = pd.read_parquet(train_path).get("auxiliary_snapshot_month", pd.Series(["n/a"])).iloc[0]
    macro_cur = pd.read_parquet(feature_store_path).get("auxiliary_snapshot_month", pd.Series(["n/a"])).iloc[0]

    n_sig = int((table["Status"] == "Significant").sum())
    n_mod = int((table["Status"] == "Moderate").sum())

    styled = (
        table.style
        .format({"PSI": "{:.3f}"})
        .apply(lambda r: [f"background-color: {PSI_COLOURS.get(r['Status'], '')}"] * len(r), axis=1)
        .hide(axis="index")
        .set_caption("Feature & target drift — training snapshot (reference) vs latest snapshot (current)")
    )

    legend = (
        "<table><tr><th>Status</th><th>PSI range</th><th>Action</th></tr>"
        "<tr style='background:#e3f4e1'><td>🟢 Stable</td><td>&lt; 0.10</td><td>none</td></tr>"
        "<tr style='background:#fff0cc'><td>🟠 Moderate</td><td>0.10 – 0.25</td><td>watch</td></tr>"
        "<tr style='background:#ffd6d6'><td>🔴 Significant</td><td>&gt; 0.25</td><td>investigate / retrain</td></tr>"
        "</table>"
    )
    headline = (
        f"<p><strong>{n_sig} signal(s) show significant drift, {n_mod} moderate.</strong> "
        f"Target churn {reference['is_churn_label'].mean():.1%} → {current['is_churn_label'].mean():.1%}; "
        f"predicted churn rate (current) {pred_rate:.1%}.</p>"
    )
    context = (
        "<p style='color:#555'>Reference = training snapshot · Current = latest snapshot. "
        f"Macro features excluded (single value per snapshot): reference month {macro_ref}, "
        f"current month {macro_cur}.<br>"
        "Note: PSI compresses for a binary target, so the churn row can read 'Stable' even on a "
        "material rate move — read the target by its rate change (shown), not its PSI.</p>"
    )

    output_path = Path(output_dir) / f"drift_summary_{date.today():%Y%m%d}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("<html><head><meta charset='utf-8'>")
        f.write("<style>body{font-family:Arial,sans-serif;margin:2em} table{border-collapse:collapse} "
                "td,th{border:1px solid #ccc;padding:6px 12px;text-align:left} th{background:#f5f5f5}</style>")
        f.write("</head><body><h2>Drift Summary (PSI)</h2>")
        f.write(headline)
        f.write("<h3>Legend</h3>")
        f.write(legend)
        f.write("<h3>Signals</h3>")
        f.write(styled.to_html())
        f.write(context)
        f.write("</body></html>")

    table.to_csv(Path(output_dir) / f"drift_summary_{date.today():%Y%m%d}.csv", index=False)
    print(f"Drift summary saved: {output_path} ({n_sig} significant, {n_mod} moderate)")

    generate_bias_drift_report(reference, current, output_dir)
    generate_fairness_report(feature_store_path, predictions_path, output_dir)

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PSI drift summary + bias + fairness reports")
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
