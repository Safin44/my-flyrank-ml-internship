from __future__ import annotations

"""Reproducible public-safe FlyRank ML capstone.

The pipeline aggregates only the March 2026 feature month and April 2026 label
month from the gated warehouse. It uses a client-grouped holdout, a logistic
regression model, and a transparent March-only CTR-gap baseline. The Hugging Face
read token is read from HF_TOKEN at runtime and is never written to disk or
printed. Only aggregate metrics and charts are exported; no client/content IDs or
row-level prediction files are written.
"""

import argparse
import html
import json
import os
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import duckdb
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATASET_REVISION = "50cbf7c3909d07be4d1b5906b4d09e882e5acbf2"
DATASET_BASE = f"hf://datasets/FlyRank/internship-warehouse@{DATASET_REVISION}"
FACT_BASE = f"{DATASET_BASE}/fact_content_daily_performance"
FEATURE_MONTH = "2026-03"
LABEL_MONTH = "2026-04"
SEED = 42
TEST_SIZE = 0.25
TOP_K = 50
MIN_MARCH_IMPRESSIONS = 100
POSITION_BINS = [-np.inf, 3, 5, 8, 12, 20, np.inf]
POSITION_LABELS = ["≤3", "4–5", "6–8", "9–12", "13–20", ">20"]


def _safe_float(value: object) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    count = min(k, len(y))
    if count == 0:
        return 0.0
    order = np.argsort(-np.asarray(score), kind="stable")[:count]
    return float(np.asarray(y)[order].mean())


def _score_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    return {
        "average_precision": float(average_precision_score(y, score)),
        "precision_at_10": _precision_at_k(y, score, 10),
        "precision_at_50": _precision_at_k(y, score, 50),
        "roc_auc": float(roc_auc_score(y, score)),
    }


def _format_num(value: object, digits: int = 3) -> str:
    return f"{_safe_float(value):,.{digits}f}"


def _format_pct(value: object, digits: int = 1) -> str:
    return f"{100 * _safe_float(value):,.{digits}f}%"


def _query_warehouse(token: str) -> pd.DataFrame:
    """Read two month partitions once and return only page-level aggregates in memory."""
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        safe_token = token.replace("'", "''")
        con.execute(f"CREATE SECRET (TYPE huggingface, TOKEN '{safe_token}')")
        march_path = f"{FACT_BASE}/month={FEATURE_MONTH}/data_0.parquet"
        april_path = f"{FACT_BASE}/month={LABEL_MONTH}/data_0.parquet"
        dim_path = f"{DATASET_BASE}/dim_clients.parquet"
        query = f"""
        WITH client_dim AS MATERIALIZED (
            SELECT client_hash_id AS client_id, TRY_CAST(gsc_data_start AS DATE) AS gsc_data_start
            FROM read_parquet('{dim_path}')
        ),
        history_clients AS MATERIALIZED (
            SELECT client_id FROM client_dim
            WHERE gsc_data_start IS NOT NULL AND gsc_data_start <= DATE '2026-03-01'
        ),
        history_stats AS (
            SELECT
                COUNT(*)::BIGINT AS dimension_clients,
                COUNT(*) FILTER (WHERE gsc_data_start IS NOT NULL AND gsc_data_start <= DATE '2026-03-01')::BIGINT AS clients_started_by_cutoff,
                COUNT(*) FILTER (WHERE gsc_data_start IS NULL)::BIGINT AS clients_missing_start_date,
                COUNT(*) FILTER (WHERE gsc_data_start > DATE '2026-03-01')::BIGINT AS clients_started_after_cutoff
            FROM client_dim
        ),
        march_raw AS MATERIALIZED (
            SELECT report_date, client_hash_id, content_hash_id,
                   gsc_impressions, gsc_clicks, gsc_avg_position
            FROM read_parquet('{march_path}', hive_partitioning=true)
            WHERE report_date >= DATE '2026-03-01'
              AND report_date < DATE '2026-04-01'
              AND gsc_data_available IS TRUE
              AND client_hash_id IN (SELECT client_id FROM history_clients)
        ),
        april_raw AS MATERIALIZED (
            SELECT report_date, client_hash_id, content_hash_id, gsc_clicks
            FROM read_parquet('{april_path}', hive_partitioning=true)
            WHERE report_date >= DATE '2026-04-01'
              AND report_date < DATE '2026-05-01'
              AND gsc_data_available IS TRUE
              AND client_hash_id IN (SELECT client_id FROM history_clients)
        ),
        march_grain_probe AS (
            SELECT COUNT(*) FILTER (WHERE daily_rows > 1)::BIGINT AS duplicate_daily_grain_groups
            FROM (
                SELECT report_date, client_hash_id, content_hash_id, COUNT(*) AS daily_rows
                FROM march_raw
                GROUP BY report_date, client_hash_id, content_hash_id
            ) AS daily_grain
        ),
        april_grain_probe AS (
            SELECT COUNT(*) FILTER (WHERE daily_rows > 1)::BIGINT AS duplicate_daily_grain_groups
            FROM (
                SELECT report_date, client_hash_id, content_hash_id, COUNT(*) AS daily_rows
                FROM april_raw
                GROUP BY report_date, client_hash_id, content_hash_id
            ) AS daily_grain
        ),
        march AS MATERIALIZED (
            SELECT
                client_hash_id AS client_id,
                content_hash_id AS content_id,
                SUM(COALESCE(gsc_impressions, 0))::DOUBLE AS march_impressions,
                SUM(COALESCE(gsc_clicks, 0))::DOUBLE AS march_clicks,
                AVG(NULLIF(TRY_CAST(gsc_avg_position AS DOUBLE), 0))::DOUBLE AS march_avg_position
            FROM march_raw
            GROUP BY client_hash_id, content_hash_id
        ),
        april AS MATERIALIZED (
            SELECT
                client_hash_id AS client_id,
                content_hash_id AS content_id,
                SUM(COALESCE(gsc_clicks, 0))::DOUBLE AS april_clicks
            FROM april_raw
            GROUP BY client_hash_id, content_hash_id
        ),
        april_active_clients AS (
            SELECT DISTINCT client_id FROM april
        )
        SELECT
            m.client_id,
            m.content_id,
            m.march_impressions,
            m.march_clicks,
            CASE WHEN m.march_impressions > 0
                 THEN 100.0 * m.march_clicks / m.march_impressions
                 ELSE NULL END AS march_ctr_pct,
            m.march_avg_position,
            COALESCE(a.april_clicks, 0)::DOUBLE AS april_clicks,
            CASE WHEN a.content_id IS NULL THEN FALSE ELSE TRUE END AS has_april_page_row,
            hs.dimension_clients,
            hs.clients_started_by_cutoff,
            hs.clients_missing_start_date,
            hs.clients_started_after_cutoff,
            mg.duplicate_daily_grain_groups AS march_duplicate_daily_grain_groups,
            ag.duplicate_daily_grain_groups AS april_duplicate_daily_grain_groups
        FROM march AS m
        INNER JOIN april_active_clients AS active USING (client_id)
        LEFT JOIN april AS a USING (client_id, content_id)
        CROSS JOIN history_stats AS hs
        CROSS JOIN march_grain_probe AS mg
        CROSS JOIN april_grain_probe AS ag
        WHERE m.march_impressions > 0
          AND m.march_clicks > 0
        """
        return con.execute(query).df()
    finally:
        con.close()


def _build_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            random_state=SEED,
        )),
    ])


def _write_figures(
    root: Path,
    y_test: np.ndarray,
    model_score: np.ndarray,
    baseline_score: np.ndarray,
    feature_names: list[str],
    importance: list[dict[str, float | str]],
    prevalence: float,
) -> list[str]:
    figure_dir = root / "work" / "figures"
    page_assets = root / "docs" / "assets"
    figure_dir.mkdir(parents=True, exist_ok=True)
    page_assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
    })
    names: list[str] = []

    # Precision-recall curves show ranking quality on the same client-held-out rows.
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for label, score, color in [
        ("Logistic regression", model_score, "#176b73"),
        ("CTR-gap baseline", baseline_score, "#d17b30"),
    ]:
        precision, recall, _ = precision_recall_curve(y_test, score)
        ap = average_precision_score(y_test, score)
        ax.plot(recall, precision, linewidth=2.4, color=color, label=f"{label} (AP {ap:.3f})")
    ax.axhline(prevalence, color="#6b7478", linestyle="--", linewidth=1.4, label=f"Test positive rate {prevalence:.1%}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title="Held-out ranking: precision–recall")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    path = figure_dir / "capstone_precision_recall.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    (page_assets / path.name).write_bytes(path.read_bytes())
    names.append(path.name)

    # Precision@K comparison with base-rate reference.
    k_values = [10, 50]
    model_values = [_precision_at_k(y_test, model_score, k) for k in k_values]
    baseline_values = [_precision_at_k(y_test, baseline_score, k) for k in k_values]
    x = np.arange(len(k_values))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(x - width / 2, model_values, width, label="Logistic regression", color="#176b73")
    ax.bar(x + width / 2, baseline_values, width, label="CTR-gap baseline", color="#d17b30")
    ax.axhline(prevalence, color="#6b7478", linestyle="--", linewidth=1.4, label=f"Test positive rate {prevalence:.1%}")
    ax.set_xticks(x, [f"Precision@{k}" for k in k_values])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of selected pages with decline proxy")
    ax.set_title("Top-ranked pages versus the test base rate")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = figure_dir / "capstone_precision_at_k.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    (page_assets / path.name).write_bytes(path.read_bytes())
    names.append(path.name)

    # Held-out permutation importance; values are AP changes, not causal effects.
    sorted_importance = sorted(importance, key=lambda row: float(row["mean_ap_decrease"]))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = [str(row["feature"]).replace("march_", "March ").replace("log1p_", "log1p ").replace("_", " ") for row in sorted_importance]
    vals = [float(row["mean_ap_decrease"]) for row in sorted_importance]
    colors = ["#176b73" if value >= 0 else "#b85c4a" for value in vals]
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="#54666c", linewidth=1)
    ax.set_xlabel("Mean decrease in held-out average precision when shuffled")
    ax.set_title("Permutation importance (descriptive, correlated signals)")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = figure_dir / "capstone_permutation_importance.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    (page_assets / path.name).write_bytes(path.read_bytes())
    names.append(path.name)
    return names


def _render_report(root: Path, metrics: dict[str, object]) -> None:
    data = metrics["data"]
    split = metrics["split"]
    target = metrics["target"]
    model = metrics["model_metrics"]
    baseline = metrics["baseline_metrics"]
    dummy = metrics["base_rate_reference"]
    importance_rows = metrics["permutation_importance"]
    top10 = metrics["top_risk_aggregates"]
    importance_table = "\n".join(
        f"| {row['feature']} | {_format_num(row['mean_ap_decrease'], 4)} ± {_format_num(row['std'], 4)} |"
        for row in importance_rows
    )
    best_text = (
        f"The logistic model's average precision ({_format_num(model['average_precision'], 3)}) is modestly above the CTR-gap baseline ({_format_num(baseline['average_precision'], 3)}) and no-discrimination reference ({_format_num(dummy['average_precision'], 3)}); its ROC-AUC ({_format_num(model['roc_auc'], 3)}) is near chance, so this is weak, directional ranking evidence."
        if model["average_precision"] > baseline["average_precision"]
        else "The transparent CTR-gap baseline matched or exceeded the logistic model on held-out average precision; keep the simpler rule unless repeated tests justify complexity."
    )
    sensitivity_rows = "\n".join(
        f"| {row['min_march_impressions']:,} | {row['eligible_rows']:,} | {_format_pct(row['positive_rate'])} |"
        for row in metrics["label_sensitivity"]
    )
    report = f"""# Capstone Report — Search Decline Risk and CTR Review

- **Author:** Imam Hossain Safin
- **Lane:** CTR / Engagement Opportunity Scoring
- **Repo:** https://github.com/Safin44/my-flyrank-ml-internship
- **Date:** 2026-10-28

## 0. Abstract

This capstone asks whether a small, interpretable model can rank content items whose clicks fall in the month after a March 2026 decision snapshot. It uses the FlyRank anonymized warehouse daily-performance table, aggregating March features and April outcomes for {data['eligible_rows']:,} eligible items from {data['eligible_clients']} clients. A class-balanced logistic regression uses March-only impressions, clicks, click-through rate, and average position, and is compared with a transparent within-position CTR-shortfall rule on a client-held-out test set. On the held-out clients, logistic regression achieves average precision {_format_num(model['average_precision'], 3)} versus {_format_num(baseline['average_precision'], 3)} for the baseline, against a held-out positive-label base rate of {_format_pct(split['test_positive_rate'])}; Precision@50 is {_format_pct(model['precision_at_50'])} versus {_format_pct(baseline['precision_at_50'])}. ROC-AUC is {_format_num(model['roc_auc'], 3)}, so the result is weak, directional review evidence against a noisy proxy—not a forecast of Google’s algorithm or evidence that editing a page causes clicks to recover.

## 1. Problem framing

The decision is which eligible pages a FlyRank editor should inspect first for a possible content refresh or search-result review. The unit is one client-content pair; the output is a ranked decline-risk score and aggregate action summary. A human checks the actual page and search context before making an editorial decision. A false positive wastes review time; a false negative may leave a decline unnoticed. The model supports prioritization only; it never edits or publishes content automatically.

## 2. Data safety

The analysis uses the gated FlyRank warehouse release's `fact_content_daily_performance` table, aggregated from March 2026 (features) and April 2026 (label). This run used dataset revision `50cbf7c3909d07be4d1b5906b4d09e882e5acbf2`. Only rows with `gsc_data_available IS TRUE` are included. Before choosing the shared March cutoff, I checked `dim_clients.gsc_data_start`: {data['dimension_clients']} dimension clients were listed; {data['dimension_clients_started_by_cutoff']} had a known start on or before March 1, {data['dimension_clients_missing_start_date']} had no start date, and {data['dimension_clients_started_after_cutoff']} started later. Unknown and later starts were excluded. The eligible cohort also requires active client-level GSC coverage in April, positive March clicks, and at least {data['minimum_march_impressions']} March impressions. The daily fact grain was checked for duplicate date/client/content groups: March {data['march_duplicate_daily_grain_groups']}, April {data['april_duplicate_daily_grain_groups']}. Where an eligible client had no matching April content row, April clicks are treated as zero; this is an explicit assumption and a limitation. Pseudonymous client IDs are used only for the grouped split and join; content IDs are used only to align monthly aggregates. Neither ID is a predictor or appears in committed outputs. April outcomes, product scores, trend labels, and `trend_pct` are excluded from features. No row-level export is written.

## 3. Baseline

The baseline is a March-only, within-position CTR opportunity heuristic. On training clients, it computes the median March CTR within position bands (≤3, 4–5, 6–8, 9–12, 13–20, >20); a page receives a larger score when its March CTR falls below its band reference, with the gap weighted by log(1 + March impressions). The rule is transparent and available without the learned model. Both it and logistic regression are scored on the same held-out rows and April-derived labels.

## 4. Model / analysis

The target is 1 when April clicks are at least 20% below March clicks, among pages with positive March clicks. It is retrospective and noisy; it is not a label for “needs a refresh.” The model is a median-imputed, standardized, class-balanced logistic regression (`C=1`, `max_iter=2000`, `random_state=42`). Its exact features are `log1p(march_impressions)`, `log1p(march_clicks)`, `march_ctr_pct`, and `march_avg_position`. All are derived from March only. Client and content IDs are keys only; April clicks are target construction only.

## 5. Evaluation

We used a reproducible 25% client-group holdout with seed {split['seed']}; no client appears in both partitions. The model and baseline use the same {split['test_rows']:,} held-out rows and the same April label. Positive-label prevalence is {_format_pct(split['test_positive_rate'])} in test versus {_format_pct(split['train_positive_rate'])} in training, a substantial client-level distribution shift; the held-out base rate is also the no-discrimination average-precision reference. Precision@10 and Precision@50 are shown beside that rate to avoid interpreting precision without context.

| Method | Average precision | Precision@10 | Precision@50 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic regression | {_format_num(model['average_precision'], 3)} | {_format_pct(model['precision_at_10'])} | {_format_pct(model['precision_at_50'])} | {_format_num(model['roc_auc'], 3)} |
| CTR-gap baseline | {_format_num(baseline['average_precision'], 3)} | {_format_pct(baseline['precision_at_10'])} | {_format_pct(baseline['precision_at_50'])} | {_format_num(baseline['roc_auc'], 3)} |
| No-discrimination reference | {_format_num(dummy['average_precision'], 3)} | {_format_pct(dummy['precision_at_10'])} | {_format_pct(dummy['precision_at_50'])} | {_format_num(dummy['roc_auc'], 3)} |

The target rate varies with the March volume floor, underscoring why low-volume declines need caution:

| Minimum March impressions | Eligible items | Positive-label rate |
|---:|---:|---:|
{sensitivity_rows}

{best_text}

![Precision-recall curves for the held-out client groups](../docs/assets/capstone_precision_recall.png)

*Caption: Model and baseline ranking quality on the same client-held-out labels; the horizontal reference is the test positive rate.*

![Precision at K compared with the positive base rate](../docs/assets/capstone_precision_at_k.png)

*Caption: The base rate is shown because a high top-K precision can otherwise be misleading.*

## 6. Interpretation

{best_text} The held-out positive rate is {_format_pct(split['test_positive_rate'])} versus {_format_pct(split['train_positive_rate'])} in training, showing substantial client heterogeneity in this single split. Precision@50 is above the held-out base rate, but near-chance ROC-AUC and this prevalence shift limit confidence in generalization. Held-out permutation importance (mean change in average precision after shuffling one feature) is:

| Feature | Mean AP decrease ± standard deviation |
|---|---:|
{importance_table}

![Held-out permutation importance](../docs/assets/capstone_permutation_importance.png)

*Caption: Importance is predictive and conditional on this correlated feature set; it is not causal evidence.*

Error review at the 0.5 probability threshold counted {top10['false_positives_at_0_5']:,} false positives and {top10['false_negatives_at_0_5']:,} false negatives on the held-out rows. The probability threshold is descriptive; class-weighted logistic probabilities should not be treated as calibrated action probabilities.

## 7. Recommendation

1. **Start with a human-reviewed top-risk queue.** Review the highest model scores alongside the CTR-gap reasons; do not make the model an automatic editor.
2. **Check seasonality and measurement first.** Confirm that March and April tracking was available, look for demand or SERP changes, and verify that the page remains relevant.
3. **Inspect the search result and page before editing.** Consider intent match, snippet accuracy, and technical context; the data contains no titles, URLs, or raw queries.
4. **Validate any change prospectively.** Compare an intervention with a suitable control or time-aware design before claiming recovery.

The evidence is limited: average precision is only modestly above the no-discrimination reference, ROC-AUC is near chance, and the positive rate differs substantially between training and held-out clients ({_format_pct(split['train_positive_rate'])} versus {_format_pct(split['test_positive_rate'])}) in this single split. The result supports an exploratory review aid, not a reliable general-purpose ranking; confidence that a high-risk score identifies a fixable content problem or predicts future traffic is low.

## 8. Reproducibility

From the repo root, install `requirements.txt`, accept the dataset terms, and place a Hugging Face **read** token in the `HF_TOKEN` environment variable . Never write the token in a notebook or repository file.

```bash
python -m pip install -r requirements.txt
python work/scripts/run_capstone.py --source warehouse
```

The script reads only the March and April 2026 partitions of the daily fact table, uses seed {split['seed']}, and writes aggregate metrics to `work/outputs/capstone_metrics.json`, aggregate action counts to `work/outputs/capstone_action_summary.json`, and charts to `work/figures/` and `docs/assets/`. The [capstone notebook](https://github.com/Safin44/my-flyrank-ml-internship/blob/main/work/notebooks/capstone.ipynb) runs the pipeline. The [repository](https://github.com/Safin44/my-flyrank-ml-internship) contains the code, and the [data dictionary](../docs/data-dictionary.md) describes the release fields.

## 9. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai). The analysis uses only pseudonymized search-performance measurements and publishes aggregate results.
"""
    (root / "work" / "capstone_report.md").write_text(report, encoding="utf-8")


def _render_page(root: Path, metrics: dict[str, object]) -> None:
    data = metrics["data"]
    split = metrics["split"]
    target = metrics["target"]
    model = metrics["model_metrics"]
    baseline = metrics["baseline_metrics"]
    dummy = metrics["base_rate_reference"]
    importance_rows = metrics["permutation_importance"]
    importance_table = "\n".join(
        f"<tr><th scope=\"row\">{html.escape(str(row['feature']))}</th><td>{_format_num(row['mean_ap_decrease'], 4)} ± {_format_num(row['std'], 4)}</td></tr>"
        for row in importance_rows
    )
    sensitivity_table = "\n".join(
        f"<tr><th scope=\"row\">{row['min_march_impressions']:,}</th><td>{row['eligible_rows']:,}</td><td>{_format_pct(row['positive_rate'])}</td></tr>"
        for row in metrics["label_sensitivity"]
    )
    split_text = (
        f"Model AP ({_format_num(model['average_precision'],3)}) is modestly above the baseline ({_format_num(baseline['average_precision'],3)}) and no-discrimination reference ({_format_num(dummy['average_precision'],3)}), but ROC-AUC is near chance ({_format_num(model['roc_auc'],3)}); evidence is weak and directional."
        if model["average_precision"] > baseline["average_precision"]
        else "The baseline matched or exceeded the model on average precision in this held-out split."
    )
    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="A public-safe FlyRank ML capstone predicting a 20% April click decline proxy from March-only search signals, with grouped client validation.">
<title>Can March search signals help prioritize pages for review?</title>
<style>
:root {{ color-scheme:light; --ink:#17313a; --muted:#52666c; --paper:#f4f7f6; --white:#fff; --teal:#176b73; --orange:#d17b30; --line:#dbe5e2; --soft:#e7f1ef; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink); font:16px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:940px; margin:0 auto; padding:26px 20px 60px; }}
header {{ padding:38px 28px; border-radius:22px; color:#fff; background:linear-gradient(135deg,#123840,#176b73 72%,#328079); }}
.eyebrow {{ text-transform:uppercase; letter-spacing:.12em; font-size:.78rem; font-weight:700; opacity:.85; }} h1 {{ font-size:clamp(2rem,6vw,3.45rem); line-height:1.1; margin:.6rem 0 1rem; max-width:800px; }}
header p {{ max-width:760px; color:#e4f0ed; margin:.35rem 0; }} .pills {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:18px; }} .pill {{ border:1px solid #ffffff55; background:#ffffff14; border-radius:999px; padding:5px 11px; font-size:.85rem; }}
section {{ background:var(--white); border:1px solid var(--line); border-radius:18px; padding:24px; margin:18px 0; box-shadow:0 4px 18px #203c3510; }} h2 {{ font-size:1.38rem; line-height:1.3; margin:0 0 12px; }} p {{ margin:10px 0; }} .lead {{ font-size:1.05rem; }} .metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:11px; margin:20px 0 4px; }} .metric {{ background:var(--soft); border-radius:13px; padding:14px; }} .metric strong {{ display:block; font-size:1.42rem; color:var(--teal); line-height:1.2; }} .metric span {{ font-size:.82rem; color:var(--muted); }}
figure {{ margin:18px 0 8px; }} figure img {{ width:100%; height:auto; border-radius:12px; border:1px solid var(--line); }} figcaption {{ margin-top:8px; color:var(--muted); font-size:.9rem; }} table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:.92rem; }} th,td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); }} thead th {{ background:#eff5f2; }} .note {{ border-left:4px solid var(--orange); background:#fff7ed; padding:12px 15px; border-radius:0 10px 10px 0; color:#4a443b; }} li {{ margin:.5rem 0; }} a {{ color:#12636a; text-underline-offset:3px; }} pre {{ overflow:auto; background:#f1f5f3; padding:14px; border-radius:10px; }} footer {{ padding:12px 8px; color:var(--muted); font-size:.9rem; }}
@media(max-width:720px) {{ main {{ padding:14px 12px 40px; }} header {{ padding:27px 20px; }} section {{ padding:19px 17px; }} .metrics {{ grid-template-columns:repeat(2,1fr); }} table {{ font-size:.82rem; }} th,td {{ padding:7px 6px; }} }} @media(max-width:400px) {{ .metrics {{ grid-template-columns:1fr; }} }}
</style></head>
<body><main>
<header><div class="eyebrow">FlyRank ML Internship · Machine Learning Capstone · CTR / Engagement Opportunity Scoring</div>
<h1>Can March search signals help prioritize pages for review?</h1>
<p>A client-held-out study of a simple decline-risk model and a transparent CTR baseline on the gated, anonymized warehouse release. It is a review aid, not a prediction of Google’s algorithm.</p>
<div class="pills"><span class="pill">March 2026 features → April 2026 proxy</span><span class="pill">{data['eligible_rows']:,} eligible items</span><span class="pill">{data['eligible_clients']} clients</span><span class="pill">No client split overlap</span></div></header>

<section id="abstract"><h2>Title + Abstract</h2>
<p class="lead">This capstone asks whether a small, interpretable model can rank content items whose clicks fall in the month after a March 2026 decision snapshot. It uses the FlyRank anonymized warehouse daily-performance table, aggregating March features and April outcomes for {data['eligible_rows']:,} eligible items from {data['eligible_clients']} clients. A class-balanced logistic regression uses March-only impressions, clicks, click-through rate, and average position, and is compared with a transparent within-position CTR-shortfall rule on a client-held-out test set. On the held-out clients, logistic regression achieves average precision {_format_num(model['average_precision'],3)} versus {_format_num(baseline['average_precision'],3)} for the baseline, against a held-out positive-label base rate of {_format_pct(split['test_positive_rate'])}; Precision@50 is {_format_pct(model['precision_at_50'])} versus {_format_pct(baseline['precision_at_50'])}. ROC-AUC is {_format_num(model['roc_auc'],3)}, so this is weak, directional review evidence against a noisy proxy—not a forecast of Google’s algorithm or evidence that editing a page causes clicks to recover.</p>
<div class="metrics"><div class="metric"><strong>{data['eligible_rows']:,}</strong><span>eligible client-content items</span></div><div class="metric"><strong>{data['eligible_clients']}</strong><span>clients with GSC coverage in both months</span></div><div class="metric"><strong>{_format_num(model['average_precision'],3)}</strong><span>model average precision</span></div><div class="metric"><strong>{_format_pct(split['test_positive_rate'])}</strong><span>held-out positive-label rate</span></div></div></section>

<section id="problem"><h2>Introduction / Problem statement</h2><p>Editors have limited time. This study asks whether observable March search signals can help order pages for review when April clicks are at least 20% below March clicks. The output is a ranked review list: a human checks seasonality, measurement coverage, SERP context, and page relevance before taking action. A false positive costs review time; a false negative may leave a decline unchecked. The model supports prioritization only and never edits or publishes content automatically.</p></section>

<section id="data"><h2>Data</h2><p>The source is the gated <a href="https://huggingface.co/datasets/FlyRank/internship-warehouse">FlyRank/internship-warehouse</a> release, build v20260703. It contains a 78,835,655-row daily fact table overall; this analysis reads only the March 2026 feature partition and April 2026 label partition, then aggregates to client-content-month. It filters to rows where <code>gsc_data_available IS TRUE</code>. Before using the shared March cutoff, I checked <code>dim_clients.gsc_data_start</code> and excluded clients with unknown starts or a start after 2026-03-01. The eligible cohort also requires March clicks &gt; 0, at least {data['minimum_march_impressions']} March impressions, and active April GSC coverage.</p><p>Inputs are March Google Search Console impressions, clicks, derived CTR, and average position. The label is April clicks ≤ 80% of March clicks. If a client has April GSC coverage but an individual content item has no April fact row, that item is treated as having zero April clicks. The daily grain check found {data['march_duplicate_daily_grain_groups']} duplicate March and {data['april_duplicate_daily_grain_groups']} duplicate April date/client/content groups. Client and content hashes are used only for joining and the client-grouped split; no IDs, URLs, titles, domains, or queries appear in this paper or its committed outputs. Trend fields and April columns are never model features. Read the <a href="../docs/data-dictionary.md">data dictionary</a> and <a href="../DATA_USE.md">data-use rules</a>.</p><p class="note">The label is retrospective and noisy: a 20% click decline is a directional review proxy, not proof that a page needs a refresh.</p></section>

<section id="method"><h2>Methodology</h2><p>The eligible client-content items are split by client using <code>GroupShuffleSplit</code>: {split['train_clients']} clients for training and {split['test_clients']} held out, with seed {split['seed']} and zero overlap. The model is median imputation, standardization, and class-balanced logistic regression (<code>C=1</code>, <code>max_iter=2000</code>). Features are <code>log1p(March impressions)</code>, <code>log1p(March clicks)</code>, March CTR, and March mean position. March data is the only feature window; April is used only to define the evaluation label.</p><p>The baseline is a March-only CTR-opportunity heuristic: within position bands, subtract observed CTR from the training-client median CTR and weight a positive gap by <code>log(1 + impressions)</code>. Both rankings are evaluated on the same held-out rows and label. Identifiers are not features. <a href="../work/scripts/run_capstone.py">View the reproducible script</a>.</p></section>

<section id="results"><h2>Results: model versus baseline</h2><p>Test positive prevalence is {_format_pct(split['test_positive_rate'])}, versus {_format_pct(split['train_positive_rate'])} in training; this client-level shift limits generalization from one split. Precision@K is shown beside the test base rate. {html.escape(split_text)}</p>
<table><thead><tr><th>Method</th><th>Average precision</th><th>Precision@10</th><th>Precision@50</th><th>ROC-AUC</th></tr></thead><tbody>
<tr><th scope="row">Logistic regression</th><td>{_format_num(model['average_precision'],3)}</td><td>{_format_pct(model['precision_at_10'])}</td><td>{_format_pct(model['precision_at_50'])}</td><td>{_format_num(model['roc_auc'],3)}</td></tr>
<tr><th scope="row">CTR-gap baseline</th><td>{_format_num(baseline['average_precision'],3)}</td><td>{_format_pct(baseline['precision_at_10'])}</td><td>{_format_pct(baseline['precision_at_50'])}</td><td>{_format_num(baseline['roc_auc'],3)}</td></tr>
<tr><th scope="row">No-discrimination reference</th><td>{_format_num(dummy['average_precision'],3)}</td><td>{_format_pct(dummy['precision_at_10'])}</td><td>{_format_pct(dummy['precision_at_50'])}</td><td>{_format_num(dummy['roc_auc'],3)}</td></tr>
</tbody></table>
<h3>Label-rate sensitivity to the March volume floor</h3><p>The decline-proxy rate changes as low-impression items are excluded; the main model uses the 100-impression floor.</p><table><thead><tr><th>Minimum March impressions</th><th>Eligible items</th><th>Positive-label rate</th></tr></thead><tbody>{sensitivity_table}</tbody></table>
<figure><img src="assets/capstone_precision_recall.png" alt="Precision-recall curves for logistic regression and the CTR baseline on the same held-out client groups"><figcaption>Held-out precision-recall curves; the dashed reference is the test positive-label rate. Average precision measures ranking quality against this proxy, not causal impact.</figcaption></figure>
<figure><img src="assets/capstone_precision_at_k.png" alt="Bar chart comparing model and baseline precision at 10 and 50 against the test base rate"><figcaption>Precision among the top-ranked held-out items, shown beside the proxy base rate.</figcaption></figure>
<h3>Held-out permutation importance</h3><table><thead><tr><th>Feature</th><th>Mean decrease in AP ± standard deviation</th></tr></thead><tbody>{importance_table}</tbody></table>
<figure><img src="assets/capstone_permutation_importance.png" alt="Horizontal bar chart of held-out permutation importance for March search features"><figcaption>Importance is conditional on this correlated feature set; it is not causal evidence.</figcaption></figure></section>

<section id="limitations"><h2>Limitations &amp; honest framing</h2><ul><li>The 20% decline threshold is a proxy; it can reflect noise, seasonality, or measurement changes rather than an actionable content problem.</li><li>March-to-April is one adjacent month pair; it does not establish generalization to other seasons, clients, or future cohorts/periods.</li><li>Clients are held out, but items from each client still share broader market and measurement context; the test set is not a randomized experiment.</li><li>Average position and CTR are aggregates, and page-level April rows missing for an active client are treated as zero clicks.</li><li>Class-balanced logistic probabilities are not calibrated action probabilities. Feature importance is predictive, not causal.</li><li>This analysis does not predict Google’s ranking algorithm or prove that a refresh will recover clicks.</li></ul></section>

<section id="recommendations"><h2>Ranked recommendations</h2><ol><li><strong>Review the highest-scored items first.</strong> Use the score to prioritize attention; the public artifacts contain only aggregate results, not item-level IDs.</li><li><strong>Check measurement and seasonality.</strong> Verify April GSC availability and examine demand changes before interpreting a decline as content decay.</li><li><strong>Inspect snippet and intent fit.</strong> A human should compare search intent, SERP context, page quality, and technical status before making an edit.</li><li><strong>Test changes prospectively.</strong> Record the change date and use a suitable control or time-aware evaluation before claiming an uplift.</li></ol><p>Operational guardrail: review queue only; no automatic publishing, pruning, or causal claims.</p></section>

<section id="reproducibility"><h2>Reproducibility</h2><p>Install the repo requirements, accept the Hugging Face dataset terms, and provide a READ token through <code>HF_TOKEN</code> . The token is never hardcoded in the notebook or repository.</p><pre><code>python -m pip install -r requirements.txt
python work/scripts/run_capstone.py --source warehouse</code></pre><p>Seed {split['seed']}; {100*split['test_size']:.0f}% grouped client holdout. The script reads only March and April partitions and saves aggregate results to <a href="../work/outputs/capstone_metrics.json"><code>work/outputs/capstone_metrics.json</code></a>, with aggregate action counts in <a href="../work/outputs/capstone_action_summary.json"><code>work/outputs/capstone_action_summary.json</code></a>. See the <a href="https://github.com/Safin44/my-flyrank-ml-internship/blob/main/work/notebooks/capstone.ipynb">capstone notebook</a>, <a href="https://github.com/Safin44/my-flyrank-ml-internship">repository</a>, and <a href="../work/capstone_report.md">full capstone report</a>.</p></section>

<p class="note">Dataset reads are pinned to immutable Hugging Face revision <a href="https://huggingface.co/datasets/FlyRank/internship-warehouse/tree/{DATASET_REVISION}"><code>{DATASET_REVISION}</code></a>. A future update to the dataset will not silently change this run's source.</p>

<section id="acknowledgments"><h2>Acknowledgments &amp; data credit</h2><p>Built on the <a href="https://flyrank.ai">FlyRank ML Internship dataset</a>. The analysis publishes only aggregate, pseudonymized data results.</p></section>
<footer>Author: Imam Hossain Safin · Educational decision-support analysis · No row-level records or identifiers published.</footer>
</main></body></html>
"""
    # Pages serves /docs as the site root, so link to files elsewhere in the repo via GitHub.
    page_links = {
        'href="../docs/data-dictionary.md"': 'href="https://github.com/Safin44/my-flyrank-ml-internship/blob/main/docs/data-dictionary.md"',
        'href="../DATA_USE.md"': 'href="https://github.com/Safin44/my-flyrank-ml-internship/blob/main/DATA_USE.md"',
        'href="../work/scripts/run_capstone.py"': 'href="https://github.com/Safin44/my-flyrank-ml-internship/blob/main/work/scripts/run_capstone.py"',
        'href="../work/outputs/capstone_metrics.json"': 'href="https://github.com/Safin44/my-flyrank-ml-internship/blob/main/work/outputs/capstone_metrics.json"',
        'href="../work/outputs/capstone_action_summary.json"': 'href="https://github.com/Safin44/my-flyrank-ml-internship/blob/main/work/outputs/capstone_action_summary.json"',
        'href="../work/capstone_report.md"': 'href="https://github.com/Safin44/my-flyrank-ml-internship/blob/main/work/capstone_report.md"',
    }
    for old_link, new_link in page_links.items():
        page = page.replace(old_link, new_link)
    (root / "docs" / "index.html").write_text(page, encoding="utf-8")


def run(root: Path = ROOT) -> dict[str, object]:
    root = root.resolve()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required. Accept access to FlyRank/internship-warehouse and set a READ token as an environment variable ."
        )
    print("Querying only the March 2026 feature and April 2026 label partitions…")
    candidates = _query_warehouse(token)
    if candidates.empty:
        raise RuntimeError("The March/April query returned no eligible rows; verify dataset access and field coverage.")
    qc_columns = [
        "dimension_clients", "clients_started_by_cutoff", "clients_missing_start_date",
        "clients_started_after_cutoff", "march_duplicate_daily_grain_groups",
        "april_duplicate_daily_grain_groups",
    ]
    warehouse_qc = {name: int(candidates.iloc[0][name]) for name in qc_columns}
    if warehouse_qc["march_duplicate_daily_grain_groups"] or warehouse_qc["april_duplicate_daily_grain_groups"]:
        raise ValueError("Daily fact grain is not unique by date/client/content; stop and resolve duplicates before aggregating.")
    candidates = candidates.drop(columns=qc_columns).sort_values(
        ["client_id", "content_id"], kind="mergesort"
    ).reset_index(drop=True)
    candidates["decline_label"] = (
        pd.to_numeric(candidates["april_clicks"], errors="coerce").fillna(0)
        <= 0.8 * pd.to_numeric(candidates["march_clicks"], errors="coerce")
    ).astype(int)
    sensitivity = []
    for impression_floor in [1, 100, 500]:
        subset = candidates[candidates["march_impressions"] >= impression_floor]
        sensitivity.append({
            "min_march_impressions": int(impression_floor),
            "eligible_rows": int(len(subset)),
            "positive_rate": _safe_float(subset["decline_label"].mean()) if len(subset) else 0.0,
        })
    candidate_rows_before_volume_floor = int(len(candidates))
    data = candidates[candidates["march_impressions"] >= MIN_MARCH_IMPRESSIONS].copy()
    if data.empty:
        raise ValueError(f"No eligible rows remain after the {MIN_MARCH_IMPRESSIONS}-impression floor.")
    if data["client_id"].nunique() < 5:
        raise ValueError("Need at least five eligible clients for a meaningful client-grouped holdout.")

    data["march_avg_position"] = pd.to_numeric(data["march_avg_position"], errors="coerce")
    data.loc[data["march_avg_position"] <= 0, "march_avg_position"] = np.nan
    data["position_band"] = pd.cut(
        data["march_avg_position"], bins=POSITION_BINS, labels=POSITION_LABELS, right=True
    ).astype("object").fillna("unknown")
    data["decline_label"] = (
        pd.to_numeric(data["april_clicks"], errors="coerce").fillna(0)
        <= 0.8 * pd.to_numeric(data["march_clicks"], errors="coerce")
    ).astype(int)
    data["log1p_march_impressions"] = np.log1p(pd.to_numeric(data["march_impressions"], errors="coerce"))
    data["log1p_march_clicks"] = np.log1p(pd.to_numeric(data["march_clicks"], errors="coerce"))
    feature_names = [
        "log1p_march_impressions",
        "log1p_march_clicks",
        "march_ctr_pct",
        "march_avg_position",
    ]
    x = data[feature_names].replace([np.inf, -np.inf], np.nan)
    y = data["decline_label"].astype(int).to_numpy()
    groups = data["client_id"].astype(str).to_numpy()
    if np.unique(y).size < 2:
        raise ValueError("The target has one class only; stop rather than report misleading metrics.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_clients = set(train["client_id"].astype(str))
    test_clients = set(test["client_id"].astype(str))
    overlap = train_clients & test_clients
    if overlap:
        raise AssertionError("Client overlap found in grouped holdout.")
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        raise ValueError("Grouped split has one target class in train or test; adjust the documented seed/split, do not use row-level fallback.")

    model = _build_model()
    model.fit(x_train, y_train)
    model_score = model.predict_proba(x_test)[:, 1]

    # Training-client CTR references prevent baseline information from leaking across the client split.
    train_ctr = train["march_ctr_pct"].astype(float)
    global_train_ctr = float(train_ctr.median())
    position_references = train.groupby("position_band", dropna=False)["march_ctr_pct"].median().to_dict()
    expected_test_ctr = test["position_band"].map(position_references).fillna(global_train_ctr).to_numpy(dtype=float)
    baseline_score = np.maximum(expected_test_ctr - test["march_ctr_pct"].to_numpy(dtype=float), 0.0) * np.log1p(test["march_impressions"].to_numpy(dtype=float))

    model_metrics = _score_metrics(y_test, model_score)
    baseline_metrics = _score_metrics(y_test, baseline_score)
    prevalence = float(np.mean(y_test))
    base_rate_reference = {
        "average_precision": prevalence,
        "precision_at_10": prevalence,
        "precision_at_50": prevalence,
        "roc_auc": 0.5,
        "description": "Expected random ranking / no-discrimination reference using the held-out positive prevalence.",
    }

    binary = (model_score >= 0.5).astype(int)
    false_positives = int(((binary == 1) & (y_test == 0)).sum())
    false_negatives = int(((binary == 0) & (y_test == 1)).sum())
    top_order = np.argsort(-model_score, kind="stable")
    top10_idx = top_order[:min(10, len(top_order))]
    top50_idx = top_order[:min(50, len(top_order))]
    top_risk_aggregates = {
        "top_10_decline_proxy_count": int(y_test[top10_idx].sum()),
        "top_10_items": int(len(top10_idx)),
        "top_50_decline_proxy_count": int(y_test[top50_idx].sum()),
        "top_50_items": int(len(top50_idx)),
        "false_positives_at_0_5": false_positives,
        "false_negatives_at_0_5": false_negatives,
        "threshold": 0.5,
        "note": "Aggregate counts only; no row-level IDs or predictions are exported.",
    }

    permutation = permutation_importance(
        model, x_test, y_test, scoring="average_precision", n_repeats=5,
        random_state=SEED, n_jobs=-1,
    )
    importance = [
        {"feature": str(name), "mean_ap_decrease": _safe_float(mean), "std": _safe_float(std)}
        for name, mean, std in zip(feature_names, permutation.importances_mean, permutation.importances_std)
    ]
    importance.sort(key=lambda row: row["mean_ap_decrease"], reverse=True)

    # The public action summary is drawn from the held-out top-risk set, never an in-sample fit.
    heldout_ranked = test.copy()
    heldout_ranked["model_score"] = model_score
    heldout_ranked["decline_label"] = y_test
    heldout_top = heldout_ranked.sort_values("model_score", ascending=False, kind="mergesort").head(min(100, len(test)))
    action_summary = {
        "ranked_items_summarized": int(len(heldout_top)),
        "decline_proxy_items": int(heldout_top["decline_label"].sum()),
        "positive_proxy_share": _safe_float(heldout_top["decline_label"].mean()) if len(heldout_top) else 0.0,
        "position_band_counts": {
            str(key): int(value)
            for key, value in heldout_top["position_band"].value_counts().reindex(POSITION_LABELS + ["unknown"], fill_value=0).items()
        },
        "median_march_impressions": _safe_float(heldout_top["march_impressions"].median()) if len(heldout_top) else 0.0,
        "source": "client-held-out top-risk items only; descriptive, not an independent validation metric",
        "output_policy": "Aggregate-only; no page/client IDs or row-level scores are persisted.",
    }

    metrics: dict[str, object] = {
        "project": "FlyRank ML Capstone — Search Decline Risk and CTR Review",
        "source": {
            "dataset": "FlyRank/internship-warehouse",
            "release_build": "flyrank_pseudonymized_warehouse_release_v20260703",
            "dataset_revision": DATASET_REVISION,
            "table": "fact_content_daily_performance",
            "total_release_rows_reference": 78835655,
            "feature_partition": FEATURE_MONTH,
            "label_partition": LABEL_MONTH,
            "partition_rows_scanned_in_analysis": "March and April 2026 partitions only",
        },
        "data": {
            "eligible_rows": int(len(data)),
            "eligible_clients": int(data["client_id"].nunique()),
            "clients_active_in_both_months": int(data["client_id"].nunique()),
            "candidate_rows_before_volume_floor": candidate_rows_before_volume_floor,
            "rows_excluded_by_volume_floor": int(candidate_rows_before_volume_floor - len(data)),
            "march_rows_with_april_page_row": int(data["has_april_page_row"].sum()),
            "march_rows_without_april_page_row_treated_as_zero": int((~data["has_april_page_row"]).sum()),
            "minimum_march_impressions": MIN_MARCH_IMPRESSIONS,
            "minimum_march_clicks": 1,
            "client_history_start_filter": "gsc_data_start is not null and <= 2026-03-01",
            "dimension_clients": warehouse_qc["dimension_clients"],
            "dimension_clients_started_by_cutoff": warehouse_qc["clients_started_by_cutoff"],
            "dimension_clients_missing_start_date": warehouse_qc["clients_missing_start_date"],
            "dimension_clients_started_after_cutoff": warehouse_qc["clients_started_after_cutoff"],
            "march_duplicate_daily_grain_groups": warehouse_qc["march_duplicate_daily_grain_groups"],
            "april_duplicate_daily_grain_groups": warehouse_qc["april_duplicate_daily_grain_groups"],
        },
        "target": {
            "definition": f"1 if April clicks <= 0.8 * March clicks, among items with March impressions >= {MIN_MARCH_IMPRESSIONS} and positive March clicks; absent April item rows are treated as 0 clicks only for clients active in April.",
            "positive_rows": int(y.sum()),
            "positive_rate": _safe_float(y.mean()),
            "majority_class_rate": _safe_float(max(y.mean(), 1 - y.mean())),
        },
        "features": {
            "model_inputs": feature_names,
            "feature_window": "March 2026 only",
            "excluded_from_features": ["April clicks", "trend fields", "client_hash_id", "content_hash_id", "product decisions", "GA4 metrics"],
            "position_bands_for_baseline": POSITION_LABELS + ["unknown"],
        },
        "split": {
            "strategy": "GroupShuffleSplit by client_hash_id",
            "seed": SEED,
            "test_size": TEST_SIZE,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_clients": int(len(train_clients)),
            "test_clients": int(len(test_clients)),
            "client_overlap": int(len(overlap)),
            "train_positive_rate": _safe_float(y_train.mean()),
            "test_positive_rate": prevalence,
        },
        "model": {
            "name": "class-balanced LogisticRegression",
            "imputation": "median",
            "scaling": "StandardScaler",
            "C": 1.0,
            "max_iter": 2000,
            "random_state": SEED,
        },
        "model_metrics": model_metrics,
        "baseline": {
            "name": "March-only within-position CTR-shortfall heuristic",
            "description": "Training-client median CTR by March average-position band; positive CTR gap weighted by log1p(March impressions).",
            "expected_ctr_source": "training clients only",
            "training_global_median_ctr_pct": _safe_float(global_train_ctr),
        },
        "baseline_metrics": baseline_metrics,
        "base_rate_reference": base_rate_reference,
        "permutation_importance": importance,
        "top_risk_aggregates": top_risk_aggregates,
        "label_sensitivity": sensitivity,
        "action_summary": action_summary,
        "outputs": {
            "metrics": "work/outputs/capstone_metrics.json",
            "aggregate_actions": "work/outputs/capstone_action_summary.json",
            "figures": ["work/figures/" + name for name in ["capstone_precision_recall.png", "capstone_precision_at_k.png", "capstone_permutation_importance.png"]],
            "row_level_exports_written": False,
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "duckdb": duckdb.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    output_dir = root / "work" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "capstone_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "capstone_action_summary.json").write_text(json.dumps(action_summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_figures(root, y_test, model_score, baseline_score, feature_names, importance, prevalence)
    _render_report(root, metrics)
    _render_page(root, metrics)

    print("Warehouse capstone complete (March features, April label).")
    print(f"Eligible items: {len(data):,}; train/test items: {len(train):,}/{len(test):,}.")
    print(f"Grouped split: {len(train_clients)} train clients, {len(test_clients)} test clients; overlap={len(overlap)}.")
    print(f"Positive-label base rate: {prevalence:.3%}; majority-class rate: {max(prevalence, 1-prevalence):.3%}.")
    print(f"Model AP={model_metrics['average_precision']:.4f}, P@50={model_metrics['precision_at_50']:.3f}, ROC-AUC={model_metrics['roc_auc']:.4f}.")
    print(f"Baseline AP={baseline_metrics['average_precision']:.4f}, P@50={baseline_metrics['precision_at_50']:.3f}, ROC-AUC={baseline_metrics['roc_auc']:.4f}.")
    print("Only aggregate metrics, action counts, and charts were written; no IDs or row-level predictions were exported.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FlyRank warehouse capstone.")
    parser.add_argument("--source", choices=["warehouse"], default="warehouse", help="The capstone uses the gated warehouse release.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root (defaults to the detected root).")
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
