# ML-CAP-01 analysis protocol

**Lane:** CTR / Engagement Opportunity Scoring  
**Repository:** https://github.com/Safin44/my-flyrank-ml-internship

## Research question

Can March-only search-performance signals rank content items that experience a substantial April click decline for human editorial review?

The unit is a client-content pair. The output is a ranked review queue, not an automatic content change. A human editor must inspect the actual page and its search context before choosing an action. A missed decline may delay a useful review; a false positive consumes review time.

## Verified source and windows

This run uses the [FlyRank ML Internship warehouse](https://huggingface.co/datasets/FlyRank/internship-warehouse), release v20260703, repository revision `50cbf7c3909d07be4d1b5906b4d09e882e5acbf2`. The March daily fact partition contains 9,841,378 rows covering March 1–31, 2026; April contains 10,424,730 rows covering April 1–30, 2026. March supplies the features and April supplies the retrospective outcome.

The cohort requires a known client GSC start on or before March 1, GSC-available daily rows, active client-level GSC coverage in April, positive March clicks, and at least 100 March impressions. The outcome is a 20% or greater month-over-month decline in clicks. Missing April content rows for an otherwise active client are treated as zero April clicks; this is an explicit assumption, not a verified absence of traffic.

## Comparison and validation

- **Transparent baseline:** training-set median March CTR by position band; rank below-band CTR shortfall weighted by `log1p(March impressions)`. Fit the reference medians on training clients only.
- **Interpretable model:** median imputation, standardization, and class-balanced logistic regression (`C=1`, `max_iter=2000`, `random_state=42`) using only `log1p(March impressions)`, `log1p(March clicks)`, March CTR, and March average position.
- **Validation:** 25% client-group holdout, seed 42. No client appears in both partitions. Both methods use the same held-out rows and outcome labels.
- **Reporting:** average precision, Precision@10, Precision@50, and ROC-AUC; held-out prevalence is shown beside precision metrics. See [`capstone_report.md`](capstone_report.md) and [`outputs/capstone_metrics.json`](outputs/capstone_metrics.json) for the measured results and caveats.

The held-out ROC-AUC is near chance. The analysis is exploratory review evidence against a noisy proxy, not a reliable general-purpose forecast or evidence that editing a page causes recovery.

## Safety and reproducibility

Client/content identifiers are used only to align monthly aggregates and keep clients separated across validation partitions. They are not model features and do not appear in public outputs. April-derived fields, trend labels, and other post-cutoff information are excluded from predictors.

No client names, domains, URLs, private queries, credentials, raw exports, or row-level records are included in the capstone outputs. The read token is supplied only through the runtime `HF_TOKEN` environment/Colab secret and is never committed, printed, or stored in the notebook.

The reusable pipeline is [`scripts/run_capstone.py`](scripts/run_capstone.py). It regenerates aggregate metrics, action counts, figures, the report, and the static paper page. The [capstone notebook](notebooks/capstone.ipynb) was executed top to bottom for this run with all 13 code cells completing without errors.

The public paper must contain the five-sentence abstract, problem statement, data, methodology, model-versus-baseline results, limitations, ranked recommendations, reproducibility links, and a data credit linking to [FlyRank](https://flyrank.ai). Portal submission remains the account holder's responsibility; this workflow does not submit to FlyRank.
