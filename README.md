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

