"""
Synthetic Customer Metadata Generator
======================================
Generates a CRM-like table of customer attributes that are NOT observable from
transaction history alone. This simulates what a real business CRM system
(Salesforce, HubSpot) would hold about each wholesale account.

WHY GENERATE RATHER THAN FIND A REAL DATASET?
- No publicly available CRM dataset maps onto these exact CustomerIDs
- The anchor dataset has no firmographic information
- Generating it lets us explicitly control the correlation structure
  (e.g. large companies → higher credit limits) so it is realistic for ML

HOW WE MAKE IT REALISTIC (not just random noise):
- We first compute each customer's actual transaction stats (frequency, revenue,
  country) from the anchor dataset
- We use those stats to INFORM the synthetic attributes so the correlations
  make business sense:
    high revenue customer → more likely "Large" company tier
    UK customer          → UK region
    1-order customer     → less likely to have a named account manager
- This means the synthetic metadata adds real signal to the ML model, not just
  noise

OUTPUT:
    project/data/bronze_customer_metadata_synthetic.csv

COLUMNS:
    customer_id         - matches CustomerID in anchor dataset
    company_size        - Small / Medium / Large (revenue-informed)
    vertical            - industry segment of the buyer
    onboard_channel     - how they were acquired
    region              - geographic region (derived from transaction country)
    account_manager     - named AM (only Medium/Large; Small = self-serve)
    credit_limit_gbp    - trade credit limit (company_size-informed)
    payment_terms_days  - Net 30 / 60 / 90 (company_size-informed)
    years_in_business   - years the company has been trading (random, seeded)
    is_vip              - flag for top-decile revenue customers
"""

import pandas as pd
import numpy as np
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Load anchor dataset and compute customer-level stats
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, "data.csv")
out_dir    = os.path.join(script_dir, "data")
out_path   = os.path.join(out_dir, "bronze_customer_metadata_synthetic.csv")

os.makedirs(out_dir, exist_ok=True)

print("Loading transaction data...")
df = pd.read_csv(data_path, encoding="latin-1")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Keep only rows with a CustomerID (anonymous rows have no customer to label)
df = df[df["CustomerID"].notna()].copy()
df["CustomerID"] = df["CustomerID"].astype(int)

# Derive basic stats per customer from actual transactions
df_clean = df[
    ~df["InvoiceNo"].astype(str).str.startswith("C") &
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
].copy()
df_clean["Revenue"] = df_clean["Quantity"] * df_clean["UnitPrice"]

# Primary country per customer (mode)
country_mode = (
    df_clean.groupby("CustomerID")["Country"]
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
    .rename(columns={"Country": "primary_country"})
)

cust_stats = df_clean.groupby("CustomerID").agg(
    total_revenue  = ("Revenue",    "sum"),
    order_count    = ("InvoiceNo",  "nunique"),
    first_order    = ("InvoiceDate","min"),
).reset_index()
cust_stats = cust_stats.merge(country_mode, on="CustomerID", how="left")

n = len(cust_stats)
print(f"Generating metadata for {n:,} customers...")

# ---------------------------------------------------------------------------
# 2. company_size  —  based on revenue percentile
#    Bottom 60% → Small, next 30% → Medium, top 10% → Large
# ---------------------------------------------------------------------------
rev_60 = cust_stats["total_revenue"].quantile(0.60)
rev_90 = cust_stats["total_revenue"].quantile(0.90)

def assign_size(rev):
    if rev >= rev_90:
        return "Large"
    elif rev >= rev_60:
        return "Medium"
    else:
        return "Small"

cust_stats["company_size"] = cust_stats["total_revenue"].apply(assign_size)

# ---------------------------------------------------------------------------
# 3. vertical  —  industry of the wholesale buyer
#    Probabilities loosely reflect a UK gift/homeware wholesale market.
#    No strong correlation needed here — verticals cut across all sizes.
# ---------------------------------------------------------------------------
verticals = ["Gift Shop", "Homeware Retail", "Online Retailer",
             "Department Store", "Market Trader", "Florist", "Other"]
v_probs   = [0.28,         0.22,              0.18,
             0.08,           0.12,            0.06,     0.06]

cust_stats["vertical"] = np.random.choice(verticals, size=n, p=v_probs)

# ---------------------------------------------------------------------------
# 4. onboard_channel  —  correlated with company_size
#    Large accounts tend to come via Sales/Trade Show; Small via online/cold.
# ---------------------------------------------------------------------------
channels = ["Trade Show", "Direct Sales", "Online Enquiry", "Referral", "Cold Outreach"]

channel_probs = {
    "Large":  [0.35, 0.35, 0.10, 0.15, 0.05],
    "Medium": [0.25, 0.25, 0.25, 0.15, 0.10],
    "Small":  [0.10, 0.10, 0.40, 0.15, 0.25],
}

cust_stats["onboard_channel"] = cust_stats["company_size"].apply(
    lambda s: np.random.choice(channels, p=channel_probs[s])
)

# ---------------------------------------------------------------------------
# 5. region  —  derived from actual primary_country in transactions
#    UK, Western Europe, Rest of World
# ---------------------------------------------------------------------------
eu_countries = {
    "Germany", "France", "Netherlands", "Belgium", "Spain",
    "Portugal", "Switzerland", "Austria", "Denmark", "Sweden",
    "Norway", "Finland", "Italy", "EIRE", "Channel Islands", "Cyprus",
    "Poland", "Czech Republic", "Lithuania", "Greece", "Malta",
}

def assign_region(country):
    if country == "United Kingdom":
        return "UK"
    elif country in eu_countries:
        return "Western Europe"
    else:
        return "Rest of World"

cust_stats["region"] = cust_stats["primary_country"].apply(assign_region)

# ---------------------------------------------------------------------------
# 6. account_manager  —  only Medium and Large customers get a named AM;
#    Small customers are self-serve (AM = None)
# ---------------------------------------------------------------------------
am_pool = [
    "Sarah Bennett", "James Thornton", "Priya Kapoor",
    "Oliver Walsh",  "Emily Hartley",  "Mohammed Al-Rashid",
]

def assign_am(size, cust_id):
    if size == "Small":
        return "Self-Serve"
    # deterministic assignment based on CustomerID so it's consistent
    return am_pool[cust_id % len(am_pool)]

cust_stats["account_manager"] = cust_stats.apply(
    lambda row: assign_am(row["company_size"], int(row["CustomerID"])), axis=1
)

# ---------------------------------------------------------------------------
# 7. credit_limit_gbp  —  correlated with company_size, with noise
# ---------------------------------------------------------------------------
credit_base = {"Small": 500, "Medium": 2000, "Large": 10000}
credit_sd   = {"Small": 200, "Medium":  800, "Large":  3000}

cust_stats["credit_limit_gbp"] = cust_stats["company_size"].apply(
    lambda s: max(250, int(np.random.normal(credit_base[s], credit_sd[s]) // 250 * 250))
)

# For VIPs, credit limit is at least twice the base
rev_90_val = cust_stats["total_revenue"].quantile(0.90)
cust_stats.loc[cust_stats["total_revenue"] >= rev_90_val, "credit_limit_gbp"] = \
    cust_stats.loc[cust_stats["total_revenue"] >= rev_90_val, "credit_limit_gbp"].apply(
        lambda x: max(x, credit_base["Large"])
    )

# ---------------------------------------------------------------------------
# 8. payment_terms_days  —  Net 30 / 60 / 90 correlated with company_size
# ---------------------------------------------------------------------------
terms_options = [30, 60, 90]
terms_probs   = {
    "Small":  [0.80, 0.15, 0.05],
    "Medium": [0.50, 0.40, 0.10],
    "Large":  [0.20, 0.50, 0.30],
}

cust_stats["payment_terms_days"] = cust_stats["company_size"].apply(
    lambda s: np.random.choice(terms_options, p=terms_probs[s])
)

# ---------------------------------------------------------------------------
# 9. years_in_business  —  random 1–35, weakly correlated with size
# ---------------------------------------------------------------------------
years_mean = {"Small": 5, "Medium": 10, "Large": 18}
years_sd   = {"Small": 3, "Medium":  5, "Large":  8}

cust_stats["years_in_business"] = cust_stats["company_size"].apply(
    lambda s: max(1, int(round(np.random.normal(years_mean[s], years_sd[s]))))
)

# ---------------------------------------------------------------------------
# 10. is_vip  —  top 10% of customers by revenue
# ---------------------------------------------------------------------------
cust_stats["is_vip"] = (cust_stats["total_revenue"] >= rev_90_val).astype(int)

# ---------------------------------------------------------------------------
# 11. Select and save output columns
# ---------------------------------------------------------------------------
output_cols = [
    "CustomerID", "company_size", "vertical", "onboard_channel",
    "region", "account_manager", "credit_limit_gbp",
    "payment_terms_days", "years_in_business", "is_vip",
]

output = cust_stats[output_cols].rename(columns={"CustomerID": "customer_id"})
output.to_csv(out_path, index=False)

# ---------------------------------------------------------------------------
# 12. Print summary
# ---------------------------------------------------------------------------
print(f"\nSaved to: {out_path}")
print(f"Shape: {output.shape}")
print()
print("=== company_size distribution ===")
print(output["company_size"].value_counts().to_string())
print()
print("=== vertical distribution ===")
print(output["vertical"].value_counts().to_string())
print()
print("=== region distribution ===")
print(output["region"].value_counts().to_string())
print()
print("=== credit_limit_gbp by size ===")
print(output.groupby("company_size")["credit_limit_gbp"].describe().round(0).to_string())
print()
print("=== VIP customers ===")
print(f"  {output['is_vip'].sum()} VIP customers ({output['is_vip'].mean()*100:.1f}%)")
print()
print("Sample rows:")
print(output.head(5).to_string(index=False))
