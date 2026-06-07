# Timestamp Standardization Module

This module provides functions to standardize and enrich timestamps across different datasets (transactions, ancillary economic data, and customer metadata) in preparation for join operations.

## Overview

The timestamp standardization module includes:

1. **`standardize_transactions_timestamps()`** - Standardizes transaction timestamps and extracts temporal features
2. **`standardize_ancillary_timestamps()`** - Standardizes economic indicator timestamps and extracts temporal features
3. **`standardize_customer_metadata_timestamps()`** - Adds a reference timestamp for customer metadata
4. **`create_join_keys()`** - Generates a summary of available join keys across datasets

## Features

### Transaction Timestamp Standardization

- Converts `invoice_date` to proper datetime format
- Extracts temporal features:
  - `invoice_year` - Year of invoice
  - `invoice_month` - Month of invoice (1-12)
  - `invoice_day` - Day of month (1-31)
  - `invoice_dayofweek` - Day of week (0=Monday, 6=Sunday)
  - `invoice_quarter` - Quarter (1-4)
  - `invoice_week` - ISO week number
  - `invoice_date_only` - Date-only version for joining with ancillary data

**Input:** `data/silver/transactions.parquet`  
**Output:** `data/silver/transactions_timestamped.parquet`

### Ancillary Timestamp Standardization

- Converts `Date` column to lowercase `date` and proper datetime format
- Renames `Is_Holiday` to `is_holiday` (snake_case)
- Extracts temporal features:
  - `year` - Year of record
  - `month` - Month (1-12)
  - `day` - Day of month (1-31)
  - `dayofweek` - Day of week (0=Monday, 6=Sunday)
  - `quarter` - Quarter (1-4)
  - `week` - ISO week number
  - `date_only` - Date-only version for joining with transactions

**Input:** `data/bronze/ancillary.parquet`  
**Output:** `data/silver/ancillary_timestamped.parquet`

### Customer Metadata Timestamp Standardization

- Adds `metadata_snapshot_date` (2011-12-31) as a reference timestamp
- Useful for tracking when metadata was captured relative to transactions

**Input:** `data/silver/customer_metadata.parquet`  
**Output:** `data/silver/customer_metadata_timestamped.parquet`

## Usage

### Basic Usage

```python
from utils.timestamps import (
    standardize_transactions_timestamps,
    standardize_ancillary_timestamps,
    standardize_customer_metadata_timestamps,
    create_join_keys,
)

# Standardize each dataset
standardize_transactions_timestamps()
standardize_ancillary_timestamps()
standardize_customer_metadata_timestamps()

# Get available join keys
join_keys = create_join_keys()
print(join_keys)
```

### Using the Full Pipeline Script

```bash
python scripts/standardize_timestamps.py
```

This runs the complete standardization pipeline and displays:
- Processing statistics
- Date ranges for each dataset
- Available join keys
- Output file locations

## Join Operations

After standardization, you can perform joins using the available keys:

### By Date
```python
import pandas as pd

tx = pd.read_parquet('data/silver/transactions_timestamped.parquet')
anc = pd.read_parquet('data/silver/ancillary_timestamped.parquet')

# Join transactions with daily ancillary data
merged = tx.merge(
    anc,
    left_on='invoice_date_only',
    right_on='date_only',
    how='left'
)
```

### By Month
```python
# Join transactions with monthly aggregates
merged = tx.merge(
    anc,
    on=['invoice_year', 'invoice_month'],
    how='left'
)
```

### By Quarter
```python
# Join transactions with quarterly data
merged = tx.merge(
    anc,
    on=['invoice_year', 'invoice_quarter'],
    how='left'
)
```

## Data Types After Standardization

### Transactions
- `invoice_date`: `datetime64[ns]` ✓
- `invoice_year`, `invoice_month`, `invoice_day`: `int32`
- `invoice_dayofweek`, `invoice_quarter`, `invoice_week`: `int32` / `UInt32`
- `invoice_date_only`: `object` (date for joining)

### Ancillary
- `date`: `datetime64[ns]` ✓
- `year`, `month`, `day`: `int32`
- `dayofweek`, `quarter`, `week`: `int32` / `UInt32`
- `date_only`: `object` (date for joining)
- `is_holiday`: `object` (original format preserved)

### Customer Metadata
- `metadata_snapshot_date`: `object` (2011-12-31)

## Date Ranges

- **Transactions**: 2010-12-01 to 2011-12-09
- **Ancillary**: 2011-04-01 to 2011-12-30
- **Overlap**: 2011-04-01 to 2011-12-09

**Note**: Ancillary data only covers April-December 2011. Transactions from December 2010 to March 2011 will have null values when joined with ancillary data.

## Testing

Run the timestamp standardization tests:

```bash
pytest tests/test_timestamps.py -v
```

Tests verify:
- Datetime types are correct
- Temporal features are present and numeric
- Column naming follows conventions
- Original columns are properly renamed/removed

## Performance

- Standardizing ~396K transactions: < 1 second
- Standardizing ~193 ancillary records: < 1 second
- Output files are compressed parquet format
