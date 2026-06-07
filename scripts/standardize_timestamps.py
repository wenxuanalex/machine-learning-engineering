"""
Example usage of timestamp standardization for join operations.

This script demonstrates how to:
1. Standardize timestamps across transactions, ancillary, and customer metadata
2. Prepare datasets for join operations
3. Preview available join keys
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.silver import clean_transactions, clean_customer_metadata
from utils.timestamps import (
    create_join_keys,
    standardize_ancillary_timestamps,
    standardize_customer_metadata_timestamps,
    standardize_transactions_timestamps,
)


def main():
    """Run the complete timestamp standardization pipeline."""
    print("=" * 80)
    print("TIMESTAMP STANDARDIZATION PIPELINE")
    print("=" * 80)

    # Step 1: Clean silver layer data (if not already done)
    print("\n[1/5] Cleaning transactions...")
    tx_result = clean_transactions()
    print(f"    ✓ Cleaned {tx_result['rows_out']} transaction records")

    print("\n[2/5] Cleaning customer metadata...")
    cm_result = clean_customer_metadata()
    print(f"    ✓ Cleaned customer metadata")

    # Step 2: Standardize timestamps
    print("\n[3/5] Standardizing transaction timestamps...")
    tx_ts_result = standardize_transactions_timestamps()
    print(f"    ✓ Added temporal features: {', '.join(tx_ts_result['temporal_features_added'])}")

    print("\n[4/5] Standardizing ancillary timestamps...")
    anc_ts_result = standardize_ancillary_timestamps()
    print(f"    ✓ Added temporal features: {', '.join(anc_ts_result['temporal_features_added'])}")

    print("\n[5/5] Preparing customer metadata for joins...")
    cm_ts_result = standardize_customer_metadata_timestamps()
    print(f"    ✓ Added timestamp reference: {cm_ts_result['temporal_features_added']}")

    # Step 3: Show join key summary
    print("\n" + "=" * 80)
    print("AVAILABLE JOIN KEYS FOR DOWNSTREAM OPERATIONS")
    print("=" * 80)

    join_keys = create_join_keys()

    print(f"\nTransactions Date Range: {join_keys['transactions_date_range']['min']} to {join_keys['transactions_date_range']['max']}")
    print(f"  • Total records: {join_keys['transactions_rows']:,}")

    print(f"\nAncillary Date Range: {join_keys['ancillary_date_range']['min']} to {join_keys['ancillary_date_range']['max']}")
    print(f"  • Total records: {join_keys['ancillary_rows']:,}")

    print("\nJoin Options:")
    for join_type, description in join_keys["join_keys_available"].items():
        print(f"  • {join_type}: {description}")

    print("\n" + "=" * 80)
    print("OUTPUT FILES CREATED")
    print("=" * 80)
    print("  • data/silver/transactions_timestamped.parquet")
    print("  • data/silver/ancillary_timestamped.parquet")
    print("  • data/silver/customer_metadata_timestamped.parquet")

    print("\n✓ Pipeline complete! Ready for join operations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
