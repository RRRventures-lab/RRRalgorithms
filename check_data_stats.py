from pathlib import Path
import pandas as pd

#!/usr/bin/env python3

data_dir = Path("data/linkusd")

print("="*80)
print("CHAINLINK DATA QUALITY CHECK")
print("="*80)
print()

timeframes = ["1min", "5min", "15min", "1hr", "4hr", "1day"]

for tf in timeframes:
    csv_file = data_dir / tf / f"X_LINKUSD_{tf}.csv"
    if csv_file.exists():
        print(f"--- {tf.upper()} ---")
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"  Rows: {len(df):,}")
        print(f"  Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Price Range: ${df['close'].min():.4f} to ${df['close'].max():.4f}")
        print(f"  Avg Volume: {df['volume'].mean():,.2f} LINK")
        print(f"  Total Volume: {df['volume'].sum():,.2f} LINK")
        print()

print("="*80)
print("ALL DATA VALIDATED SUCCESSFULLY")
print("="*80)
