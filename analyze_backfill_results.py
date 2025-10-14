from datetime import datetime, timedelta
from pathlib import Path
import json

#!/usr/bin/env python3
"""
Analyze Solana backfill results from progress files and calculate estimated bars
"""


def main():
    """Analyze backfill progress files."""

    print("=" * 80)
    print("SOLANA (X:SOLUSD) BACKFILL ANALYSIS")
    print("=" * 80)

    progress_dir = Path(__file__).parent / "backfill_progress"

    # Timeframe configurations with expected bars calculation
    TIMEFRAMES = {
        "1min": {"multiplier": 1, "timespan": "minute", "bars_per_day": 1440},
        "5min": {"multiplier": 5, "timespan": "minute", "bars_per_day": 288},
        "15min": {"multiplier": 15, "timespan": "minute", "bars_per_day": 96},
        "1hr": {"multiplier": 1, "timespan": "hour", "bars_per_day": 24},
        "4hr": {"multiplier": 4, "timespan": "hour", "bars_per_day": 6},
        "1day": {"multiplier": 1, "timespan": "day", "bars_per_day": 1},
    }

    ticker = "X:SOLUSD"
    months = 24
    days = months * 30  # 720 days

    print(f"\nTicker: {ticker}")
    print(f"Period: {months} months ({days} days)")
    print(f"Date range: 2023-10-22 to 2025-10-11")
    print("\n" + "-" * 80)

    total_bars = 0
    results = {}

    for tf_name, config in TIMEFRAMES.items():
        progress_file = progress_dir / f"backfill_progress_{ticker.replace(':', '_')}_{tf_name}.json"

        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)

            if ticker in progress and progress[ticker].get("completed"):
                completed_at = progress[ticker]["completed_at"]
                completed_time = datetime.fromisoformat(completed_at)

                # Calculate estimated bars (crypto markets are 24/7)
                estimated_bars = days * config["bars_per_day"]

                results[tf_name] = {
                    "completed": True,
                    "completed_at": completed_at,
                    "estimated_bars": estimated_bars,
                    "bars_per_day": config["bars_per_day"]
                }

                total_bars += estimated_bars

                print(f"✅ {tf_name:8s}: ~{estimated_bars:,} bars (estimated)")
            else:
                print(f"❌ {tf_name:8s}: Not completed")
                results[tf_name] = {"completed": False}
        else:
            print(f"❌ {tf_name:8s}: Progress file not found")
            results[tf_name] = {"completed": False}

    print("-" * 80)
    print(f"\nTotal estimated bars downloaded: ~{total_bars:,}")

    # Calculate timing
    print("\n" + "=" * 80)
    print("TIMING ANALYSIS")
    print("=" * 80)

    completion_times = []
    for tf_name, result in results.items():
        if result.get("completed"):
            completed_at = datetime.fromisoformat(result["completed_at"])
            completion_times.append((tf_name, completed_at))

    if completion_times:
        completion_times.sort(key=lambda x: x[1])

        start_time = completion_times[0][1]
        end_time = completion_times[-1][1]
        total_duration = (end_time - start_time).total_seconds()

        print(f"\nFirst timeframe completed: {completion_times[0][0]} at {completion_times[0][1].strftime('%H:%M:%S')}")
        print(f"Last timeframe completed: {completion_times[-1][0]} at {completion_times[-1][1].strftime('%H:%M:%S')}")
        print(f"Total backfill duration: {total_duration / 60:.1f} minutes ({total_duration:.0f} seconds)")
        print(f"Average bars per second: ~{total_bars / total_duration:.1f}")

        print("\nCompletion timeline:")
        for i, (tf_name, time) in enumerate(completion_times, 1):
            print(f"  {i}. {tf_name:8s} - {time.strftime('%H:%M:%S')}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    completed_count = sum(1 for r in results.values() if r.get("completed"))
    print(f"Timeframes completed: {completed_count}/6")
    print(f"Success rate: {completed_count/6*100:.0f}%")
    print(f"Estimated total bars: ~{total_bars:,}")

    # Breakdown by resolution
    print("\nData by resolution:")
    print("  Intraday (1min-15min): ~{:,} bars".format(
        sum(results[tf]["estimated_bars"] for tf in ["1min", "5min", "15min"] if results[tf].get("completed"))
    ))
    print("  Hourly (1hr-4hr): ~{:,} bars".format(
        sum(results[tf]["estimated_bars"] for tf in ["1hr", "4hr"] if results[tf].get("completed"))
    ))
    print("  Daily: ~{:,} bars".format(
        results["1day"]["estimated_bars"] if results["1day"].get("completed") else 0
    ))

    # Storage estimate
    # Assume ~200 bytes per bar (ticker, timestamp, OHLCV, vwap, trade_count, metadata)
    bytes_per_bar = 200
    total_bytes = total_bars * bytes_per_bar
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_mb / 1024

    print(f"\nEstimated storage used:")
    print(f"  ~{total_mb:.1f} MB ({total_gb:.2f} GB)")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    main()
